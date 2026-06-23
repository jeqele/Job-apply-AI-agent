"""Alibaba Cloud Model Studio client (OpenAI-compatible DashScope API)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import requests

from job_apply_ai.cv_modifier.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_FAST_MODEL = os.environ.get("ALIBABA_CV_FAST_MODEL", "qwen-turbo")
DEFAULT_MAIN_MODEL = os.environ.get("ALIBABA_CV_MODEL", "qwen-plus")
DEFAULT_MAX_TOKENS = int(os.environ.get("ALIBABA_MAX_TOKENS", "8192"))
DEFAULT_MODEL_MODE = os.environ.get("ALIBABA_MODEL_MODE", "fixed")

MODEL_MODES = ("fixed", "round_robin", "auto")

KNOWN_MODELS = (
    "qwen-turbo",
    "qwen-plus",
    "qwen-max",
    "qwen-turbo-latest",
    "qwen-plus-latest",
    "qwen-max-latest",
    "qwen3.5-plus",
    "qwen3-32b",
    "qwen3-235b-a22b",
)


class AlibabaAPIError(RuntimeError):
    """Alibaba API failure that may trigger model failover."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def parse_model_pool(value: str) -> list[str]:
    """Split a comma/newline-separated model list into unique model names."""
    models: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;\n]+", value):
        name = part.strip()
        if name and name not in seen:
            seen.add(name)
            models.append(name)
    return models


def get_alibaba_client() -> AlibabaClient:
    """Build an Alibaba client using saved app settings when available."""
    try:
        from job_apply_ai.storage.app_settings import AppSettingsRepository

        settings = AppSettingsRepository().get_alibaba_settings()
        return AlibabaClient(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            fast_model=settings["fast_model"],
            main_model=settings["main_model"],
            num_predict=settings["num_predict"],
            model_mode=settings["model_mode"],
        )
    except Exception as exc:
        logger.warning("Could not load Alibaba settings from storage: %s", exc)
        return AlibabaClient()


class AlibabaClient:
    """Thin wrapper around Alibaba Cloud Model Studio's OpenAI-compatible API."""

    provider_label = "Alibaba Cloud Model Studio"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        fast_model: str | None = None,
        main_model: str | None = None,
        num_predict: int | None = None,
        model_mode: str | None = None,
        timeout: int = 300,
    ):
        self.api_key = (api_key or os.environ.get("DASHSCOPE_API_KEY", "")).strip()
        self.base_url = (base_url or os.environ.get("ALIBABA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self._fast_model_config = fast_model or DEFAULT_FAST_MODEL
        self._main_model_config = main_model or DEFAULT_MAIN_MODEL
        self.num_predict = num_predict if num_predict is not None else DEFAULT_MAX_TOKENS
        mode = (model_mode or os.environ.get("ALIBABA_MODEL_MODE", DEFAULT_MODEL_MODE)).strip().lower()
        self.model_mode = mode if mode in MODEL_MODES else "fixed"
        self.timeout = timeout
        self._available_models: list[str] | None = None
        self._round_robin_index = {"fast": 0, "main": 0}
        self._auto_index = {"fast": 0, "main": 0}

    @property
    def fast_model(self) -> str:
        pool = self._model_pool("fast")
        if self.model_mode == "auto":
            return pool[self._auto_index["fast"] % len(pool)]
        return pool[0]

    @fast_model.setter
    def fast_model(self, value: str) -> None:
        self._fast_model_config = value

    @property
    def main_model(self) -> str:
        pool = self._model_pool("main")
        if self.model_mode == "auto":
            return pool[self._auto_index["main"] % len(pool)]
        return pool[0]

    @main_model.setter
    def main_model(self, value: str) -> None:
        self._main_model_config = value

    def _model_pool(self, role: str) -> list[str]:
        raw = self._fast_model_config if role == "fast" else self._main_model_config
        pool = parse_model_pool(raw)
        return pool or [raw.strip() or (DEFAULT_FAST_MODEL if role == "fast" else DEFAULT_MAIN_MODEL)]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=5,
            )
            return response.ok
        except requests.RequestException:
            return False

    def list_models(self, refresh: bool = False) -> list[str]:
        if self._available_models is not None and not refresh:
            return self._available_models

        if not self.api_key:
            return list(KNOWN_MODELS)

        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            models = [
                item["id"]
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
            self._available_models = models or list(KNOWN_MODELS)
            return self._available_models
        except requests.RequestException as exc:
            logger.warning("Could not list Alibaba models: %s", exc)
            return list(KNOWN_MODELS)

    def validate_models(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(
                "Alibaba Cloud API key is not configured. "
                "Add your DashScope API key in Settings or set DASHSCOPE_API_KEY."
            )
        if not self.is_available():
            raise RuntimeError(
                f"Alibaba Cloud Model Studio is not reachable at {self.base_url}. "
                "Check your API key, region endpoint, and billing in the Model Studio console."
            )

        available = self.list_models(refresh=True)
        fast_pool = [
            self._resolve_model(model, available, role="fast")
            for model in self._model_pool("fast")
        ]
        main_pool = [
            self._resolve_model(model, available, role="main")
            for model in self._model_pool("main")
        ]
        self._fast_model_config = ", ".join(fast_pool)
        self._main_model_config = ", ".join(main_pool)
        return {"fast": self.fast_model, "main": self.main_model}

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.3,
        json_format: bool = False,
        json_schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> str:
        role = self._infer_role(model)
        if self.model_mode == "fixed":
            configured = model or (self.fast_model if role == "fast" else self.main_model)
            resolved_model = self._resolve_model(
                configured,
                self.list_models(),
                role=role,
                allow_unlisted=configured in self._model_pool(role),
            )
            return self._generate_once(
                resolved_model,
                prompt,
                system=system,
                temperature=temperature,
                json_format=json_format,
                json_schema=json_schema,
                num_predict=num_predict,
            )

        pool = self._model_pool(role)
        if model and model not in pool:
            pool = [model, *[name for name in pool if name != model]]

        if self.model_mode == "round_robin":
            start_idx = self._round_robin_index[role] % len(pool)
            self._round_robin_index[role] = (start_idx + 1) % len(pool)
        else:
            start_idx = self._auto_index[role] % len(pool)

        last_error: Exception | None = None
        for offset in range(len(pool)):
            idx = (start_idx + offset) % len(pool)
            candidate = pool[idx]
            resolved_model = self._resolve_model(
                candidate,
                self.list_models(),
                role=role,
                allow_unlisted=True,
            )
            try:
                content = self._generate_once(
                    resolved_model,
                    prompt,
                    system=system,
                    temperature=temperature,
                    json_format=json_format,
                    json_schema=json_schema,
                    num_predict=num_predict,
                )
                if self.model_mode == "auto" and offset > 0:
                    self._auto_index[role] = idx
                    logger.info(
                        "Alibaba auto mode switched %s model to '%s' after error",
                        role,
                        resolved_model,
                    )
                return content
            except Exception as exc:
                if not self._should_failover(exc, len(pool), offset):
                    raise
                last_error = exc
                logger.warning(
                    "Alibaba %s model '%s' failed (%s); trying next model",
                    role,
                    resolved_model,
                    exc,
                )

        if last_error:
            raise last_error
        raise RuntimeError("Alibaba Cloud returned no model candidates")

    def _generate_once(
        self,
        resolved_model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        json_format: bool = False,
        json_schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": num_predict if num_predict is not None else self.num_predict,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        elif json_format:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise AlibabaAPIError(f"Alibaba Cloud request failed: {exc}") from exc

        if not response.ok:
            raise AlibabaAPIError(
                self._format_api_error(response, resolved_model),
                status_code=response.status_code,
            )

        choices = response.json().get("choices") or []
        if not choices:
            raise AlibabaAPIError("Alibaba Cloud returned an empty response")
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise AlibabaAPIError("Alibaba Cloud returned an empty response")
        return content

    def generate_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_attempts: int = 2,
        schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> dict[str, Any]:
        json_system = (system or "") + " Return only a single valid JSON object."
        last_error: Exception | None = None
        last_raw = ""

        for attempt in range(max_attempts):
            attempt_prompt = prompt
            if attempt > 0:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "Your previous answer was not valid JSON. "
                    "Reply again with ONLY one JSON object. "
                    "Use double quotes for all keys and strings. "
                    "Do not include markdown fences, comments, trailing commas, or prose."
                )

            raw = self.generate(
                attempt_prompt,
                model=model,
                system=json_system.strip(),
                temperature=max(temperature - (attempt * 0.05), 0.05),
                json_format=schema is None,
                json_schema=schema,
                num_predict=num_predict,
            )
            last_raw = raw
            try:
                return OllamaClient._parse_json_response(raw)
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "Failed to parse Alibaba JSON on attempt %s/%s: %s. Raw response preview: %r",
                    attempt + 1,
                    max_attempts,
                    exc,
                    raw[:500],
                )

        if last_raw:
            logger.error(
                "Alibaba JSON parse failed after %s attempts. Last raw preview: %r",
                max_attempts,
                last_raw[:1000],
            )
        raise ValueError(str(last_error) if last_error else "Model response was not valid JSON")

    def _infer_role(self, model: str | None) -> str:
        if model is None:
            return "main"

        fast_pool = self._model_pool("fast")
        if model in fast_pool:
            return "fast"

        main_pool = self._model_pool("main")
        if model in main_pool:
            return "main"

        if model == self.fast_model:
            return "fast"
        if model == self.main_model:
            return "main"
        return "main"

    @staticmethod
    def _should_failover(exc: Exception, pool_size: int, attempt_offset: int) -> bool:
        if pool_size <= 1 or attempt_offset >= pool_size - 1:
            return False
        if isinstance(exc, AlibabaAPIError) and exc.status_code == 401:
            return False
        return isinstance(exc, (AlibabaAPIError, requests.RequestException))

    def _resolve_model(
        self,
        model: str,
        available: list[str],
        role: str,
        *,
        allow_unlisted: bool = False,
    ) -> str:
        if model in available:
            return model

        model_base = model.split(":", 1)[0]
        for name in available:
            if name.split(":", 1)[0] == model_base:
                logger.warning(
                    "Alibaba %s model '%s' not found; using '%s' instead",
                    role,
                    model,
                    name,
                )
                return name

        if allow_unlisted or not available:
            return model

        available_text = ", ".join(available)
        raise RuntimeError(
            f"Alibaba model '{model}' is not available for {role} generation. "
            f"Available models: {available_text}."
        )

    @staticmethod
    def _format_api_error(response: requests.Response, model: str) -> str:
        try:
            payload = response.json()
            error = payload.get("error") or {}
            if isinstance(error, dict):
                error_message = error.get("message") or error
            else:
                error_message = error or payload
        except ValueError:
            error_message = response.text.strip() or response.reason

        if response.status_code == 401:
            return (
                "Alibaba Cloud API key is invalid or expired (401). "
                "Create or refresh your key in the Model Studio console. "
                f"Details: {error_message}"
            )
        if response.status_code == 404:
            return (
                f"Alibaba model '{model}' was not found (404). "
                f"Details: {error_message}"
            )
        return f"Alibaba Cloud API error ({response.status_code}): {error_message}"
