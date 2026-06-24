"""FreeLLMAPI client — OpenAI-compatible local proxy with router failover."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from job_apply_ai.cv_modifier.alibaba_client import (
    AlibabaAPIError,
    AlibabaClient,
)

logger = logging.getLogger(__name__)

AUTO_MODEL = "auto"
DEFAULT_BASE_URL = os.environ.get("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")
DEFAULT_FAST_MODEL = os.environ.get("FREELLMAPI_CV_FAST_MODEL", AUTO_MODEL)
DEFAULT_MAIN_MODEL = os.environ.get("FREELLMAPI_CV_MODEL", AUTO_MODEL)
DEFAULT_MAX_TOKENS = int(os.environ.get("FREELLMAPI_MAX_TOKENS", "8192"))
DEFAULT_MODEL_MODE = os.environ.get("FREELLMAPI_MODEL_MODE", "fixed")

MODEL_MODES = ("fixed", "round_robin", "auto")

DEFAULT_MODEL_STATE = {
    "round_robin_index": {"fast": 0, "main": 0},
    "auto_index": {"fast": 0, "main": 0},
    "active_fast_model": "",
    "active_main_model": "",
}

RESERVED_MODELS = (AUTO_MODEL,)


class FreeLLMAPIError(RuntimeError):
    """FreeLLMAPI failure that may trigger model failover."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def get_freellmapi_client() -> FreeLLMAPIClient:
    """Build a FreeLLMAPI client using saved app settings when available."""
    try:
        from job_apply_ai.storage.app_settings import AppSettingsRepository

        settings = AppSettingsRepository().get_freellmapi_settings()
        return FreeLLMAPIClient(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            fast_model=settings["fast_model"],
            main_model=settings["main_model"],
            num_predict=settings["num_predict"],
            model_mode=settings["model_mode"],
            model_state=settings.get("model_state"),
        )
    except Exception as exc:
        logger.warning("Could not load FreeLLMAPI settings from storage: %s", exc)
        return FreeLLMAPIClient()


class FreeLLMAPIClient(AlibabaClient):
    """Thin wrapper around FreeLLMAPI's OpenAI-compatible /v1 API."""

    provider_label = "FreeLLMAPI"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        fast_model: str | None = None,
        main_model: str | None = None,
        num_predict: int | None = None,
        model_mode: str | None = None,
        model_state: dict[str, Any] | None = None,
        timeout: int = 300,
    ):
        self.api_key = (api_key or os.environ.get("FREELLMAPI_API_KEY", "")).strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._fast_model_config = fast_model or DEFAULT_FAST_MODEL
        self._main_model_config = main_model or DEFAULT_MAIN_MODEL
        self.num_predict = num_predict if num_predict is not None else DEFAULT_MAX_TOKENS
        mode = (model_mode or DEFAULT_MODEL_MODE).strip().lower()
        self.model_mode = mode if mode in MODEL_MODES else "fixed"
        self.timeout = timeout
        self._available_models: list[str] | None = None
        self._apply_model_state(model_state or DEFAULT_MODEL_STATE)

    def _persist_model_state(self) -> None:
        try:
            from job_apply_ai.storage.app_settings import AppSettingsRepository

            AppSettingsRepository().save_freellmapi_model_state(self.get_model_state())
        except Exception as exc:
            logger.debug("Could not persist FreeLLMAPI model state: %s", exc)

    def list_models(self, refresh: bool = False) -> list[str]:
        if self._available_models is not None and not refresh:
            return self._prepend_reserved(self._available_models)

        if not self.api_key:
            return list(RESERVED_MODELS)

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
            self._available_models = models or list(RESERVED_MODELS)
            return self._prepend_reserved(self._available_models)
        except requests.RequestException as exc:
            logger.warning("Could not list FreeLLMAPI models: %s", exc)
            return list(RESERVED_MODELS)

    def validate_models(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(
                "FreeLLMAPI API key is not configured. "
                "Add your unified freellmapi-… key in Settings or set FREELLMAPI_API_KEY."
            )
        if not self.is_available():
            raise RuntimeError(
                f"FreeLLMAPI is not reachable at {self.base_url}. "
                "Start the FreeLLMAPI server (Docker or desktop app) and check your unified API key."
            )

        available = self.list_models(refresh=True)
        for role in ("fast", "main"):
            configured = self._model_pool(role)
            rotating = self.rotation_pool(role)
            if self.model_mode in ("round_robin", "auto") and len(rotating) < 2:
                if not (len(rotating) == 1 and rotating[0] == AUTO_MODEL):
                    logger.warning(
                        "FreeLLMAPI %s model pool has only one rotatable model (%s). "
                        "Add comma-separated models in Settings for %s rotation, or use '%s'.",
                        role,
                        ", ".join(rotating) or ", ".join(configured) or "(empty)",
                        self.model_mode,
                        AUTO_MODEL,
                    )
            for candidate in configured:
                self._resolve_model(
                    candidate,
                    available,
                    role=role,
                    allow_unlisted=candidate in configured or candidate in RESERVED_MODELS,
                )
        return {"fast": self.fast_model, "main": self.main_model}

    def _resolve_model(
        self,
        model: str,
        available: list[str],
        role: str,
        *,
        allow_unlisted: bool = False,
    ) -> str:
        if model in RESERVED_MODELS:
            return model
        return super()._resolve_model(model, available, role=role, allow_unlisted=allow_unlisted)

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
        try:
            return super()._generate_once(
                resolved_model,
                prompt,
                system=system,
                temperature=temperature,
                json_format=json_format,
                json_schema=json_schema,
                num_predict=num_predict,
            )
        except AlibabaAPIError as exc:
            raise FreeLLMAPIError(
                str(exc).replace("Alibaba Cloud", "FreeLLMAPI"),
                status_code=exc.status_code,
            ) from exc

    @staticmethod
    def _should_failover(exc: Exception, pool_size: int, attempt_offset: int) -> bool:
        if pool_size <= 1 or attempt_offset >= pool_size - 1:
            return False
        if isinstance(exc, FreeLLMAPIError) and exc.status_code == 401:
            return False
        return isinstance(exc, (FreeLLMAPIError, AlibabaAPIError, requests.RequestException))

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
        try:
            return super().generate(
                prompt,
                model=model,
                system=system,
                temperature=temperature,
                json_format=json_format,
                json_schema=json_schema,
                num_predict=num_predict,
            )
        except AlibabaAPIError as exc:
            raise FreeLLMAPIError(
                str(exc).replace("Alibaba Cloud", "FreeLLMAPI"),
                status_code=exc.status_code,
            ) from exc

    @staticmethod
    def _prepend_reserved(models: list[str]) -> list[str]:
        return list(RESERVED_MODELS) + [name for name in models if name not in RESERVED_MODELS]

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
                "FreeLLMAPI API key is invalid (401). "
                "Copy your unified freellmapi-… key from the FreeLLMAPI dashboard. "
                f"Details: {error_message}"
            )
        if response.status_code == 404:
            return (
                f"FreeLLMAPI model '{model}' was not found (404). "
                f"Use '{AUTO_MODEL}' to let the router pick, or choose a model from GET /v1/models. "
                f"Details: {error_message}"
            )
        return f"FreeLLMAPI error ({response.status_code}): {error_message}"
