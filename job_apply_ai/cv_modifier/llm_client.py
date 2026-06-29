"""LLM provider factory — Ollama, Alibaba Cloud, and FreeLLMAPI."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from job_apply_ai.cv_modifier.alibaba_client import AlibabaClient, get_alibaba_client
from job_apply_ai.cv_modifier.freellmapi_client import FreeLLMAPIClient
from job_apply_ai.cv_modifier.ollama_client import OllamaClient, get_ollama_client

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Shared interface implemented by Ollama and Alibaba clients."""

    fast_model: str
    main_model: str
    num_predict: int
    provider_label: str

    def is_available(self) -> bool: ...

    def list_models(self, refresh: bool = False) -> list[str]: ...

    def validate_models(self) -> dict[str, str]: ...

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.3,
        json_format: bool = False,
        json_schema: dict | None = None,
        num_predict: int | None = None,
    ) -> str: ...

    def generate_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_attempts: int = 2,
        schema: dict | None = None,
        num_predict: int | None = None,
    ) -> dict: ...


class DevLoggingLLMClient:
    """Wraps an LLM client to log conversations when developer mode is on."""

    def __init__(self, client: LLMClient):
        self._client = client

    @property
    def fast_model(self) -> str:
        return self._client.fast_model

    @property
    def main_model(self) -> str:
        return self._client.main_model

    @property
    def num_predict(self) -> int:
        return self._client.num_predict

    @property
    def provider_label(self) -> str:
        return self._client.provider_label

    def is_available(self) -> bool:
        return self._client.is_available()

    def list_models(self, refresh: bool = False) -> list[str]:
        return self._client.list_models(refresh=refresh)

    def validate_models(self) -> dict[str, str]:
        return self._client.validate_models()

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        parse = getattr(self._client, "_parse_json_response", None)
        if callable(parse):
            return parse(raw)
        if not raw or not raw.strip():
            raise ValueError("Model returned an empty response")
        from job_apply_ai.cv_modifier.ollama_client import OllamaClient

        return OllamaClient._parse_json_response(raw)

    def _generate_raw(
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
        return self._client.generate(
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            json_format=json_format,
            json_schema=json_schema,
            num_predict=num_predict,
        )

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
        from job_apply_ai.dev_logging import log_llm_conversation

        resolved_model = model or self._client.main_model
        response = self._generate_raw(
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            json_format=json_format,
            json_schema=json_schema,
            num_predict=num_predict,
        )
        log_llm_conversation(
            call_type="generate",
            provider=self._client.provider_label,
            model=resolved_model,
            system=system,
            prompt=prompt,
            raw_response=response,
            temperature=temperature,
            schema=json_schema,
            json_format=json_format,
        )
        return response

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
    ) -> dict:
        from job_apply_ai.dev_logging import log_llm_conversation

        resolved_model = model or self._client.main_model
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

            raw = self._generate_raw(
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
                parsed = self._parse_json_response(raw)
                log_llm_conversation(
                    call_type="generate_json",
                    provider=self._client.provider_label,
                    model=resolved_model,
                    system=system,
                    prompt=attempt_prompt,
                    raw_response=raw,
                    parsed_response=parsed,
                    temperature=temperature,
                    schema=schema,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
                return parsed
            except ValueError as exc:
                last_error = exc
                log_llm_conversation(
                    call_type="generate_json",
                    provider=self._client.provider_label,
                    model=resolved_model,
                    system=system,
                    prompt=attempt_prompt,
                    raw_response=raw,
                    temperature=temperature,
                    schema=schema,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
                logger.warning(
                    "Failed to parse JSON on attempt %s/%s via %s: %s",
                    attempt + 1,
                    max_attempts,
                    self._client.provider_label,
                    exc,
                )

        raise ValueError(str(last_error) if last_error else "Model response was not valid JSON")


class CompositeLLMClient:
    """Routes fast and main model calls to different LLM providers."""

    def __init__(self, fast_client: LLMClient, main_client: LLMClient):
        self._fast = fast_client
        self._main = main_client

    @property
    def fast_model(self) -> str:
        return self._fast.fast_model

    @property
    def main_model(self) -> str:
        return self._main.main_model

    @property
    def num_predict(self) -> int:
        return max(self._fast.num_predict, self._main.num_predict)

    @property
    def provider_label(self) -> str:
        if self._fast is self._main:
            return self._main.provider_label
        return f"{self._fast.provider_label} (fast) + {self._main.provider_label} (main)"

    def is_available(self) -> bool:
        return self._fast.is_available() and self._main.is_available()

    def list_models(self, refresh: bool = False) -> list[str]:
        models = self._fast.list_models(refresh=refresh) + self._main.list_models(refresh=refresh)
        return list(dict.fromkeys(models))

    def validate_models(self) -> dict[str, str]:
        fast = self._fast.validate_models()
        main = self._main.validate_models()
        return {"fast": fast["fast"], "main": main["main"]}

    def _client_for_model(self, model: str | None) -> LLMClient:
        if model == self.fast_model:
            return self._fast
        if model == self.main_model:
            return self._main
        return self._main

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
        client = self._client_for_model(model)
        return client.generate(
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            json_format=json_format,
            json_schema=json_schema,
            num_predict=num_predict,
        )

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
    ) -> dict:
        client = self._client_for_model(model)
        return client.generate_json(
            prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_attempts=max_attempts,
            schema=schema,
            num_predict=num_predict,
        )


def _build_ollama_client(settings: dict[str, Any]) -> OllamaClient:
    ollama = settings["ollama"]
    return OllamaClient(
        base_url=ollama["base_url"],
        fast_model=ollama["fast_model"],
        main_model=ollama["main_model"],
        num_predict=ollama["num_predict"],
    )


def _build_alibaba_client(settings: dict[str, Any]) -> AlibabaClient:
    alibaba = settings["alibaba"]
    return AlibabaClient(
        api_key=alibaba["api_key"],
        base_url=alibaba["base_url"],
        fast_model=alibaba["fast_model"],
        main_model=alibaba["main_model"],
        num_predict=alibaba["num_predict"],
        model_mode=alibaba["model_mode"],
        model_state=alibaba.get("model_state"),
    )


def _build_freellmapi_client(settings: dict[str, Any]) -> FreeLLMAPIClient:
    freellmapi = settings["freellmapi"]
    return FreeLLMAPIClient(
        api_key=freellmapi["api_key"],
        base_url=freellmapi["base_url"],
        fast_model=freellmapi["fast_model"],
        main_model=freellmapi["main_model"],
        num_predict=freellmapi["num_predict"],
        model_mode=freellmapi["model_mode"],
        model_state=freellmapi.get("model_state"),
    )


def _build_client_for_provider(provider: str, settings: dict[str, Any]) -> LLMClient:
    if provider == "alibaba":
        return _build_alibaba_client(settings)
    if provider == "freellmapi":
        return _build_freellmapi_client(settings)
    return _build_ollama_client(settings)


def _maybe_wrap_dev_logging(client: LLMClient, settings: dict[str, Any]) -> LLMClient:
    if settings.get("dev_mode"):
        return DevLoggingLLMClient(client)
    return client


def build_llm_client(settings: dict[str, Any]) -> LLMClient:
    """Build an LLM client from normalized app settings."""
    fast_provider = settings.get("fast_model_provider", settings.get("llm_provider", "ollama"))
    main_provider = settings.get("main_model_provider", settings.get("llm_provider", "ollama"))

    if fast_provider == main_provider:
        client: LLMClient = _build_client_for_provider(fast_provider, settings)
        return _maybe_wrap_dev_logging(client, settings)

    fast_client = _maybe_wrap_dev_logging(
        _build_client_for_provider(fast_provider, settings),
        settings,
    )
    main_client = _maybe_wrap_dev_logging(
        _build_client_for_provider(main_provider, settings),
        settings,
    )
    return CompositeLLMClient(fast_client, main_client)


def get_llm_client() -> LLMClient:
    """Return the configured LLM client (single or multi-provider)."""
    try:
        from job_apply_ai.storage.app_settings import AppSettingsRepository

        settings = AppSettingsRepository().get_settings()
        return build_llm_client(settings)
    except Exception as exc:
        logger.warning("Could not load LLM provider from storage: %s", exc)

    return get_ollama_client()


def get_ollama_client_for_settings() -> OllamaClient:
    """Backward-compatible alias used by settings and tests."""
    return get_ollama_client()
