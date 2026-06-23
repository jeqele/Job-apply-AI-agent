"""LLM provider factory — Ollama (local) and/or Alibaba Cloud Model Studio."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from job_apply_ai.cv_modifier.alibaba_client import AlibabaClient, get_alibaba_client
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
    )


def build_llm_client(settings: dict[str, Any]) -> LLMClient:
    """Build an LLM client from normalized app settings."""
    fast_provider = settings.get("fast_model_provider", settings.get("llm_provider", "ollama"))
    main_provider = settings.get("main_model_provider", settings.get("llm_provider", "ollama"))

    if fast_provider == main_provider:
        if fast_provider == "alibaba":
            return _build_alibaba_client(settings)
        return _build_ollama_client(settings)

    fast_client = (
        _build_alibaba_client(settings)
        if fast_provider == "alibaba"
        else _build_ollama_client(settings)
    )
    main_client = (
        _build_alibaba_client(settings)
        if main_provider == "alibaba"
        else _build_ollama_client(settings)
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
