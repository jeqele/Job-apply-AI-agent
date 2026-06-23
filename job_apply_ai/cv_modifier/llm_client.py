"""LLM provider factory — Ollama (local) or Alibaba Cloud Model Studio."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

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


def get_llm_client() -> LLMClient:
    """Return the configured LLM client (Ollama or Alibaba Cloud)."""
    try:
        from job_apply_ai.storage.app_settings import AppSettingsRepository

        settings = AppSettingsRepository().get_settings()
        provider = settings.get("llm_provider", "ollama")
        if provider == "alibaba":
            alibaba = settings["alibaba"]
            return AlibabaClient(
                api_key=alibaba["api_key"],
                base_url=alibaba["base_url"],
                fast_model=alibaba["fast_model"],
                main_model=alibaba["main_model"],
                num_predict=alibaba["num_predict"],
            )
    except Exception as exc:
        logger.warning("Could not load LLM provider from storage: %s", exc)

    return get_ollama_client()


def get_ollama_client_for_settings() -> OllamaClient:
    """Backward-compatible alias used by settings and tests."""
    return get_ollama_client()
