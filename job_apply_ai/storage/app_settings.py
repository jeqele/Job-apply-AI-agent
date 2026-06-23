"""Application-wide settings persisted in SQLite."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from job_apply_ai.storage.database import get_connection

DEFAULT_OLLAMA_SETTINGS: dict[str, Any] = {
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "fast_model": os.environ.get("OLLAMA_CV_FAST_MODEL", "gemma4:e4b"),
    "main_model": os.environ.get("OLLAMA_CV_MODEL", "gemma4:e4b"),
    "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "8192")),
}

DEFAULT_ALIBABA_SETTINGS: dict[str, Any] = {
    "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
    "base_url": os.environ.get(
        "ALIBABA_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    ),
    "fast_model": os.environ.get("ALIBABA_CV_FAST_MODEL", "qwen-turbo"),
    "main_model": os.environ.get("ALIBABA_CV_MODEL", "qwen-plus"),
    "num_predict": int(os.environ.get("ALIBABA_MAX_TOKENS", "8192")),
}

DEFAULT_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
DEFAULT_FAST_MODEL_PROVIDER = os.environ.get("LLM_FAST_PROVIDER", DEFAULT_LLM_PROVIDER)
DEFAULT_MAIN_MODEL_PROVIDER = os.environ.get("LLM_MAIN_PROVIDER", DEFAULT_LLM_PROVIDER)

OLLAMA_SETTING_KEYS = ("base_url", "fast_model", "main_model", "num_predict")
ALIBABA_SETTING_KEYS = ("api_key", "base_url", "fast_model", "main_model", "num_predict")
LLM_PROVIDERS = ("ollama", "alibaba")


def normalize_ollama_settings(data: dict[str, Any] | None) -> dict[str, Any]:
    """Merge stored Ollama settings with defaults."""
    settings = deepcopy(DEFAULT_OLLAMA_SETTINGS)
    if not data:
        return settings

    base_url = str(data.get("base_url") or "").strip()
    if base_url:
        settings["base_url"] = base_url.rstrip("/")

    fast_model = str(data.get("fast_model") or "").strip()
    if fast_model:
        settings["fast_model"] = fast_model

    main_model = str(data.get("main_model") or "").strip()
    if main_model:
        settings["main_model"] = main_model

    try:
        settings["num_predict"] = max(256, int(data.get("num_predict", settings["num_predict"])))
    except (TypeError, ValueError):
        pass

    return settings


def normalize_alibaba_settings(data: dict[str, Any] | None) -> dict[str, Any]:
    """Merge stored Alibaba Cloud settings with defaults."""
    settings = deepcopy(DEFAULT_ALIBABA_SETTINGS)
    if not data:
        return settings

    api_key = str(data.get("api_key") or "").strip()
    if api_key:
        settings["api_key"] = api_key

    base_url = str(data.get("base_url") or "").strip()
    if base_url:
        settings["base_url"] = base_url.rstrip("/")

    fast_model = str(data.get("fast_model") or "").strip()
    if fast_model:
        settings["fast_model"] = fast_model

    main_model = str(data.get("main_model") or "").strip()
    if main_model:
        settings["main_model"] = main_model

    try:
        settings["num_predict"] = max(256, int(data.get("num_predict", settings["num_predict"])))
    except (TypeError, ValueError):
        pass

    return settings


def normalize_llm_provider(provider: str | None) -> str:
    value = str(provider or DEFAULT_LLM_PROVIDER).strip().lower()
    return value if value in LLM_PROVIDERS else DEFAULT_LLM_PROVIDER


def normalize_model_providers(
    data: dict[str, Any] | None,
    *,
    legacy_provider: str | None = None,
) -> dict[str, str]:
    """Resolve fast/main model providers, falling back to legacy llm_provider."""
    fallback = normalize_llm_provider(legacy_provider)
    if data is None:
        return {
            "fast_model_provider": normalize_llm_provider(DEFAULT_FAST_MODEL_PROVIDER),
            "main_model_provider": normalize_llm_provider(DEFAULT_MAIN_MODEL_PROVIDER),
        }

    fast = normalize_llm_provider(data.get("fast_model_provider") or fallback)
    main = normalize_llm_provider(data.get("main_model_provider") or fallback)
    return {"fast_model_provider": fast, "main_model_provider": main}


def uses_alibaba_provider(providers: dict[str, str]) -> bool:
    return (
        providers["fast_model_provider"] == "alibaba"
        or providers["main_model_provider"] == "alibaba"
    )


def ollama_settings_from_form(form_data: Any) -> dict[str, Any]:
    """Build Ollama settings from a submitted settings form."""
    if hasattr(form_data, "to_dict"):
        scalar_data = form_data.to_dict()
    else:
        scalar_data = dict(form_data)

    return normalize_ollama_settings(
        {
            "base_url": scalar_data.get("ollama_base_url", ""),
            "fast_model": scalar_data.get("ollama_fast_model", ""),
            "main_model": scalar_data.get("ollama_main_model", ""),
            "num_predict": scalar_data.get("ollama_num_predict", ""),
        }
    )


def alibaba_settings_from_form(
    form_data: Any,
    *,
    existing_api_key: str = "",
) -> dict[str, Any]:
    """Build Alibaba settings from a submitted settings form."""
    if hasattr(form_data, "to_dict"):
        scalar_data = form_data.to_dict()
    else:
        scalar_data = dict(form_data)

    api_key = str(scalar_data.get("alibaba_api_key") or "").strip()
    if not api_key:
        api_key = existing_api_key

    return normalize_alibaba_settings(
        {
            "api_key": api_key,
            "base_url": scalar_data.get("alibaba_base_url", ""),
            "fast_model": scalar_data.get("alibaba_fast_model", ""),
            "main_model": scalar_data.get("alibaba_main_model", ""),
            "num_predict": scalar_data.get("alibaba_num_predict", ""),
        }
    )


def llm_settings_from_form(
    form_data: Any,
    *,
    existing_alibaba_api_key: str = "",
) -> dict[str, Any]:
    """Build full LLM settings from a submitted settings form."""
    if hasattr(form_data, "to_dict"):
        scalar_data = form_data.to_dict()
    else:
        scalar_data = dict(form_data)

    legacy_provider = normalize_llm_provider(scalar_data.get("llm_provider"))
    providers = normalize_model_providers(
        {
            "fast_model_provider": scalar_data.get("fast_model_provider"),
            "main_model_provider": scalar_data.get("main_model_provider"),
        },
        legacy_provider=legacy_provider,
    )

    return {
        "llm_provider": legacy_provider,
        **providers,
        "ollama": ollama_settings_from_form(form_data),
        "alibaba": alibaba_settings_from_form(
            form_data,
            existing_api_key=existing_alibaba_api_key,
        ),
    }


class AppSettingsRepository:
    """Persist application settings (single row, id=1)."""

    def get_settings(self) -> dict[str, Any]:
        with get_connection() as conn:
            row = conn.execute("SELECT data FROM app_settings WHERE id = 1").fetchone()
        if not row:
            return self._default_settings()
        try:
            data = json.loads(row["data"])
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        legacy_provider = normalize_llm_provider(data.get("llm_provider"))
        providers = normalize_model_providers(data, legacy_provider=legacy_provider)
        return {
            "llm_provider": legacy_provider,
            **providers,
            "ollama": normalize_ollama_settings(data.get("ollama")),
            "alibaba": normalize_alibaba_settings(data.get("alibaba")),
        }

    def _default_settings(self) -> dict[str, Any]:
        providers = normalize_model_providers(None)
        return {
            "llm_provider": normalize_llm_provider(None),
            **providers,
            "ollama": normalize_ollama_settings(None),
            "alibaba": normalize_alibaba_settings(None),
        }

    def get_ollama_settings(self) -> dict[str, Any]:
        return self.get_settings()["ollama"]

    def get_alibaba_settings(self) -> dict[str, Any]:
        return self.get_settings()["alibaba"]

    def get_llm_provider(self) -> str:
        return self.get_settings()["llm_provider"]

    def save_ollama_settings(self, ollama_settings: dict[str, Any]) -> dict[str, Any]:
        current = self.get_settings()
        current["ollama"] = normalize_ollama_settings(ollama_settings)
        return self._persist(current)

    def save_alibaba_settings(self, alibaba_settings: dict[str, Any]) -> dict[str, Any]:
        current = self.get_settings()
        current["alibaba"] = normalize_alibaba_settings(alibaba_settings)
        return self._persist(current)

    def save_llm_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        current = self.get_settings()
        if "llm_provider" in data:
            current["llm_provider"] = normalize_llm_provider(data["llm_provider"])
        if "fast_model_provider" in data or "main_model_provider" in data:
            providers = normalize_model_providers(
                {
                    "fast_model_provider": data.get("fast_model_provider", current.get("fast_model_provider")),
                    "main_model_provider": data.get("main_model_provider", current.get("main_model_provider")),
                },
                legacy_provider=current.get("llm_provider"),
            )
            current.update(providers)
        if "ollama" in data:
            current["ollama"] = normalize_ollama_settings(data["ollama"])
        if "alibaba" in data:
            incoming = normalize_alibaba_settings(data["alibaba"])
            if not incoming.get("api_key") and current["alibaba"].get("api_key"):
                incoming["api_key"] = current["alibaba"]["api_key"]
            current["alibaba"] = incoming
        return self._persist(current)

    def save_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.save_llm_settings(data)

    def _persist(self, current: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(current, ensure_ascii=False)
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (id, data, updated_at)
                VALUES (1, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                    data = excluded.data,
                    updated_at = datetime('now')
                """,
                (payload,),
            )
        return current
