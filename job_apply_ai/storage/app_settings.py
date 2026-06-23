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

OLLAMA_SETTING_KEYS = ("base_url", "fast_model", "main_model", "num_predict")


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


class AppSettingsRepository:
    """Persist application settings (single row, id=1)."""

    def get_settings(self) -> dict[str, Any]:
        with get_connection() as conn:
            row = conn.execute("SELECT data FROM app_settings WHERE id = 1").fetchone()
        if not row:
            return {"ollama": normalize_ollama_settings(None)}
        try:
            data = json.loads(row["data"])
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        return {
            "ollama": normalize_ollama_settings(data.get("ollama")),
        }

    def get_ollama_settings(self) -> dict[str, Any]:
        return self.get_settings()["ollama"]

    def save_ollama_settings(self, ollama_settings: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_ollama_settings(ollama_settings)
        payload = json.dumps({"ollama": normalized}, ensure_ascii=False)
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
        return normalized

    def save_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        current = self.get_settings()
        if "ollama" in data:
            current["ollama"] = normalize_ollama_settings(data["ollama"])
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
