"""Tests for application settings storage."""

from job_apply_ai.storage.app_settings import (
    AppSettingsRepository,
    normalize_ollama_settings,
    ollama_settings_from_form,
)
from job_apply_ai.storage.database import init_db


def test_normalize_ollama_settings_uses_defaults():
    settings = normalize_ollama_settings(None)
    assert settings["base_url"] == "http://localhost:11434"
    assert settings["fast_model"] == "gemma4:e4b"
    assert settings["main_model"] == "gemma4:e4b"
    assert settings["num_predict"] == 8192


def test_ollama_settings_from_form():
    settings = ollama_settings_from_form(
        {
            "ollama_base_url": "http://127.0.0.1:11434",
            "ollama_fast_model": "gemma4:e4b",
            "ollama_main_model": "gemma4:12b",
            "ollama_num_predict": "4096",
        }
    )
    assert settings["base_url"] == "http://127.0.0.1:11434"
    assert settings["fast_model"] == "gemma4:e4b"
    assert settings["main_model"] == "gemma4:12b"
    assert settings["num_predict"] == 4096


def test_app_settings_repository_round_trip(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    repo = AppSettingsRepository()
    saved = repo.save_ollama_settings(
        {
            "base_url": "http://localhost:11434",
            "fast_model": "gemma4:e4b",
            "main_model": "gemma4:12b",
            "num_predict": 6144,
        }
    )
    assert saved["main_model"] == "gemma4:12b"

    loaded = repo.get_ollama_settings()
    assert loaded == saved
