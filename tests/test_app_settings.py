"""Tests for application settings storage."""

from job_apply_ai.storage.app_settings import (
    AppSettingsRepository,
    alibaba_settings_from_form,
    llm_settings_from_form,
    normalize_alibaba_settings,
    normalize_llm_provider,
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
    assert saved["ollama"]["main_model"] == "gemma4:12b"

    loaded = repo.get_ollama_settings()
    assert loaded == saved["ollama"]

    full = repo.get_settings()
    assert full["llm_provider"] == "ollama"
    assert full["ollama"] == saved["ollama"]


def test_normalize_alibaba_settings_uses_defaults():
    settings = normalize_alibaba_settings(None)
    assert settings["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert settings["fast_model"] == "qwen-turbo"
    assert settings["main_model"] == "qwen-plus"


def test_alibaba_settings_from_form_preserves_existing_key():
    settings = alibaba_settings_from_form(
        {
            "alibaba_api_key": "",
            "alibaba_base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "alibaba_fast_model": "qwen-turbo",
            "alibaba_main_model": "qwen-max",
            "alibaba_num_predict": "4096",
        },
        existing_api_key="sk-existing",
    )
    assert settings["api_key"] == "sk-existing"
    assert settings["main_model"] == "qwen-max"


def test_llm_settings_from_form():
    settings = llm_settings_from_form(
        {
            "llm_provider": "alibaba",
            "alibaba_api_key": "sk-test",
            "alibaba_fast_model": "qwen-turbo",
            "alibaba_main_model": "qwen-plus",
        }
    )
    assert normalize_llm_provider(settings["llm_provider"]) == "alibaba"
    assert settings["alibaba"]["api_key"] == "sk-test"


def test_save_llm_settings_preserves_alibaba_api_key(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    repo = AppSettingsRepository()
    repo.save_llm_settings(
        {
            "llm_provider": "alibaba",
            "alibaba": {
                "api_key": "sk-secret",
                "fast_model": "qwen-turbo",
                "main_model": "qwen-plus",
            },
        }
    )
    repo.save_llm_settings(
        {
            "llm_provider": "alibaba",
            "alibaba": {
                "api_key": "",
                "fast_model": "qwen-turbo",
                "main_model": "qwen-plus",
            },
        }
    )
    assert repo.get_alibaba_settings()["api_key"] == "sk-secret"
