"""Tests for application settings storage."""

from job_apply_ai.storage.app_settings import (
    AppSettingsRepository,
    alibaba_settings_from_form,
    llm_settings_from_form,
    normalize_alibaba_settings,
    normalize_llm_provider,
    normalize_model_providers,
    normalize_ollama_settings,
    ollama_settings_from_form,
    uses_alibaba_provider,
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
    assert full["fast_model_provider"] == "ollama"
    assert full["main_model_provider"] == "ollama"
    assert full["ollama"] == saved["ollama"]
    assert full["freellmapi"]["fast_model"] == "auto"


def test_normalize_freellmapi_settings_uses_defaults():
    from job_apply_ai.storage.app_settings import normalize_freellmapi_settings

    settings = normalize_freellmapi_settings(None)
    assert settings["base_url"] == "http://localhost:3001/v1"
    assert settings["fast_model"] == "auto"
    assert settings["main_model"] == "auto"
    assert settings["model_mode"] == "fixed"


def test_freellmapi_settings_from_form_preserves_existing_key():
    from job_apply_ai.storage.app_settings import freellmapi_settings_from_form

    settings = freellmapi_settings_from_form(
        {
            "freellmapi_api_key": "",
            "freellmapi_base_url": "http://localhost:3001/v1",
            "freellmapi_fast_model": "auto",
            "freellmapi_main_model": "auto",
        },
        existing_api_key="freellmapi-existing",
    )
    assert settings["api_key"] == "freellmapi-existing"


def test_uses_freellmapi_provider():
    from job_apply_ai.storage.app_settings import uses_freellmapi_provider

    assert uses_freellmapi_provider({"fast_model_provider": "freellmapi", "main_model_provider": "ollama"})
    assert not uses_freellmapi_provider({"fast_model_provider": "ollama", "main_model_provider": "ollama"})


def test_normalize_alibaba_settings_uses_defaults():
    settings = normalize_alibaba_settings(None)
    assert settings["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert settings["fast_model"] == "qwen-turbo"
    assert settings["main_model"] == "qwen-plus"
    assert settings["model_mode"] == "fixed"


def test_normalize_alibaba_settings_model_mode():
    settings = normalize_alibaba_settings({"model_mode": "round_robin"})
    assert settings["model_mode"] == "round_robin"
    settings = normalize_alibaba_settings({"model_mode": "invalid"})
    assert settings["model_mode"] == "fixed"


def test_normalize_alibaba_model_state():
    from job_apply_ai.storage.app_settings import normalize_alibaba_model_state

    state = normalize_alibaba_model_state(
        {
            "round_robin_index": {"fast": 2, "main": 1},
            "auto_index": {"fast": 0, "main": 3},
            "active_fast_model": "qwen-plus",
            "active_main_model": "qwen-max",
        }
    )
    assert state["round_robin_index"]["fast"] == 2
    assert state["auto_index"]["main"] == 3
    assert state["active_main_model"] == "qwen-max"


def test_save_alibaba_model_state_round_trip(monkeypatch, tmp_path):
    from job_apply_ai.storage.app_settings import normalize_alibaba_model_state

    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    repo = AppSettingsRepository()
    repo.save_alibaba_settings({"api_key": "sk-test", "model_mode": "round_robin"})
    saved = repo.save_alibaba_model_state(
        normalize_alibaba_model_state(
            {
                "round_robin_index": {"fast": 1, "main": 0},
                "active_main_model": "qwen-max",
            }
        )
    )
    assert saved["alibaba"]["model_state"]["active_main_model"] == "qwen-max"
    assert repo.get_alibaba_settings()["model_state"]["active_main_model"] == "qwen-max"


def test_ensure_alibaba_rotation_pools_expands_single_model():
    from job_apply_ai.storage.app_settings import ensure_alibaba_rotation_pools

    expanded = ensure_alibaba_rotation_pools(
        {
            "model_mode": "round_robin",
            "fast_model": "qwen-turbo",
            "main_model": "qwen-plus",
        }
    )
    assert "qwen-turbo" in expanded["fast_model"]
    assert "qwen-plus" in expanded["fast_model"]
    assert "qwen-plus" in expanded["main_model"]
    assert "qwen-max" in expanded["main_model"]


def test_save_llm_settings_preserves_alibaba_model_state(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    repo = AppSettingsRepository()
    repo.save_llm_settings(
        {
            "alibaba": {
                "api_key": "sk-test",
                "model_mode": "round_robin",
                "fast_model": "qwen-turbo, qwen-plus",
                "main_model": "qwen-plus, qwen-max",
                "model_state": {
                    "round_robin_index": {"fast": 0, "main": 2},
                    "active_main_model": "qwen-max",
                },
            },
        }
    )
    repo.save_llm_settings(
        {
            "alibaba": {
                "api_key": "sk-test",
                "model_mode": "round_robin",
                "fast_model": "qwen-turbo, qwen-plus",
                "main_model": "qwen-plus, qwen-max",
            },
        }
    )
    state = repo.get_alibaba_settings()["model_state"]
    assert state["active_main_model"] == "qwen-max"
    assert state["round_robin_index"]["main"] == 2


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
            "fast_model_provider": "ollama",
            "main_model_provider": "alibaba",
            "alibaba_api_key": "sk-test",
            "alibaba_fast_model": "qwen-turbo",
            "alibaba_main_model": "qwen-plus",
        }
    )
    assert settings["fast_model_provider"] == "ollama"
    assert settings["main_model_provider"] == "alibaba"
    assert settings["alibaba"]["api_key"] == "sk-test"


def test_normalize_model_providers_falls_back_to_legacy_provider():
    providers = normalize_model_providers({}, legacy_provider="alibaba")
    assert providers == {"fast_model_provider": "alibaba", "main_model_provider": "alibaba"}


def test_uses_alibaba_provider():
    assert uses_alibaba_provider({"fast_model_provider": "ollama", "main_model_provider": "alibaba"})
    assert not uses_alibaba_provider({"fast_model_provider": "ollama", "main_model_provider": "ollama"})


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


def test_dev_mode_round_trip(monkeypatch, tmp_path):
    from job_apply_ai.storage.app_settings import normalize_dev_mode

    assert normalize_dev_mode("on") is True
    assert normalize_dev_mode("false") is False

    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    repo = AppSettingsRepository()
    repo.save_dev_mode(True)
    assert repo.get_dev_mode() is True
    repo.save_llm_settings({"dev_mode": False})
    assert repo.get_dev_mode() is False
