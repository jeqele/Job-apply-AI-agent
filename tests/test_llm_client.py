"""Tests for LLM provider factory."""

from unittest.mock import MagicMock, patch

from job_apply_ai.cv_modifier.alibaba_client import AlibabaClient
from job_apply_ai.cv_modifier.llm_client import get_llm_client
from job_apply_ai.cv_modifier.ollama_client import OllamaClient


def test_get_llm_client_defaults_to_ollama():
    with patch("job_apply_ai.cv_modifier.llm_client.get_ollama_client") as get_ollama:
        get_ollama.return_value = MagicMock(spec=OllamaClient)
        client = get_llm_client()
        get_ollama.assert_called_once()
        assert client is get_ollama.return_value


def test_get_llm_client_uses_alibaba_when_configured():
    settings = {
        "llm_provider": "alibaba",
        "alibaba": {
            "api_key": "sk-test",
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "fast_model": "qwen-turbo",
            "main_model": "qwen-plus",
            "num_predict": 4096,
        },
    }
    with patch("job_apply_ai.storage.app_settings.AppSettingsRepository") as repo_cls:
        repo_cls.return_value.get_settings.return_value = settings
        client = get_llm_client()
    assert isinstance(client, AlibabaClient)
    assert client.api_key == "sk-test"
    assert client.main_model == "qwen-plus"
