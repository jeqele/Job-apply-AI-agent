"""Tests for LLM provider factory."""

from unittest.mock import MagicMock, patch

from job_apply_ai.cv_modifier.alibaba_client import AlibabaClient
from job_apply_ai.cv_modifier.llm_client import (
    CompositeLLMClient,
    build_llm_client,
    get_llm_client,
)
from job_apply_ai.cv_modifier.ollama_client import OllamaClient


def _base_settings(**overrides):
    settings = {
        "llm_provider": "ollama",
        "fast_model_provider": "ollama",
        "main_model_provider": "ollama",
        "ollama": {
            "base_url": "http://localhost:11434",
            "fast_model": "gemma4:e4b",
            "main_model": "gemma4:12b",
            "num_predict": 8192,
        },
        "alibaba": {
            "api_key": "sk-test",
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "fast_model": "qwen-turbo",
            "main_model": "qwen-plus",
            "num_predict": 4096,
        },
    }
    settings.update(overrides)
    return settings


def test_get_llm_client_defaults_to_ollama():
    with patch("job_apply_ai.storage.app_settings.AppSettingsRepository") as repo_cls:
        repo_cls.side_effect = RuntimeError("no settings")
        with patch("job_apply_ai.cv_modifier.llm_client.get_ollama_client") as get_ollama:
            get_ollama.return_value = MagicMock(spec=OllamaClient)
            client = get_llm_client()
            get_ollama.assert_called_once()
            assert client is get_ollama.return_value


def test_build_llm_client_uses_alibaba_when_both_providers_alibaba():
    settings = _base_settings(
        llm_provider="alibaba",
        fast_model_provider="alibaba",
        main_model_provider="alibaba",
    )
    client = build_llm_client(settings)
    assert isinstance(client, AlibabaClient)
    assert client.api_key == "sk-test"
    assert client.main_model == "qwen-plus"


def test_build_llm_client_mixed_ollama_fast_alibaba_main():
    settings = _base_settings(
        fast_model_provider="ollama",
        main_model_provider="alibaba",
    )
    client = build_llm_client(settings)
    assert isinstance(client, CompositeLLMClient)
    assert client.fast_model == "gemma4:e4b"
    assert client.main_model == "qwen-plus"
    assert "Ollama" in client.provider_label
    assert "Alibaba" in client.provider_label


def test_composite_routes_generate_to_correct_provider():
    fast = MagicMock(spec=OllamaClient)
    fast.fast_model = "fast-local"
    fast.main_model = "main-local"
    fast.num_predict = 4096
    fast.provider_label = "Ollama"
    fast.generate.return_value = "fast-response"

    main = MagicMock(spec=AlibabaClient)
    main.fast_model = "qwen-turbo"
    main.main_model = "qwen-plus"
    main.num_predict = 8192
    main.provider_label = "Alibaba Cloud Model Studio"
    main.generate.return_value = "main-response"

    client = CompositeLLMClient(fast, main)
    assert client.generate("prompt", model=client.fast_model) == "fast-response"
    assert client.generate("prompt", model=client.main_model) == "main-response"
    fast.generate.assert_called_once()
    main.generate.assert_called_once()


def test_get_llm_client_uses_alibaba_when_configured():
    settings = _base_settings(
        llm_provider="alibaba",
        fast_model_provider="alibaba",
        main_model_provider="alibaba",
    )
    with patch("job_apply_ai.storage.app_settings.AppSettingsRepository") as repo_cls:
        repo_cls.return_value.get_settings.return_value = settings
        client = get_llm_client()
    assert isinstance(client, AlibabaClient)
    assert client.api_key == "sk-test"
    assert client.main_model == "qwen-plus"
