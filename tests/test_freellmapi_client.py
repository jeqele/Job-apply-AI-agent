"""Tests for FreeLLMAPI client."""

from unittest.mock import patch

import pytest

from job_apply_ai.cv_modifier.freellmapi_client import (
    AUTO_MODEL,
    FreeLLMAPIClient,
    FreeLLMAPIError,
)


def test_resolve_model_allows_auto():
    client = FreeLLMAPIClient(api_key="freellmapi-test")
    assert client._resolve_model("auto", [], role="main", allow_unlisted=False) == AUTO_MODEL


def test_list_models_prepends_auto():
    client = FreeLLMAPIClient(api_key="freellmapi-test")
    with patch(
        "job_apply_ai.cv_modifier.freellmapi_client.requests.get",
    ) as mock_get:
        mock_get.return_value.ok = True
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {
            "data": [{"id": "gemini-2.5-flash"}, {"id": "auto"}],
        }
        models = client.list_models(refresh=True)
    assert models[0] == AUTO_MODEL
    assert "gemini-2.5-flash" in models


def test_auto_sticks_with_working_model():
    client = FreeLLMAPIClient(
        api_key="freellmapi-test",
        main_model="model-a, model-b",
        model_mode="auto",
    )

    with patch.object(client, "_generate_once", return_value="ok-a") as generate_once, patch.object(
        client, "_persist_model_state"
    ):
        assert client.generate("prompt") == "ok-a"
        assert client.main_model == "model-a"
        assert generate_once.call_args_list[0].args[0] == "model-a"


def test_auto_switches_after_error():
    client = FreeLLMAPIClient(
        api_key="freellmapi-test",
        main_model="model-a, model-b",
        model_mode="auto",
    )

    with patch.object(
        client,
        "_generate_once",
        side_effect=[
            FreeLLMAPIError("rate limited", status_code=429),
            "ok-b",
        ],
    ) as generate_once, patch.object(client, "_persist_model_state"):
        assert client.generate("prompt") == "ok-b"
        assert generate_once.call_count == 2
        assert client.main_model == "model-b"


def test_validate_models_requires_api_key():
    client = FreeLLMAPIClient(api_key="")
    with pytest.raises(RuntimeError, match="API key is not configured"):
        client.validate_models()
