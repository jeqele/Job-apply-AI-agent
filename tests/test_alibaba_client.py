"""Tests for Alibaba client model rotation and failover."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from job_apply_ai.cv_modifier.alibaba_client import (
    AlibabaAPIError,
    AlibabaClient,
    parse_model_pool,
)


def test_parse_model_pool_splits_comma_separated_values():
    assert parse_model_pool("qwen-turbo, qwen-plus, qwen-max") == [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
    ]


def test_parse_model_pool_deduplicates():
    assert parse_model_pool("qwen-plus, qwen-plus, qwen-max") == ["qwen-plus", "qwen-max"]


def test_round_robin_rotates_model_each_request():
    client = AlibabaClient(
        api_key="sk-test",
        fast_model="model-a, model-b, model-c",
        main_model="main-a",
        model_mode="round_robin",
    )

    with patch.object(client, "_generate_once", side_effect=["ok-a", "ok-b", "ok-c"]) as generate_once:
        assert client.generate("prompt", model=client.fast_model) == "ok-a"
        assert client.generate("prompt", model=client.fast_model) == "ok-b"
        assert client.generate("prompt", model=client.fast_model) == "ok-c"
        assert generate_once.call_args_list[0].args[0] == "model-a"
        assert generate_once.call_args_list[1].args[0] == "model-b"
        assert generate_once.call_args_list[2].args[0] == "model-c"


def test_round_robin_failover_on_error():
    client = AlibabaClient(
        api_key="sk-test",
        fast_model="model-a, model-b",
        main_model="main-a",
        model_mode="round_robin",
    )

    with patch.object(
        client,
        "_generate_once",
        side_effect=[
            AlibabaAPIError("rate limited", status_code=429),
            "ok-b",
        ],
    ) as generate_once:
        assert client.generate("prompt", model=client.fast_model) == "ok-b"
        assert generate_once.call_count == 2
        assert generate_once.call_args_list[0].args[0] == "model-a"
        assert generate_once.call_args_list[1].args[0] == "model-b"


def test_auto_sticks_with_working_model():
    client = AlibabaClient(
        api_key="sk-test",
        main_model="model-a, model-b",
        model_mode="auto",
    )

    with patch.object(client, "_generate_once", return_value="ok") as generate_once:
        assert client.generate("prompt") == "ok"
        assert client.generate("prompt") == "ok"
        assert generate_once.call_args_list[0].args[0] == "model-a"
        assert generate_once.call_args_list[1].args[0] == "model-a"


def test_auto_switches_after_error_and_stays_on_new_model():
    client = AlibabaClient(
        api_key="sk-test",
        main_model="model-a, model-b, model-c",
        model_mode="auto",
    )

    with patch.object(
        client,
        "_generate_once",
        side_effect=[
            "ok-a",
            AlibabaAPIError("server error", status_code=500),
            "ok-b",
            "ok-b-again",
        ],
    ) as generate_once:
        assert client.generate("prompt") == "ok-a"
        assert client.generate("prompt") == "ok-b"
        assert client.generate("prompt") == "ok-b-again"
        assert generate_once.call_args_list[2].args[0] == "model-b"
        assert generate_once.call_args_list[3].args[0] == "model-b"


def test_auto_does_not_failover_on_401():
    client = AlibabaClient(
        api_key="sk-test",
        main_model="model-a, model-b",
        model_mode="auto",
    )

    with patch.object(
        client,
        "_generate_once",
        side_effect=AlibabaAPIError("invalid key", status_code=401),
    ):
        with pytest.raises(AlibabaAPIError):
            client.generate("prompt")


def test_fixed_mode_uses_single_model_without_rotation():
    client = AlibabaClient(
        api_key="sk-test",
        main_model="model-a, model-b",
        model_mode="fixed",
    )

    with patch.object(client, "_generate_once", return_value="ok") as generate_once:
        client.generate("prompt")
        client.generate("prompt")
        assert generate_once.call_count == 2
        assert generate_once.call_args_list[0].args[0] == "model-a"
        assert generate_once.call_args_list[1].args[0] == "model-a"


def test_generate_once_raises_alibaba_api_error_on_http_failure():
    client = AlibabaClient(api_key="sk-test")
    response = MagicMock(spec=requests.Response)
    response.ok = False
    response.status_code = 403
    response.json.return_value = {"error": {"message": "forbidden"}}
    response.text = ""
    response.reason = "Forbidden"

    with patch("job_apply_ai.cv_modifier.alibaba_client.requests.post", return_value=response):
        with pytest.raises(AlibabaAPIError) as exc_info:
            client._generate_once("qwen-plus", "hello")
        assert exc_info.value.status_code == 403
