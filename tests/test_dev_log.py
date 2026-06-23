"""Tests for developer log storage."""

from job_apply_ai.dev_logging import dev_agent, dev_log, invalidate_dev_mode_cache
from job_apply_ai.storage.app_settings import AppSettingsRepository
from job_apply_ai.storage.database import init_db
from job_apply_ai.storage.dev_log import DevLogRepository


def test_dev_log_repository_round_trip(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    repo = DevLogRepository()
    log_id = repo.add_log(
        category="llm",
        agent="CVChatEditor",
        event="llm_request",
        message="test prompt",
        data={"prompt": "hello"},
        task_id="abc123",
        job_id=7,
    )
    assert log_id > 0

    logs = repo.list_logs(task_id="abc123")
    assert len(logs) == 1
    assert logs[0]["agent"] == "CVChatEditor"
    assert logs[0]["data"]["prompt"] == "hello"

    assert repo.count_logs(category="llm") == 1
    assert repo.clear_logs(task_id="abc123") == 1
    assert repo.count_logs() == 0


def test_dev_log_only_when_dev_mode_enabled(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    settings_repo = AppSettingsRepository()
    settings_repo.save_dev_mode(False)
    invalidate_dev_mode_cache()

    assert dev_log("system", "ignored", "should not persist") is None
    assert DevLogRepository().count_logs() == 0

    settings_repo.save_dev_mode(True)
    invalidate_dev_mode_cache()

    assert dev_log("system", "enabled", "stored") is not None
    assert DevLogRepository().count_logs() == 1


def test_log_llm_conversation_includes_chat_history(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    settings_repo = AppSettingsRepository()
    settings_repo.save_dev_mode(True)
    invalidate_dev_mode_cache()

    from job_apply_ai.dev_logging import dev_llm_context, log_llm_conversation

    history = [
        {"role": "user", "content": "Make the summary shorter"},
        {"role": "assistant", "content": "I shortened the summary."},
    ]
    with dev_llm_context(
        endpoint="POST /api/jobs/1/cv/chat",
        operation="cv_chat",
        chat_history=history,
        context={"user_message": "Add Kubernetes"},
    ):
        log_llm_conversation(
            call_type="generate_json",
            provider="Ollama",
            model="gemma4:e4b",
            system="You are a CV editor.",
            prompt="Update the CV to mention Kubernetes.",
            raw_response='{"reply":"Done","changes":{}}',
            parsed_response={"reply": "Done", "changes": {}},
        )

    logs = DevLogRepository().list_logs(category="llm")
    assert len(logs) == 1
    assert logs[0]["event"] == "conversation"
    messages = logs[0]["data"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Make the summary shorter"
    assert messages[-1]["role"] == "assistant"
    assert logs[0]["data"]["endpoint"] == "POST /api/jobs/1/cv/chat"
    assert logs[0]["data"]["extra_context"]["user_message"] == "Add Kubernetes"


def test_dev_logging_llm_client_logs_generate_json(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    settings_repo = AppSettingsRepository()
    settings_repo.save_dev_mode(True)
    invalidate_dev_mode_cache()

    from job_apply_ai.cv_modifier.llm_client import DevLoggingLLMClient
    from unittest.mock import MagicMock

    inner = MagicMock()
    inner.main_model = "test-model"
    inner.provider_label = "TestProvider"
    inner.generate.return_value = '{"ok": true}'
    inner._parse_json_response = lambda raw: {"ok": True}

    client = DevLoggingLLMClient(inner)
    result = client.generate_json("prompt", system="sys")
    assert result == {"ok": True}

    logs = DevLogRepository().list_logs(category="llm")
    assert len(logs) == 1
    assert logs[0]["event"] == "conversation"
    assert logs[0]["data"]["call_type"] == "generate_json"
    assert logs[0]["data"]["response"]["parsed"] == {"ok": True}


def test_dev_agent_context(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("JOB_APPLY_AI_DB", db_path)
    init_db(db_path)

    settings_repo = AppSettingsRepository()
    settings_repo.save_dev_mode(True)
    invalidate_dev_mode_cache()

    with dev_agent("RAGCVGenerator", task_id="task-1", job_id=3):
        dev_log("agent", "inside", "running")

    logs = DevLogRepository().list_logs(limit=10)
    agents = {entry["agent"] for entry in logs}
    assert "RAGCVGenerator" in agents
