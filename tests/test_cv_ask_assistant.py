"""Tests for read-only CV ask assistant."""

from unittest.mock import MagicMock

from job_apply_ai.cv_modifier.cv_ask_assistant import ASK_HISTORY_LIMIT, CVAskAssistant


def test_ask_returns_llm_reply_without_modifying_content():
    llm = MagicMock()
    llm.is_available.return_value = True
    llm.main_model = "test-model"
    llm.provider_label = "Test LLM"
    llm.validate_models.return_value = {}
    llm.generate.return_value = "You match most required skills."

    assistant = CVAskAssistant(llm=llm)
    content = {"professional_summary": "Backend engineer.", "technical_skills": ["Python"]}
    reply = assistant.ask(
        current_content=content,
        user_message="How well do I fit?",
        job={"title": "Engineer", "company": "Acme", "description": "Python required"},
        profile={"full_name": "Alex"},
        chat_history=[],
    )

    assert reply == "You match most required skills."
    llm.generate.assert_called_once()
    assert ASK_HISTORY_LIMIT == 6


def test_ask_falls_back_when_llm_returns_empty():
    llm = MagicMock()
    llm.is_available.return_value = True
    llm.main_model = "test-model"
    llm.provider_label = "Test LLM"
    llm.validate_models.return_value = {}
    llm.generate.return_value = "   "

    assistant = CVAskAssistant(llm=llm)
    reply = assistant.ask(
        current_content={"professional_summary": "Summary"},
        user_message="Any gaps?",
        job={"title": "Role", "description": "Needs Go"},
        profile={},
    )

    assert "could not find enough context" in reply.lower()
