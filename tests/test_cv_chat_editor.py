"""Tests for CV chat editing helpers."""

from job_apply_ai.cv_modifier.cv_chat_editor import CVChatEditor
from job_apply_ai.cv_modifier.ollama_client import OllamaClient


def test_apply_content_changes_updates_only_requested_fields():
    current = {
        "professional_title": "Backend Engineer",
        "professional_summary": "Builds APIs.",
        "technical_skills": ["Python", "SQL"],
        "experience_highlights": [],
    }
    changes = {"professional_summary": "Builds scalable APIs with Python."}

    updated = CVChatEditor._apply_content_changes(current, changes)

    assert updated["professional_summary"] == "Builds scalable APIs with Python."
    assert updated["technical_skills"] == ["Python", "SQL"]
    assert current["professional_summary"] == "Builds APIs."


def test_resolve_updated_content_prefers_changes_patch():
    current = {"professional_summary": "Old summary", "technical_skills": ["Python"]}
    result = {
        "reply": "Updated summary.",
        "changes": {"professional_summary": "New summary"},
    }

    updated = CVChatEditor._resolve_updated_content(current, result)

    assert updated["professional_summary"] == "New summary"
    assert updated["technical_skills"] == ["Python"]


def test_resolve_updated_content_supports_legacy_content_key():
    current = {"professional_summary": "Old summary"}
    replacement = {"professional_summary": "Replacement summary", "technical_skills": ["Go"]}
    result = {"reply": "Updated.", "content": replacement}

    updated = CVChatEditor._resolve_updated_content(current, result)

    assert updated == replacement


def test_parse_json_response_handles_markdown_fences():
    raw = '```json\n{"reply": "Done", "changes": {"professional_summary": "Updated"}}\n```'

    parsed = OllamaClient._parse_json_response(raw)

    assert parsed["reply"] == "Done"
    assert parsed["changes"]["professional_summary"] == "Updated"


def test_parse_json_response_extracts_embedded_object():
    raw = 'Here is the result:\n{"reply": "Done", "changes": {}}\nThanks.'

    parsed = OllamaClient._parse_json_response(raw)

    assert parsed == {"reply": "Done", "changes": {}}
