"""Chat-based editing of tailored CV content."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from job_apply_ai.cv_modifier.chat_context import (
    build_job_context,
    cv_content_to_preview_lines,
    format_numbered_cv_preview,
)
from job_apply_ai.cv_modifier.cv_generator import RAGCVGenerator
from job_apply_ai.cv_modifier.docx_builder import CVDocumentBuilder
from job_apply_ai.cv_modifier.ollama_client import OllamaClient, get_ollama_client
from job_apply_ai.storage.user_profile import get_default_cv_template_path

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = (
    "You are a professional CV editing assistant. Apply the user's requested changes to "
    "the supplied CV content. Never invent employers, dates, degrees, certifications, "
    "achievements, or skills that are not already supported by the current content or the "
    "user's explicit instruction. Keep language concise and ATS-friendly. Return valid JSON only."
)

CV_CHAT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "changes": {
            "type": "object",
            "additionalProperties": True,
        },
    },
    "required": ["reply", "changes"],
}

CONTENT_CHANGE_KEYS = frozenset({
    "professional_title",
    "professional_summary",
    "job_matched_skills",
    "job_skills_not_in_cv",
    "technical_skills",
    "tools_platforms",
    "experience_highlights",
    "personal_projects",
    "soft_skills",
    "languages",
})


class CVChatEditor:
    """Modify tailored CV content through conversational instructions."""

    def __init__(self, ollama: OllamaClient | None = None):
        self.ollama = ollama or get_ollama_client()

    def modify(
        self,
        *,
        current_content: dict[str, Any],
        user_message: str,
        job: dict[str, Any],
        profile: dict[str, Any],
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Apply a user edit request and return updated content plus assistant reply."""
        if not self.ollama.is_available():
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama locally to use the CV chat editor."
            )
        self.ollama.validate_models()

        history_text = self._format_history(chat_history or [])
        job_context = build_job_context(job)
        preview_lines = cv_content_to_preview_lines(
            current_content,
            profile_name=str(profile.get("full_name", "") or ""),
        )
        numbered_preview = format_numbered_cv_preview(preview_lines)
        compact_content = json.dumps(current_content, separators=(",", ":"), ensure_ascii=False)
        prompt = f"""
The user wants to refine their tailored CV for a job application.

TARGET JOB:
{job_context}

NUMBERED CV PREVIEW (line numbers match the preview panel; the user may reference lines by number):
{numbered_preview}

CURRENT CV CONTENT (JSON — source of truth; do not invent facts beyond this):
{compact_content}

CONVERSATION SO FAR:
{history_text or 'None'}

USER REQUEST:
{user_message}

Instructions:
1. Apply only the changes the user requested.
2. Put only modified top-level fields inside "changes". Omit unchanged fields.
3. When the user cites a line number, map it to the numbered preview above.
4. If experience or project bullets change, include the full updated array in "changes".
5. Do not invent facts.

Return JSON with this exact shape:
{{
  "reply": "brief friendly explanation of what you changed",
  "changes": {{
    "professional_summary": "only when changed",
    "experience_highlights": []
  }}
}}
"""
        result = self.ollama.generate_json(
            prompt,
            model=self.ollama.main_model,
            system=CHAT_SYSTEM_PROMPT,
            temperature=0.2,
            max_attempts=3,
            schema=CV_CHAT_RESPONSE_SCHEMA,
        )
        reply = str(result.get("reply", "")).strip() or "I've updated your CV based on your request."
        updated_content = self._resolve_updated_content(current_content, result)
        updated = RAGCVGenerator._normalize_generated_content(
            updated_content,
            profile=profile,
            job=job,
        )
        return {"reply": reply, "content": updated}

    @staticmethod
    def _resolve_updated_content(
        current_content: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        content = result.get("content")
        if isinstance(content, dict):
            return content
        changes = result.get("changes")
        if isinstance(changes, dict):
            return CVChatEditor._apply_content_changes(current_content, changes)
        return current_content

    @staticmethod
    def _apply_content_changes(
        current_content: dict[str, Any],
        changes: dict[str, Any],
    ) -> dict[str, Any]:
        updated = deepcopy(current_content)
        for key, value in changes.items():
            if key in CONTENT_CHANGE_KEYS and value is not None:
                updated[key] = value
        return updated

    @staticmethod
    def rebuild_document(
        output_path: str,
        content: dict[str, Any],
        profile: dict[str, Any],
        *,
        template_path: str | None = None,
    ) -> None:
        """Rebuild the Word document from updated structured content."""
        builder = CVDocumentBuilder(template_path or get_default_cv_template_path())
        builder.build(output_path, content, profile=profile)

    @staticmethod
    def content_to_matched_categories(content: dict[str, Any]) -> dict[str, list[str]]:
        """Map tailored content fields to the job matched_categories shape."""
        return {
            "Skills Matching Job Description": content.get("job_matched_skills", []),
            "Job Skills Not In CV": content.get("job_skills_not_in_cv", []),
            "Technical Skills": content.get("technical_skills", content.get("key_skills", [])),
            "Tools & Platforms": content.get("tools_platforms", []),
        }

    @staticmethod
    def _format_history(chat_history: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for message in chat_history[-8:]:
            role = message.get("role", "user")
            label = "User" if role == "user" else "Assistant"
            text = str(message.get("content", "")).strip()
            if text:
                lines.append(f"{label}: {text}")
        return "\n".join(lines)
