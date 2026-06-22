"""Chat-based editing of tailored CV content."""

from __future__ import annotations

import json
import logging
from typing import Any

from job_apply_ai.cv_modifier.chat_context import build_job_context, build_profile_context
from job_apply_ai.cv_modifier.cv_generator import RAGCVGenerator
from job_apply_ai.cv_modifier.docx_builder import CVDocumentBuilder
from job_apply_ai.cv_modifier.ollama_client import OllamaClient
from job_apply_ai.storage.user_profile import get_default_cv_template_path

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = (
    "You are a professional CV editing assistant. Apply the user's requested changes to "
    "the supplied CV content JSON. Never invent employers, dates, degrees, certifications, "
    "achievements, or skills that are not already supported by the current content or the "
    "user's explicit instruction. Keep language concise and ATS-friendly. Return valid JSON only."
)


class CVChatEditor:
    """Modify tailored CV content through conversational instructions."""

    def __init__(self, ollama: OllamaClient | None = None):
        self.ollama = ollama or OllamaClient()

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
        profile_context = build_profile_context(profile)
        prompt = f"""
The user wants to refine their tailored CV for a job application.

TARGET JOB:
{job_context}

CANDIDATE PROFILE (full stored profile — only use facts from here):
{profile_context}

CURRENT CV CONTENT (JSON):
{json.dumps(current_content, indent=2)}

CONVERSATION SO FAR:
{history_text or 'None'}

USER REQUEST:
{user_message}

Instructions:
1. Apply only the changes the user requested.
2. Preserve all other sections unless the user asked to change them.
3. Keep the same JSON shape as the current content.
4. Do not invent facts.

Return JSON with this exact shape:
{{
  "reply": "brief friendly explanation of what you changed",
  "content": {{
    "professional_title": "string",
    "professional_summary": "string",
    "job_matched_skills": ["skill"],
    "job_skills_not_in_cv": ["skill"],
    "technical_skills": ["skill"],
    "tools_platforms": ["tool or platform"],
    "experience_highlights": [
      {{
        "role": "string",
        "company": "string",
        "period": "string",
        "bullets": ["bullet"]
      }}
    ],
    "personal_projects": [
      {{
        "name": "string",
        "description": "string",
        "bullets": ["bullet"]
      }}
    ],
    "soft_skills": ["skill"],
    "languages": ["language and level"]
  }}
}}
"""
        result = self.ollama.generate_json(
            prompt,
            model=self.ollama.main_model,
            system=CHAT_SYSTEM_PROMPT,
            temperature=0.2,
            max_attempts=2,
        )
        reply = str(result.get("reply", "")).strip() or "I've updated your CV based on your request."
        updated = RAGCVGenerator._normalize_generated_content(
            result.get("content") or current_content,
            profile=profile,
            job=job,
        )
        return {"reply": reply, "content": updated}

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
