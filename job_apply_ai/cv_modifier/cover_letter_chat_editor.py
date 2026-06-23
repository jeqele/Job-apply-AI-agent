"""Chat-based editing of tailored cover letter content."""

from __future__ import annotations

import json
import logging
from typing import Any

from job_apply_ai.cv_modifier.chat_context import build_job_context, build_profile_context
from job_apply_ai.cv_modifier.cover_letter_builder import CoverLetterBuilder
from job_apply_ai.cv_modifier.cover_letter_generator import CoverLetterGenerator
from job_apply_ai.cv_modifier.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = (
    "You are a professional cover letter editing assistant. Apply the user's requested changes "
    "to the supplied cover letter JSON. Never invent employers, dates, degrees, certifications, "
    "or achievements that are not already supported by the current letter, CV highlights, or the "
    "user's explicit instruction. Keep language concise and professional. Return valid JSON only."
)


class CoverLetterChatEditor:
    """Modify cover letter content through conversational instructions."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm_client()

    def modify(
        self,
        *,
        current_content: dict[str, Any],
        user_message: str,
        job: dict[str, Any],
        profile: dict[str, Any],
        tailored_cv_content: dict[str, Any] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if not self.llm.is_available():
            raise RuntimeError(
                f"{self.llm.provider_label} is not reachable. Check your LLM settings to use the cover letter chat editor."
            )
        self.llm.validate_models()

        cv_summary = CoverLetterGenerator._summarize_cv(tailored_cv_content or {})
        history_text = self._format_history(chat_history or [])
        job_context = build_job_context(job)
        profile_context = build_profile_context(profile)
        prompt = f"""
The user wants to refine their cover letter for a job application.

TARGET JOB:
{job_context}

CANDIDATE PROFILE (full stored profile — only use facts from here):
{profile_context}

CV HIGHLIGHTS (facts you may reference):
{cv_summary}

CURRENT COVER LETTER (JSON):
{json.dumps(current_content, indent=2)}

CONVERSATION SO FAR:
{history_text or 'None'}

USER REQUEST:
{user_message}

Instructions:
1. Apply only the changes the user requested.
2. Preserve greeting, closing, and signature unless asked to change them.
3. Keep the same JSON shape as the current cover letter.
4. Do not invent facts.

Return JSON with this exact shape:
{{
  "reply": "brief friendly explanation of what you changed",
  "content": {{
    "date": "string",
    "recipient_name": "string",
    "recipient_company": "string",
    "greeting": "string",
    "body_paragraphs": ["paragraph"],
    "closing": "string",
    "signature_name": "string",
    "candidate_email": "string",
    "candidate_phone": "string"
  }}
}}
"""
        result = self.llm.generate_json(
            prompt,
            model=self.llm.main_model,
            system=CHAT_SYSTEM_PROMPT,
            temperature=0.2,
            max_attempts=2,
        )
        reply = str(result.get("reply", "")).strip() or "I've updated your cover letter based on your request."
        updated = CoverLetterGenerator.normalize(
            result.get("content") or current_content,
            profile,
            job,
        )
        return {"reply": reply, "content": updated}

    @staticmethod
    def rebuild_document(output_path: str, content: dict[str, Any]) -> None:
        CoverLetterBuilder().build(output_path, content)

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
