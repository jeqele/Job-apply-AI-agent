"""Read-only Q&A about a tailored CV and target job."""

from __future__ import annotations

import json
import logging
from typing import Any

from job_apply_ai.cv_modifier.chat_context import (
    build_job_context,
    format_numbered_cv_preview,
    resolve_cv_preview_lines,
)
from job_apply_ai.cv_modifier.cv_chat_editor import CVChatEditor
from job_apply_ai.cv_modifier.llm_client import LLMClient, get_llm_client
from job_apply_ai.dev_logging import dev_llm_context

logger = logging.getLogger(__name__)

ASK_HISTORY_LIMIT = 6

ASK_SYSTEM_PROMPT = (
    "You are a helpful assistant for job applications. Answer the user's questions using "
    "only the supplied job description and tailored CV content. Do not modify the CV or "
    "suggest edits unless the user explicitly asks for advice. Never invent employers, "
    "dates, degrees, certifications, achievements, or skills that are not supported by "
    "the context. If the answer is not in the context, say so briefly."
)


class CVAskAssistant:
    """Answer questions about a job and tailored CV without changing documents."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm_client()

    def ask(
        self,
        *,
        current_content: dict[str, Any],
        user_message: str,
        job: dict[str, Any],
        profile: dict[str, Any],
        chat_history: list[dict[str, str]] | None = None,
        preview_lines: list[dict[str, str]] | None = None,
        preview_customized: bool = False,
    ) -> str:
        """Return an assistant reply without modifying CV content."""
        if not self.llm.is_available():
            raise RuntimeError(
                f"{self.llm.provider_label} is not reachable. Check your LLM settings to use Ask From CV."
            )
        self.llm.validate_models()

        history_text = CVChatEditor._format_history(
            (chat_history or [])[-ASK_HISTORY_LIMIT:],
        )
        job_context = build_job_context(job)
        profile_name = str(profile.get("full_name", "") or "")
        resolved_preview_lines = resolve_cv_preview_lines(
            current_content,
            profile_name,
            stored_lines=preview_lines,
            customized=preview_customized,
        )
        numbered_preview = format_numbered_cv_preview(resolved_preview_lines)
        compact_content = json.dumps(current_content, separators=(",", ":"), ensure_ascii=False)
        prompt = f"""
The user is reviewing their tailored CV for a job application and has a question.

TARGET JOB:
{job_context}

NUMBERED CV PREVIEW:
{numbered_preview}

TAILORED CV CONTENT (JSON):
{compact_content}

RECENT CONVERSATION:
{history_text or "None"}

USER QUESTION:
{user_message}

Answer the question clearly and concisely. Do not propose CV edits unless the user asks for advice.
"""
        with dev_llm_context(
            operation="cv_ask",
            chat_history=chat_history or [],
            context={
                "user_message": user_message,
                "job_title": job.get("title", ""),
                "job_company": job.get("company", ""),
                "document": "cv_ask",
            },
        ):
            reply = self.llm.generate(
                prompt,
                model=self.llm.main_model,
                system=ASK_SYSTEM_PROMPT,
                temperature=0.3,
            )
        text = str(reply or "").strip()
        return text or "I could not find enough context to answer that question."
