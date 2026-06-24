"""ATS-friendly CV analysis and improvement suggestions."""

from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy
from typing import Any

from job_apply_ai.cv_modifier.chat_context import build_job_context, cv_content_to_preview_lines
from job_apply_ai.cv_modifier.cv_chat_editor import (
    CONTENT_CHANGE_KEYS,
    CV_CHAT_RESPONSE_SCHEMA,
    CVChatEditor,
)
from job_apply_ai.cv_modifier.cv_generator import RAGCVGenerator
from job_apply_ai.cv_modifier.llm_client import LLMClient, get_llm_client
from job_apply_ai.dev_logging import dev_llm_context
from job_apply_ai.storage.user_profile import profile_to_text

logger = logging.getLogger(__name__)

ATS_SYSTEM_PROMPT = (
    "You are an expert ATS resume analyst and CV writer. Evaluate how well a candidate's "
    "current CV content aligns with ATS scanning rules and a specific job description. "
    "Suggest truthful improvements only — never invent employers, dates, degrees, "
    "certifications, achievements, or skills. Return valid JSON only."
)

ATS_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "ats_score": {"type": "number"},
        "score_summary": {"type": "string"},
        "keyword_bank": {"type": "array", "items": {"type": "string"}},
        "matched_keywords": {"type": "array", "items": {"type": "string"}},
        "missing_keywords": {"type": "array", "items": {"type": "string"}},
        "formatting_notes": {"type": "array", "items": {"type": "string"}},
        "trade_offs": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "rationale": {"type": "string"},
                    "category": {"type": "string"},
                    "priority": {"type": "string"},
                    "changes": {"type": "object", "additionalProperties": True},
                },
                "required": ["title", "description", "rationale", "category", "changes"],
            },
        },
    },
    "required": [
        "ats_score",
        "score_summary",
        "keyword_bank",
        "matched_keywords",
        "missing_keywords",
        "suggestions",
    ],
}

SUGGESTION_REAPPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "rationale": {"type": "string"},
        "changes": {"type": "object", "additionalProperties": True},
    },
    "required": ["title", "description", "rationale", "changes"],
}

SUGGESTION_STATUSES = frozenset({"pending", "applied", "denied", "failed"})


def normalize_ats_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _cv_content_as_text(content: dict[str, Any], profile_name: str = "") -> str:
    lines = cv_content_to_preview_lines(content, profile_name)
    return "\n".join(line.get("text", "") for line in lines if line.get("text"))


def _normalize_suggestion(raw: dict[str, Any]) -> dict[str, Any]:
    changes = raw.get("changes") or {}
    if not isinstance(changes, dict):
        changes = {}
    filtered_changes = {
        key: value
        for key, value in changes.items()
        if key in CONTENT_CHANGE_KEYS and value is not None
    }
    return {
        "id": str(raw.get("id") or uuid.uuid4().hex[:12]),
        "title": str(raw.get("title", "")).strip() or "CV improvement",
        "description": str(raw.get("description", "")).strip(),
        "rationale": str(raw.get("rationale", "")).strip(),
        "category": str(raw.get("category", "general")).strip() or "general",
        "priority": str(raw.get("priority", "medium")).strip().lower() or "medium",
        "changes": filtered_changes,
        "status": str(raw.get("status", "pending")).strip().lower() or "pending",
        "error": str(raw.get("error", "")).strip(),
    }


def normalize_ats_analysis(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized ATS analysis payload for storage and UI."""
    data = raw if isinstance(raw, dict) else {}
    suggestions = [
        _normalize_suggestion(item)
        for item in (data.get("suggestions") or [])
        if isinstance(item, dict)
    ]
    return {
        "ats_score": normalize_ats_score(data.get("ats_score")),
        "score_summary": str(data.get("score_summary", "")).strip(),
        "keyword_bank": [
            str(item).strip()
            for item in (data.get("keyword_bank") or [])
            if str(item).strip()
        ],
        "matched_keywords": [
            str(item).strip()
            for item in (data.get("matched_keywords") or [])
            if str(item).strip()
        ],
        "missing_keywords": [
            str(item).strip()
            for item in (data.get("missing_keywords") or [])
            if str(item).strip()
        ],
        "formatting_notes": [
            str(item).strip()
            for item in (data.get("formatting_notes") or [])
            if str(item).strip()
        ],
        "trade_offs": str(data.get("trade_offs", "")).strip(),
        "suggestions": suggestions,
        "analyzed_at": str(data.get("analyzed_at", "")).strip(),
        "method": str(data.get("method", "ai")).strip() or "ai",
    }


def _build_analysis_prompt(
    *,
    job: dict[str, Any],
    cv_content: dict[str, Any],
    profile: dict[str, Any],
) -> str:
    job_context = build_job_context(job)
    cv_text = _cv_content_as_text(cv_content, str(profile.get("full_name", "") or ""))
    profile_text = profile_to_text(profile).strip()
    compact_cv = json.dumps(cv_content, separators=(",", ":"), ensure_ascii=False)

    return f"""
Objective: Evaluate how ATS-friendly the candidate's current CV is for this specific job,
then suggest truthful improvements aligned with ATS scanning rules.

TARGET JOB:
{job_context}

CURRENT CV (plain-text preview):
{cv_text}

CURRENT CV CONTENT (JSON — source of truth; do not invent facts beyond this):
{compact_cv}

FULL CANDIDATE PROFILE (for factual grounding only):
{profile_text}

Instructions:
1. Parse the job description for required/preferred skills, tools, certifications, and measurable outcomes.
2. Build a keyword bank of the top 10–20 ATS-relevant terms from the job description.
3. Score the current CV from 0–100 for ATS compatibility with this job:
   - Keyword alignment with the job description
   - Standard section structure and headings
   - Action-verb bullets with measurable outcomes
   - Natural keyword integration without stuffing
   - Truthfulness to the candidate profile and the candidate's self-rated skill familiarity percentages
4. Identify matched and missing high-value keywords.
5. Provide 4–10 concrete improvement suggestions. Each suggestion must include:
   - title: short label
   - description: what to change and why
   - rationale: ATS impact
   - category: one of summary, skills, experience, keywords, formatting, structure
   - priority: high, medium, or low
   - changes: only modified top-level CV fields (same keys as the JSON content).
     Include full arrays when changing experience_highlights or personal_projects.
6. Never invent employers, dates, degrees, certifications, achievements, or skills.
7. Reframe bullets with strong action verbs and metrics only when supported by the profile.
8. Note any trade-offs in trade_offs (e.g. longer summary for readability).

Return JSON with this exact shape:
{{
  "ats_score": 0,
  "score_summary": "one paragraph explaining the score",
  "keyword_bank": ["keyword"],
  "matched_keywords": ["keyword already present"],
  "missing_keywords": ["important keyword absent or weak"],
  "formatting_notes": ["ATS formatting observation"],
  "trade_offs": "brief note on compromises if any",
  "suggestions": [
    {{
      "title": "string",
      "description": "string",
      "rationale": "string",
      "category": "skills",
      "priority": "high",
      "changes": {{
        "professional_summary": "only when changed"
      }}
    }}
  ]
}}
"""


class ATSFriendlyAnalyzer:
    """Analyze CV ATS compatibility and produce actionable suggestions."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm_client()

    def analyze(
        self,
        *,
        job: dict[str, Any],
        cv_content: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.llm.is_available():
            raise RuntimeError(
                f"{self.llm.provider_label} is not reachable. Check your LLM settings to run ATS analysis."
            )
        self.llm.validate_models()

        prompt = _build_analysis_prompt(job=job, cv_content=cv_content, profile=profile)
        with dev_llm_context(
            operation="ats_analysis",
            context={
                "job_title": job.get("title", ""),
                "job_company": job.get("company", ""),
                "cv_fields": list(cv_content.keys()) if isinstance(cv_content, dict) else [],
            },
        ):
            result = self.llm.generate_json(
                prompt,
                model=self.llm.main_model,
                system=ATS_SYSTEM_PROMPT,
                temperature=0.2,
                max_attempts=3,
                schema=ATS_ANALYSIS_SCHEMA,
            )
        analysis = normalize_ats_analysis(result)
        analysis["method"] = "ai"
        return analysis

    def reapply_suggestion(
        self,
        *,
        job: dict[str, Any],
        cv_content: dict[str, Any],
        profile: dict[str, Any],
        suggestion: dict[str, Any],
    ) -> dict[str, Any]:
        """Regenerate a single suggestion when a prior apply failed due to LLM issues."""
        if not self.llm.is_available():
            raise RuntimeError(
                f"{self.llm.provider_label} is not reachable. Check your LLM settings to retry this suggestion."
            )
        self.llm.validate_models()

        job_context = build_job_context(job)
        compact_cv = json.dumps(cv_content, separators=(",", ":"), ensure_ascii=False)
        prior = json.dumps(suggestion, separators=(",", ":"), ensure_ascii=False)

        prompt = f"""
Regenerate one ATS improvement suggestion for this job application.

TARGET JOB:
{job_context}

CURRENT CV CONTENT (JSON):
{compact_cv}

FAILED OR STALE SUGGESTION:
{prior}

Return a replacement suggestion as JSON with:
{{
  "title": "string",
  "description": "string",
  "rationale": "string",
  "changes": {{}}
}}

Rules:
- Keep the same intent as the failed suggestion but produce valid, applicable changes.
- Only modify fields present in changes; use the same CONTENT_CHANGE_KEYS as the CV JSON.
- Never invent facts beyond the profile and current CV content.
"""
        with dev_llm_context(
            operation="ats_suggestion_reapply",
            context={
                "job_title": job.get("title", ""),
                "suggestion_id": suggestion.get("id", ""),
                "suggestion_title": suggestion.get("title", ""),
            },
        ):
            result = self.llm.generate_json(
                prompt,
                model=self.llm.main_model,
                system=ATS_SYSTEM_PROMPT,
                temperature=0.25,
                max_attempts=3,
                schema=SUGGESTION_REAPPLY_SCHEMA,
            )
        refreshed = _normalize_suggestion(
            {
                **suggestion,
                **result,
                "status": "pending",
                "error": "",
            }
        )
        return refreshed

    def apply_suggestion(
        self,
        *,
        job: dict[str, Any],
        cv_content: dict[str, Any],
        profile: dict[str, Any],
        suggestion: dict[str, Any],
    ) -> dict[str, Any]:
        """Rewrite the current CV by applying one ATS suggestion via the LLM."""
        if not self.llm.is_available():
            raise RuntimeError(
                f"{self.llm.provider_label} is not reachable. "
                "Check your LLM settings to apply ATS suggestions."
            )
        self.llm.validate_models()

        job_context = build_job_context(job)
        profile_name = str(profile.get("full_name", "") or "")
        cv_text = _cv_content_as_text(cv_content, profile_name)
        compact_cv = json.dumps(cv_content, separators=(",", ":"), ensure_ascii=False)
        profile_text = profile_to_text(profile).strip()
        prior_changes = suggestion.get("changes") or {}
        hint_block = ""
        if isinstance(prior_changes, dict) and prior_changes:
            hint_block = (
                "\nORIGINAL SUGGESTED CHANGES (hint only — always match the current CV state above):\n"
                f"{json.dumps(prior_changes, separators=(',', ':'), ensure_ascii=False)}"
            )

        prompt = f"""
Apply one ATS improvement suggestion to the candidate's current CV for this job application.

TARGET JOB:
{job_context}

CURRENT CV (plain-text preview):
{cv_text}

CURRENT CV CONTENT (JSON — source of truth; do not invent facts beyond this and the profile):
{compact_cv}

FULL CANDIDATE PROFILE (factual grounding only):
{profile_text}

ATS SUGGESTION TO IMPLEMENT:
- Title: {suggestion.get("title", "")}
- Category: {suggestion.get("category", "general")}
- What to change: {suggestion.get("description", "")}
- ATS rationale: {suggestion.get("rationale", "")}
{hint_block}

Instructions:
1. Implement the suggestion against the CURRENT CV content above, not a stale snapshot.
2. Put only modified top-level fields inside "changes". Omit unchanged fields.
3. Allowed change keys: {", ".join(sorted(CONTENT_CHANGE_KEYS))}
4. When experience or project bullets change, include the full updated array in "changes".
5. Never invent employers, dates, degrees, certifications, achievements, or skills.
6. Keep edits minimal — change only what this suggestion requires.

Return JSON with this exact shape:
{{
  "reply": "brief note on what you changed",
  "changes": {{
    "professional_summary": "only when changed"
  }}
}}
"""
        with dev_llm_context(
            operation="ats_suggestion_apply",
            context={
                "job_title": job.get("title", ""),
                "job_company": job.get("company", ""),
                "suggestion_id": suggestion.get("id", ""),
                "suggestion_title": suggestion.get("title", ""),
            },
        ):
            result = self.llm.generate_json(
                prompt,
                model=self.llm.main_model,
                system=ATS_SYSTEM_PROMPT,
                temperature=0.2,
                max_attempts=3,
                schema=CV_CHAT_RESPONSE_SCHEMA,
            )

        changes = result.get("changes") if isinstance(result.get("changes"), dict) else {}
        if not changes:
            raise ValueError("The AI did not return applicable CV changes for this suggestion.")

        updated = CVChatEditor._apply_content_changes(cv_content, changes)
        return RAGCVGenerator._normalize_generated_content(
            updated,
            profile=profile,
            job=job,
        )


def apply_suggestion_to_content(
    cv_content: dict[str, Any],
    suggestion: dict[str, Any],
    *,
    profile: dict[str, Any] | None = None,
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a suggestion's changes to CV content and normalize the result."""
    changes = suggestion.get("changes") or {}
    if not isinstance(changes, dict) or not changes:
        raise ValueError("Suggestion has no applicable changes.")

    updated = CVChatEditor._apply_content_changes(cv_content, changes)
    return RAGCVGenerator._normalize_generated_content(
        updated,
        profile=profile,
        job=job,
    )


def update_suggestion_status(
    analysis: dict[str, Any],
    suggestion_id: str,
    *,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    """Return analysis with one suggestion's status updated."""
    if status not in SUGGESTION_STATUSES:
        raise ValueError(f"Unsupported suggestion status: {status}")

    updated = deepcopy(normalize_ats_analysis(analysis))
    found = False
    for item in updated["suggestions"]:
        if item.get("id") == suggestion_id:
            item["status"] = status
            item["error"] = error
            found = True
            break
    if not found:
        raise KeyError(f"Suggestion not found: {suggestion_id}")
    return updated


def replace_suggestion(
    analysis: dict[str, Any],
    suggestion_id: str,
    replacement: dict[str, Any],
) -> dict[str, Any]:
    """Replace a suggestion by id (used after reapply)."""
    updated = deepcopy(normalize_ats_analysis(analysis))
    replacement_norm = _normalize_suggestion({**replacement, "id": suggestion_id, "status": "pending"})
    for index, item in enumerate(updated["suggestions"]):
        if item.get("id") == suggestion_id:
            updated["suggestions"][index] = replacement_norm
            return updated
    raise KeyError(f"Suggestion not found: {suggestion_id}")


def get_suggestion(analysis: dict[str, Any], suggestion_id: str) -> dict[str, Any]:
    for item in normalize_ats_analysis(analysis).get("suggestions", []):
        if item.get("id") == suggestion_id:
            return item
    raise KeyError(f"Suggestion not found: {suggestion_id}")
