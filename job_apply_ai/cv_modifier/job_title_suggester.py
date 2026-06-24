"""Suggest job search titles from a candidate profile using AI."""

from __future__ import annotations

import logging
from typing import Any

from job_apply_ai.cv_modifier.llm_client import LLMClient, get_llm_client
from job_apply_ai.dev_logging import dev_agent, dev_llm_context
from job_apply_ai.storage.user_profile import (
    normalize_profile,
    parse_professional_titles,
    profile_is_ready,
    profile_to_text,
)

logger = logging.getLogger(__name__)

SUGGEST_SYSTEM_PROMPT = (
    "You suggest realistic job board search titles for a candidate based on their profile. "
    "Return only valid JSON."
)

DEFAULT_MAX_TITLES = 10
MAX_SUGGESTED_TITLES = 20


def _normalize_title_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    titles: list[str] = []
    seen: set[str] = set()
    for item in value:
        title = str(item or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        titles.append(title)
    return titles


def heuristic_job_title_suggestions(profile: dict[str, Any] | None) -> list[str]:
    """Build search titles from stored professional titles and past roles."""
    normalized = normalize_profile(profile)
    titles: list[str] = []
    seen: set[str] = set()

    def add(title: str) -> None:
        cleaned = title.strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        titles.append(cleaned)

    for title in parse_professional_titles(normalized.get("professional_title", "")):
        add(title)

    for entry in normalized.get("work_experience") or []:
        if isinstance(entry, dict):
            add(str(entry.get("role") or ""))

    return titles[:MAX_SUGGESTED_TITLES]


def suggest_job_titles(
    profile: dict[str, Any] | None,
    *,
    max_titles: int = DEFAULT_MAX_TITLES,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Return job board search titles inferred from the candidate profile."""
    normalized = normalize_profile(profile)
    limit = max(1, min(int(max_titles or DEFAULT_MAX_TITLES), MAX_SUGGESTED_TITLES))

    if not profile_is_ready(normalized):
        return {
            "titles": [],
            "method": "skipped",
            "error": "Complete your profile with skills or experience before requesting title suggestions.",
        }

    client = llm or get_llm_client()
    if not client.is_available():
        titles = heuristic_job_title_suggestions(normalized)
        if not titles:
            return {
                "titles": [],
                "method": "heuristic",
                "error": f"{client.provider_label} is not reachable and no titles could be inferred from your profile.",
            }
        return {"titles": titles[:limit], "method": "heuristic"}

    profile_text = profile_to_text(normalized)
    existing_titles = parse_professional_titles(normalized.get("professional_title", ""))
    existing_block = ""
    if existing_titles:
        existing_block = (
            "\nExisting professional titles (include relevant ones and add closely related variants):\n"
            + ", ".join(existing_titles)
        )

    prompt = f"""
Suggest up to {limit} job board search titles for this candidate.

CANDIDATE PROFILE:
{profile_text}
{existing_block}

Instructions:
1. Suggest titles commonly used on LinkedIn, Indeed, and similar job boards.
2. Cover the candidate's strongest fit areas — include seniority when supported by experience.
3. Prefer titles the candidate is genuinely qualified for based on skills and work history.
4. Avoid titles that conflict with disqualifying skills, stacks, tools, or platforms in the profile.
5. Do not invent experience the profile does not support.
6. Return distinct titles only.

Return JSON with this exact shape:
{{
  "titles": ["Job Title 1", "Job Title 2"]
}}
"""

    try:
        with dev_agent("JobTitleSuggester"), dev_llm_context(
            operation="job_title_suggest",
            context={"max_titles": limit},
        ):
            client.validate_models()
            result = client.generate_json(
                prompt,
                model=client.fast_model,
                system=SUGGEST_SYSTEM_PROMPT,
                temperature=0.3,
                max_attempts=2,
            )
        titles = _normalize_title_list(result.get("titles"))
        if not titles:
            titles = heuristic_job_title_suggestions(normalized)
            return {"titles": titles[:limit], "method": "heuristic"}
        return {"titles": titles[:limit], "method": "ai"}
    except Exception as exc:
        logger.warning("AI job title suggestion failed: %s", exc)
        titles = heuristic_job_title_suggestions(normalized)
        if titles:
            return {"titles": titles[:limit], "method": "heuristic", "error": str(exc)}
        return {"titles": [], "method": "heuristic", "error": str(exc)}
