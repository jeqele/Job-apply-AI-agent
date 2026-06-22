"""Shared job and profile context for document chat editors."""

from __future__ import annotations

from typing import Any

from job_apply_ai.storage.user_profile import profile_to_text

MAX_JOB_DESCRIPTION_CHARS = 8000


def build_job_context(job: dict[str, Any]) -> str:
    """Serialize job fields including the full description for LLM prompts."""
    description = str(job.get("description", "") or "").strip()
    if len(description) > MAX_JOB_DESCRIPTION_CHARS:
        description = description[:MAX_JOB_DESCRIPTION_CHARS] + "\n… (description truncated)"

    parts = [
        f"Job title: {job.get('title', '')}",
        f"Company: {job.get('company', '')}",
        f"Location: {job.get('location', '')}",
        f"Work type: {job.get('work_type', '')}",
        f"Employment type: {job.get('employment_type', '')}",
        f"Seniority: {job.get('seniority_level', '')}",
        f"Industry: {job.get('industry', '')}",
        f"Visa sponsorship: {job.get('visa_sponsorship', '')}",
        f"Source: {job.get('source', '')}",
        f"Description:\n{description}",
    ]
    return "\n".join(part for part in parts if part and not part.endswith(": "))


def build_profile_context(profile: dict[str, Any]) -> str:
    """Serialize the full stored profile for LLM prompts."""
    return profile_to_text(profile).strip() or "No profile data available."
