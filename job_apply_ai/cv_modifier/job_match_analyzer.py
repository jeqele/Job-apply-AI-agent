"""Analyze whether job listings match a candidate's profile skills."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from job_apply_ai.cv_modifier.ollama_client import OllamaClient
from job_apply_ai.storage.user_profile import normalize_profile

logger = logging.getLogger(__name__)

MATCH_SYSTEM_PROMPT = (
    "You evaluate whether a job listing is a reasonable fit for a candidate based on their "
    "technical skills, minor skills, and technology stacks. Return only valid JSON."
)

NOT_MATCH_STATUS = "not_match"
DEFAULT_MIN_MATCH_SCORE = 50.0


def normalize_min_match_score(value: Any) -> float:
    """Clamp a user-provided threshold to 0-100."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MIN_MATCH_SCORE
    return max(0.0, min(100.0, score))


def profile_has_matchable_skills(profile: dict[str, Any] | None) -> bool:
    """Return True when the profile has skills usable for job matching."""
    normalized = normalize_profile(profile)
    return bool(
        normalized["technical_skills"]
        or normalized["minor_skills"]
        or normalized["stacks"]
    )


def collect_profile_skills(profile: dict[str, Any] | None) -> dict[str, list[str]]:
    normalized = normalize_profile(profile)
    return {
        "technical_skills": normalized["technical_skills"],
        "minor_skills": normalized["minor_skills"],
        "stacks": normalized["stacks"],
    }


def _normalize_token(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _skill_tokens(skills: dict[str, list[str]]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for group in skills.values():
        for item in group:
            normalized = _normalize_token(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            tokens.append(normalized)
            for part in re.split(r"[,/|+&]", item):
                part_norm = _normalize_token(part)
                if part_norm and part_norm not in seen:
                    seen.add(part_norm)
                    tokens.append(part_norm)
    return tokens


def _job_text(job: dict[str, Any]) -> str:
    parts = [
        str(job.get("title") or ""),
        str(job.get("company") or ""),
        str(job.get("description") or ""),
    ]
    return "\n".join(part.strip() for part in parts if part.strip()).lower()


def heuristic_job_match(job: dict[str, Any], profile: dict[str, Any] | None) -> dict[str, Any]:
    """Keyword-based fallback when Ollama is unavailable."""
    skills = collect_profile_skills(profile)
    tokens = _skill_tokens(skills)
    haystack = _job_text(job)

    if not tokens:
        return {
            "is_match": True,
            "match_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "reason": "No profile skills configured for matching.",
            "method": "skipped",
        }

    matched = [token for token in tokens if token in haystack]
    ratio = len(matched) / max(len(tokens), 1)
    is_match = bool(matched) and ratio >= 0.15

    return {
        "is_match": is_match,
        "match_score": round(min(ratio * 100, 100), 1),
        "matched_skills": matched[:12],
        "missing_skills": [],
        "reason": (
            "Matched profile skills found in the job description."
            if is_match
            else "Too few profile skills appear in the job description."
        ),
        "method": "heuristic",
    }


def analyze_job_match(
    job: dict[str, Any],
    profile: dict[str, Any] | None,
    ollama: OllamaClient | None = None,
) -> dict[str, Any]:
    """Return match analysis for a single job against the stored profile."""
    skills = collect_profile_skills(profile)
    if not any(skills.values()):
        return heuristic_job_match(job, profile)

    client = ollama or OllamaClient()
    if not client.is_available():
        return heuristic_job_match(job, profile)

    job_context = "\n".join(
        line
        for line in [
            f"Title: {job.get('title', '')}",
            f"Company: {job.get('company', '')}",
            f"Location: {job.get('location', '')}",
            f"Description:\n{job.get('description', '')}",
        ]
        if line.strip()
    )
    skills_context = "\n".join(
        f"{label}: {', '.join(values) if values else '(none)'}"
        for label, values in [
            ("Technical skills", skills["technical_skills"]),
            ("Minor skills", skills["minor_skills"]),
            ("Stacks", skills["stacks"]),
        ]
    )

    prompt = f"""
Decide if this job is a reasonable fit for the candidate.

CANDIDATE SKILLS:
{skills_context}

JOB:
{job_context}

Return JSON with this exact shape:
{{
  "is_match": true,
  "match_score": 75,
  "matched_skills": ["skill from profile that fits the job"],
  "missing_skills": ["important job requirement the profile lacks"],
  "reason": "one short sentence explaining the decision"
}}

Rules:
- is_match is true when the candidate's technical skills, minor skills, or stacks align with the role.
- Minor skills and stack familiarity can support a match even when not every job keyword is listed.
- is_match is false when the role targets a different domain or requires skills/stacks the profile does not support.
- matched_skills must come from the candidate lists only.
- missing_skills should list only important gaps from the job description.
- match_score is 0-100 indicating overall fit.
"""

    try:
        result = client.generate_json(
            prompt,
            model=client.fast_model,
            system=MATCH_SYSTEM_PROMPT,
            temperature=0.1,
            max_attempts=2,
        )
        return {
            "is_match": bool(result.get("is_match")),
            "match_score": float(result.get("match_score") or 0),
            "matched_skills": _normalize_result_list(result.get("matched_skills")),
            "missing_skills": _normalize_result_list(result.get("missing_skills")),
            "reason": str(result.get("reason") or "").strip(),
            "method": "ai",
        }
    except Exception as exc:
        logger.warning("AI job match failed for %r: %s", job.get("title", ""), exc)
        return heuristic_job_match(job, profile)


def _normalize_result_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def job_meets_threshold(analysis: dict[str, Any], min_match_score: float) -> bool:
    """Return True when a job's match score meets the configured minimum."""
    if analysis.get("method") == "skipped":
        return True
    return float(analysis.get("match_score") or 0) >= min_match_score


def apply_profile_match_to_job(
    job: dict[str, Any],
    profile: dict[str, Any] | None,
    *,
    min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
    ollama: OllamaClient | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze one job, store fit metadata, and set workflow status from the threshold."""
    updated = dict(job)
    analysis = analyze_job_match(updated, profile, ollama=ollama)
    analysis["min_match_score"] = min_match_score
    meets = job_meets_threshold(analysis, min_match_score)
    analysis["is_match"] = meets

    categories = dict(updated.get("matched_categories") or {})
    categories["Profile Fit"] = analysis
    updated["matched_categories"] = categories

    previous_status = updated.get("workflow_status") or "new"
    if not meets:
        updated["workflow_status"] = NOT_MATCH_STATUS
    elif previous_status == NOT_MATCH_STATUS:
        updated["workflow_status"] = "new"

    return updated, analysis


def classify_jobs_by_profile_match(
    jobs: list[dict[str, Any]],
    profile: dict[str, Any] | None,
    *,
    min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
    ollama: OllamaClient | None = None,
) -> list[dict[str, Any]]:
    """Annotate jobs with fit analysis and route non-matches to the not_match folder."""
    if not jobs or not profile_has_matchable_skills(profile):
        return jobs

    threshold = normalize_min_match_score(min_match_score)
    classified: list[dict[str, Any]] = []
    for job in jobs:
        updated, _analysis = apply_profile_match_to_job(
            job,
            profile,
            min_match_score=threshold,
            ollama=ollama,
        )
        classified.append(updated)
    return classified


def analyze_jobs_with_threshold(
    jobs: list[dict[str, Any]],
    profile: dict[str, Any] | None,
    min_match_score: float,
    *,
    ollama: OllamaClient | None = None,
    on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Analyze jobs and summarize how many were moved or restored."""
    threshold = normalize_min_match_score(min_match_score)
    stats = {
        "analyzed": 0,
        "moved_to_not_match": 0,
        "restored_to_new": 0,
        "unchanged": 0,
        "min_match_score": threshold,
    }
    updated_jobs: list[dict[str, Any]] = []

    total = len(jobs)
    for index, job in enumerate(jobs):
        if on_progress:
            on_progress(index, total, job)

        previous_status = job.get("workflow_status") or "new"
        updated, _analysis = apply_profile_match_to_job(
            job,
            profile,
            min_match_score=threshold,
            ollama=ollama,
        )
        updated_jobs.append(updated)
        stats["analyzed"] += 1

        new_status = updated.get("workflow_status") or previous_status
        if new_status == NOT_MATCH_STATUS and previous_status != NOT_MATCH_STATUS:
            stats["moved_to_not_match"] += 1
        elif new_status == "new" and previous_status == NOT_MATCH_STATUS:
            stats["restored_to_new"] += 1
        else:
            stats["unchanged"] += 1

    return {"jobs": updated_jobs, "stats": stats}
