"""Analyze whether job listings match a candidate's profile skills."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from job_apply_ai.cv_modifier.llm_client import LLMClient, get_llm_client
from job_apply_ai.storage.user_profile import (
    format_skills_line,
    normalize_profile,
    skill_item_name,
)

logger = logging.getLogger(__name__)

MATCH_SYSTEM_PROMPT = (
    "You evaluate whether a job listing is a reasonable fit for a candidate based on their "
    "technical skills and technology stacks. Disqualifying skills, stacks, tools, and platforms "
    "are items the candidate does not want in a role; if a job requires or heavily emphasizes "
    "them, the role is not a fit. Each skill includes a self-rated familiarity percentage "
    "(0-100). Weight stronger positive skills more heavily when judging fit. Return only valid JSON."
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
    """Return True when the profile has positive skills usable for job matching."""
    normalized = normalize_profile(profile)
    return bool(normalized["technical_skills"] or normalized["stacks"])


def collect_positive_profile_skills(profile: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Return technical skills and stacks used as positive fit signals."""
    normalized = normalize_profile(profile)
    return {
        "technical_skills": normalized["technical_skills"],
        "stacks": normalized["stacks"],
    }


def collect_disqualifying_skills(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return skills that disqualify a job when the role requires them."""
    return normalize_profile(profile)["minor_skills"]


def collect_disqualifying_tools_platforms(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return tools/platforms that disqualify a job when the role requires them."""
    return normalize_profile(profile)["disqualifying_tools_platforms"]


def collect_disqualifying_stacks(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return stacks that disqualify a job when the role requires them."""
    return normalize_profile(profile)["disqualifying_stacks"]


def collect_all_disqualifiers(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return every disqualifying skill, stack, and tool/platform entry."""
    normalized = normalize_profile(profile)
    return (
        normalized["minor_skills"]
        + normalized["disqualifying_stacks"]
        + normalized["disqualifying_tools_platforms"]
    )


def collect_profile_skills(profile: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Return all skill groups referenced during job matching."""
    normalized = normalize_profile(profile)
    return {
        "technical_skills": normalized["technical_skills"],
        "minor_skills": normalized["minor_skills"],
        "stacks": normalized["stacks"],
    }


def _skill_entries(skills: dict[str, list[dict[str, Any]]]) -> list[tuple[str, int]]:
    """Flatten profile skills into searchable tokens with familiarity weights."""
    entries: list[tuple[str, int]] = []
    seen: set[str] = set()
    for group in skills.values():
        for item in group:
            name = skill_item_name(item)
            familiarity = int(item.get("familiarity", 70)) if isinstance(item, dict) else 70
            normalized = _normalize_token(name)
            if normalized and normalized not in seen:
                seen.add(normalized)
                entries.append((normalized, familiarity))
            for part in re.split(r"[,/|+&]", name):
                part_norm = _normalize_token(part)
                if part_norm and part_norm not in seen:
                    seen.add(part_norm)
                    entries.append((part_norm, familiarity))
    return entries


def _skill_tokens(skills: dict[str, list[dict[str, Any]]]) -> list[str]:
    return [token for token, _ in _skill_entries(skills)]


def _normalize_token(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _job_text(job: dict[str, Any]) -> str:
    parts = [
        str(job.get("title") or ""),
        str(job.get("company") or ""),
        str(job.get("description") or ""),
    ]
    return "\n".join(part.strip() for part in parts if part.strip()).lower()


def _format_skill_list(skills: list[str], limit: int = 6) -> str:
    shown = [skill for skill in skills if str(skill).strip()][:limit]
    if not shown:
        return ""
    if len(skills) > limit:
        return f"{', '.join(shown)}, and others"
    return ", ".join(shown)


def _build_match_paragraphs(
    *,
    matched_skills: list[str],
    missing_skills: list[str],
    reason: str,
    is_match: bool,
    match_paragraph: str = "",
    mismatch_paragraph: str = "",
) -> tuple[str, str]:
    """Ensure readable fit summaries exist for UI display."""
    match_text = str(match_paragraph or "").strip()
    mismatch_text = str(mismatch_paragraph or "").strip()

    if not match_text:
        if matched_skills:
            skills_text = _format_skill_list(matched_skills)
            match_text = (
                f"Your profile overlaps with this role through skills such as {skills_text}. "
                "These appear in the job description and support applying for the position."
            )
        elif reason and is_match:
            match_text = reason

    if not mismatch_text:
        if missing_skills:
            gaps_text = _format_skill_list(missing_skills)
            mismatch_text = (
                f"The posting also emphasizes {gaps_text}, which are not clearly represented "
                "in your profile and may weaken your fit for this role."
            )
        elif reason and not is_match:
            mismatch_text = reason
        elif not is_match:
            mismatch_text = (
                "Only limited overlap was found between your profile skills and the job requirements, "
                "so the role may target a different focus area."
            )
        elif matched_skills:
            mismatch_text = (
                "No major skill gaps stood out, though the role may still expect seniority, domain "
                "experience, or responsibilities beyond what your profile highlights."
            )

    return match_text, mismatch_text


def _disqualifying_tokens(profile: dict[str, Any] | None) -> list[str]:
    return [
        token
        for token, _ in _skill_entries({"disqualifying": collect_all_disqualifiers(profile)})
    ]


def heuristic_job_match(job: dict[str, Any], profile: dict[str, Any] | None) -> dict[str, Any]:
    """Keyword-based fallback when the LLM provider is unavailable."""
    skills = collect_positive_profile_skills(profile)
    entries = _skill_entries(skills)
    haystack = _job_text(job)
    triggered_disqualifiers = [token for token in _disqualifying_tokens(profile) if token in haystack]

    if triggered_disqualifiers:
        reason = (
            "The job emphasizes skills, stacks, or tools you marked as disqualifying: "
            + _format_skill_list(triggered_disqualifiers)
            + "."
        )
        match_paragraph, mismatch_paragraph = _build_match_paragraphs(
            matched_skills=[],
            missing_skills=triggered_disqualifiers,
            reason=reason,
            is_match=False,
        )
        return {
            "is_match": False,
            "match_score": 0.0,
            "matched_skills": [],
            "missing_skills": triggered_disqualifiers,
            "reason": reason,
            "match_paragraph": match_paragraph,
            "mismatch_paragraph": mismatch_paragraph,
            "method": "heuristic",
        }

    if not entries:
        return {
            "is_match": True,
            "match_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "reason": "No profile skills configured for matching.",
            "match_paragraph": "",
            "mismatch_paragraph": "",
            "method": "skipped",
        }

    matched = [(token, familiarity) for token, familiarity in entries if token in haystack]
    total_weight = sum(familiarity for _, familiarity in entries)
    matched_weight = sum(familiarity for _, familiarity in matched)
    ratio = matched_weight / max(total_weight, 1)
    matched_names = [token for token, _ in matched[:12]]
    is_match = bool(matched) and ratio >= 0.15
    reason = (
        "Matched profile skills found in the job description."
        if is_match
        else "Too few profile skills appear in the job description."
    )
    match_paragraph, mismatch_paragraph = _build_match_paragraphs(
        matched_skills=matched_names,
        missing_skills=[],
        reason=reason,
        is_match=is_match,
    )

    return {
        "is_match": is_match,
        "match_score": round(min(ratio * 100, 100), 1),
        "matched_skills": matched_names,
        "missing_skills": [],
        "reason": reason,
        "match_paragraph": match_paragraph,
        "mismatch_paragraph": mismatch_paragraph,
        "method": "heuristic",
    }


def analyze_job_match(
    job: dict[str, Any],
    profile: dict[str, Any] | None,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Return match analysis for a single job against the stored profile."""
    skills = collect_positive_profile_skills(profile)
    disqualifying_skills = collect_disqualifying_skills(profile)
    disqualifying_stacks = collect_disqualifying_stacks(profile)
    disqualifying_tools = collect_disqualifying_tools_platforms(profile)
    if (
        not any(skills.values())
        and not disqualifying_skills
        and not disqualifying_stacks
        and not disqualifying_tools
    ):
        return heuristic_job_match(job, profile)

    client = llm or get_llm_client()
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
        f"{label}: {format_skills_line(values) if values else '(none)'}"
        for label, values in [
            ("Technical skills", skills["technical_skills"]),
            ("Stacks", skills["stacks"]),
            (
                "Disqualifying skills (not a fit when the job requires these)",
                disqualifying_skills,
            ),
            (
                "Disqualifying technology stacks (not a fit when the job requires these)",
                disqualifying_stacks,
            ),
            (
                "Disqualifying tools & platforms (not a fit when the job requires these)",
                disqualifying_tools,
            ),
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
  "reason": "one short sentence explaining the decision",
  "match_paragraph": "2-4 sentences explaining why the candidate matches this job, citing specific profile skills and job requirements.",
  "mismatch_paragraph": "2-4 sentences explaining why the candidate may not match, citing missing skills, domain gaps, or seniority mismatches."
}}

Rules:
- is_match is true when the candidate's technical skills or stacks align with the role.
- Stack familiarity can support a match even when not every job keyword is listed.
- Treat familiarity percentages as the candidate's self-rated proficiency. Weight high-familiarity skills more when judging fit; low-familiarity overlaps are weaker evidence.
- is_match is false when the job requires or heavily emphasizes any disqualifying skill, stack, or tool/platform, when the role targets a different domain, or when it needs skills/stacks the profile does not support.
- Disqualifying skills, stacks, and tools/platforms override positive overlap: a strong technical match still fails if the role centers on a disqualifying item.
- matched_skills must come from the technical skills or stacks lists only, never from disqualifying lists.
- missing_skills should list only important gaps from the job description.
- match_score is 0-100 indicating overall fit.
- match_paragraph and mismatch_paragraph must be written in second person ("you") for the candidate.
- Always provide both paragraphs, even when fit is strong or weak.
"""

    try:
        result = client.generate_json(
            prompt,
            model=client.fast_model,
            system=MATCH_SYSTEM_PROMPT,
            temperature=0.1,
            max_attempts=2,
        )
        matched_skills = _normalize_result_list(result.get("matched_skills"))
        missing_skills = _normalize_result_list(result.get("missing_skills"))
        reason = str(result.get("reason") or "").strip()
        is_match = bool(result.get("is_match"))
        match_paragraph, mismatch_paragraph = _build_match_paragraphs(
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            reason=reason,
            is_match=is_match,
            match_paragraph=str(result.get("match_paragraph") or ""),
            mismatch_paragraph=str(result.get("mismatch_paragraph") or ""),
        )
        return {
            "is_match": is_match,
            "match_score": float(result.get("match_score") or 0),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "reason": reason,
            "match_paragraph": match_paragraph,
            "mismatch_paragraph": mismatch_paragraph,
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
    llm: LLMClient | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze one job, store fit metadata, and set workflow status from the threshold."""
    updated = dict(job)
    analysis = analyze_job_match(updated, profile, llm=llm)
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
    llm: LLMClient | None = None,
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
            llm=llm,
        )
        classified.append(updated)
    return classified


def analyze_jobs_with_threshold(
    jobs: list[dict[str, Any]],
    profile: dict[str, Any] | None,
    min_match_score: float,
    *,
    llm: LLMClient | None = None,
    on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
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
        if should_continue is not None and not should_continue():
            break

        if on_progress:
            on_progress(index, total, job)

        previous_status = job.get("workflow_status") or "new"
        updated, _analysis = apply_profile_match_to_job(
            job,
            profile,
            min_match_score=threshold,
            llm=llm,
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
