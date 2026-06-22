"""Sort job listings for list views."""

from __future__ import annotations

from typing import Any, Callable

DEFAULT_JOB_SORT = "id_desc"

JOB_SORT_OPTIONS: dict[str, str] = {
    "match_desc": "AI Match Rating (high to low)",
    "match_asc": "AI Match Rating (low to high)",
    "posted_asc": "Posted (newest first)",
    "posted_desc": "Posted (oldest first)",
    "title_asc": "Title (A–Z)",
    "title_desc": "Title (Z–A)",
    "company_asc": "Company (A–Z)",
    "company_desc": "Company (Z–A)",
    "updated_desc": "Recently updated",
    "updated_asc": "Oldest updated",
    "id_desc": "Recently added",
    "id_asc": "Oldest added",
}


def validate_job_sort(value: str | None) -> str:
    """Return a supported sort key, falling back to the default."""
    if value and value in JOB_SORT_OPTIONS:
        return value
    return DEFAULT_JOB_SORT


def get_profile_match_analysis(job: dict[str, Any]) -> dict[str, Any] | None:
    """Return stored AI/heuristic profile fit analysis for a job, if any."""
    categories = job.get("matched_categories") or {}
    fit = categories.get("Profile Fit")
    return fit if isinstance(fit, dict) else None


def get_profile_match_score(job: dict[str, Any]) -> float | None:
    """Return the profile match score (0–100) when analysis exists."""
    analysis = get_profile_match_analysis(job)
    if not analysis or analysis.get("method") == "skipped":
        return None
    try:
        return float(analysis.get("match_score") or 0)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sort_key_match_desc(job: dict[str, Any]) -> tuple[int, float, int]:
    score = get_profile_match_score(job)
    if score is None:
        return (1, 0.0, -_int(job.get("id")))
    return (0, -score, -_int(job.get("id")))


def _sort_key_match_asc(job: dict[str, Any]) -> tuple[int, float, int]:
    score = get_profile_match_score(job)
    if score is None:
        return (1, 0.0, _int(job.get("id")))
    return (0, score, _int(job.get("id")))


_SORT_KEY_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "match_desc": _sort_key_match_desc,
    "match_asc": _sort_key_match_asc,
    "posted_asc": lambda job: (
        1 if not str(job.get("posted_days_ago", "")).strip() else 0,
        _int(job.get("posted_days_ago"), 999999),
        _int(job.get("id")),
    ),
    "posted_desc": lambda job: (
        1 if not str(job.get("posted_days_ago", "")).strip() else 0,
        -_int(job.get("posted_days_ago")),
        -_int(job.get("id")),
    ),
    "title_asc": lambda job: (_text(job.get("title")), _int(job.get("id"))),
    "title_desc": lambda job: (_text(job.get("title")), _int(job.get("id"))),
    "company_asc": lambda job: (_text(job.get("company")), _text(job.get("title"))),
    "company_desc": lambda job: (_text(job.get("company")), _text(job.get("title"))),
    "updated_desc": lambda job: (_text(job.get("updated_at")), _int(job.get("id"))),
    "updated_asc": lambda job: (_text(job.get("updated_at")), _int(job.get("id"))),
    "id_desc": lambda job: _int(job.get("id")),
    "id_asc": lambda job: _int(job.get("id")),
}

_REVERSE_SORT_KEYS = frozenset({
    "title_desc",
    "company_desc",
    "updated_desc",
    "id_desc",
})


def sort_jobs(jobs: list[dict[str, Any]], sort_by: str | None = None) -> list[dict[str, Any]]:
    """Return a copy of jobs sorted by the requested key."""
    if not jobs:
        return []

    key = validate_job_sort(sort_by)
    key_builder = _SORT_KEY_BUILDERS[key]
    reverse = key in _REVERSE_SORT_KEYS
    return sorted(jobs, key=key_builder, reverse=reverse)
