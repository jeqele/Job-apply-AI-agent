"""Tests for job list sorting."""

from job_apply_ai.job_sort import (
    DEFAULT_JOB_SORT,
    get_profile_match_score,
    sort_jobs,
    validate_job_sort,
)


def _job(
    job_id: int,
    *,
    title: str = "Engineer",
    company: str = "Acme",
    posted_days_ago: str = "",
    match_score: float | None = None,
) -> dict:
    job: dict = {
        "id": job_id,
        "title": title,
        "company": company,
        "posted_days_ago": posted_days_ago,
        "updated_at": f"2026-01-{job_id:02d}T00:00:00",
    }
    if match_score is not None:
        job["matched_categories"] = {
            "Profile Fit": {"match_score": match_score, "method": "ai"},
        }
    return job


def test_validate_job_sort_falls_back_to_default():
    assert validate_job_sort(None) == DEFAULT_JOB_SORT
    assert validate_job_sort("not-a-sort") == DEFAULT_JOB_SORT
    assert validate_job_sort("match_desc") == "match_desc"


def test_get_profile_match_score():
    assert get_profile_match_score(_job(1, match_score=82.5)) == 82.5
    assert get_profile_match_score(_job(2)) is None


def test_sort_by_match_desc_puts_unanalyzed_last():
    jobs = [
        _job(1, match_score=40),
        _job(2),
        _job(3, match_score=90),
    ]
    sorted_jobs = sort_jobs(jobs, "match_desc")
    assert [job["id"] for job in sorted_jobs] == [3, 1, 2]


def test_sort_by_title_asc():
    jobs = [
        _job(1, title="Zebra"),
        _job(2, title="Alpha"),
    ]
    sorted_jobs = sort_jobs(jobs, "title_asc")
    assert [job["title"] for job in sorted_jobs] == ["Alpha", "Zebra"]


def test_sort_by_title_desc():
    jobs = [
        _job(1, title="Zebra"),
        _job(2, title="Alpha"),
    ]
    sorted_jobs = sort_jobs(jobs, "title_desc")
    assert [job["title"] for job in sorted_jobs] == ["Zebra", "Alpha"]


def test_sort_by_posted_asc_newest_first():
    jobs = [
        _job(1, posted_days_ago="14"),
        _job(2, posted_days_ago="2"),
        _job(3, posted_days_ago="7"),
    ]
    sorted_jobs = sort_jobs(jobs, "posted_asc")
    assert [job["id"] for job in sorted_jobs] == [2, 3, 1]
