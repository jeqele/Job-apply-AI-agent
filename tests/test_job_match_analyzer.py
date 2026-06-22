"""Tests for profile skill fields and job fit analysis."""

from job_apply_ai.cv_modifier.job_match_analyzer import (
    NOT_MATCH_STATUS,
    classify_jobs_by_profile_match,
    heuristic_job_match,
    job_meets_threshold,
    normalize_min_match_score,
    profile_has_matchable_skills,
)
from job_apply_ai.storage.user_profile import profile_from_form, profile_to_form_fields


def test_profile_from_form_parses_minor_skills_and_stacks():
    profile = profile_from_form(
        {
            "full_name": "Jane Doe",
            "technical_skills": "Python\nFlask",
            "minor_skills": "Redis, Celery",
            "stacks": "Python/Django/PostgreSQL\nMERN",
        }
    )
    assert profile["technical_skills"] == ["Python", "Flask"]
    assert profile["minor_skills"] == ["Redis", "Celery"]
    assert profile["stacks"] == ["Python/Django/PostgreSQL", "MERN"]

    form = profile_to_form_fields(profile)
    assert "Redis" in form["minor_skills"]
    assert "MERN" in form["stacks"]


def test_profile_has_matchable_skills():
    assert not profile_has_matchable_skills({"full_name": "Jane Doe"})
    assert profile_has_matchable_skills({"minor_skills": ["Docker"]})


def test_heuristic_job_match_detects_overlap():
    profile = {
        "technical_skills": ["Python", "Flask", "PostgreSQL"],
        "minor_skills": ["Redis"],
        "stacks": ["Python/Django/PostgreSQL"],
    }
    matching_job = {
        "title": "Backend Engineer",
        "description": "Build Python APIs with Flask and PostgreSQL. Redis experience is a plus.",
    }
    mismatch_job = {
        "title": "iOS Developer",
        "description": "Swift and UIKit required. Objective-C experience preferred.",
    }

    match_result = heuristic_job_match(matching_job, profile)
    mismatch_result = heuristic_job_match(mismatch_job, profile)

    assert match_result["is_match"] is True
    assert "python" in match_result["matched_skills"]
    assert mismatch_result["is_match"] is False


def test_classify_jobs_routes_non_matches_to_folder():
    profile = {
        "technical_skills": ["Python", "Django"],
        "minor_skills": [],
        "stacks": [],
    }
    jobs = [
        {
            "title": "Python Developer",
            "description": "Python and Django web development.",
            "workflow_status": "new",
        },
        {
            "title": "Embedded C Engineer",
            "description": "Firmware development in C for microcontrollers.",
            "workflow_status": "new",
        },
    ]

    classified = classify_jobs_by_profile_match(jobs, profile, min_match_score=50)

    assert classified[0]["workflow_status"] == "new"
    assert classified[1]["workflow_status"] == NOT_MATCH_STATUS
    assert classified[1]["matched_categories"]["Profile Fit"]["is_match"] is False


def test_job_meets_threshold_uses_match_score():
    assert job_meets_threshold({"match_score": 72, "method": "ai"}, 50) is True
    assert job_meets_threshold({"match_score": 42, "method": "ai"}, 50) is False
    assert job_meets_threshold({"match_score": 0, "method": "skipped"}, 50) is True


def test_normalize_min_match_score():
    assert normalize_min_match_score("75") == 75.0
    assert normalize_min_match_score("150") == 100.0
    assert normalize_min_match_score("bad") == 50.0
