"""Tests for profile skill fields and job fit analysis."""

from job_apply_ai.cv_modifier.job_match_analyzer import (
    NOT_MATCH_STATUS,
    _build_match_paragraphs,
    classify_jobs_by_profile_match,
    heuristic_job_match,
    job_meets_threshold,
    normalize_min_match_score,
    profile_has_matchable_skills,
)
from job_apply_ai.storage.user_profile import (
    DEFAULT_FAMILIARITY,
    profile_from_form,
    profile_to_form_fields,
    skill_names,
)


def test_profile_from_form_parses_minor_skills_and_stacks():
    profile = profile_from_form(
        {
            "full_name": "Jane Doe",
            "technical_skills": "Python\nFlask",
            "minor_skills": "Redis, Celery",
            "stacks": "Python/Django/PostgreSQL\nMERN",
        }
    )
    assert skill_names(profile["technical_skills"]) == ["Python", "Flask"]
    assert skill_names(profile["minor_skills"]) == ["Redis", "Celery"]
    assert skill_names(profile["stacks"]) == ["Python/Django/PostgreSQL", "MERN"]

    form = profile_to_form_fields(profile)
    assert any(item["name"] == "Redis" for item in form["minor_skills_list"])
    assert any(item["name"] == "MERN" for item in form["stacks_list"])


def test_profile_has_matchable_skills():
    assert not profile_has_matchable_skills({"full_name": "Jane Doe"})
    assert not profile_has_matchable_skills({"minor_skills": ["Docker"]})
    assert profile_has_matchable_skills({"technical_skills": ["Python"]})
    assert profile_has_matchable_skills({"stacks": ["MERN"]})


def test_heuristic_job_match_weights_familiarity():
    strong_profile = {
        "technical_skills": [
            {"name": "Python", "familiarity": 95},
            {"name": "Rust", "familiarity": 5},
        ],
        "minor_skills": [],
        "stacks": [],
    }
    weak_profile = {
        "technical_skills": [
            {"name": "Python", "familiarity": 20},
            {"name": "Rust", "familiarity": 80},
        ],
        "minor_skills": [],
        "stacks": [],
    }
    job = {
        "title": "Python Developer",
        "description": "Python backend development.",
    }

    strong_result = heuristic_job_match(job, strong_profile)
    weak_result = heuristic_job_match(job, weak_profile)

    assert strong_result["is_match"] is True
    assert weak_result["match_score"] < strong_result["match_score"]


def test_heuristic_job_match_detects_overlap():
    profile = {
        "technical_skills": ["Python", "Flask", "PostgreSQL", "Redis"],
        "minor_skills": ["Java"],
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
    disqualifying_job = {
        "title": "Java Backend Engineer",
        "description": "Strong Java and Spring Boot experience required.",
    }

    match_result = heuristic_job_match(matching_job, profile)
    mismatch_result = heuristic_job_match(mismatch_job, profile)
    disqualified_result = heuristic_job_match(disqualifying_job, profile)

    assert match_result["is_match"] is True
    assert "python" in match_result["matched_skills"]
    assert match_result["match_paragraph"]
    assert match_result["mismatch_paragraph"]
    assert mismatch_result["is_match"] is False
    assert mismatch_result["mismatch_paragraph"]
    assert disqualified_result["is_match"] is False
    assert "java" in disqualified_result["missing_skills"]


def test_build_match_paragraphs_fills_missing_text():
    match_para, mismatch_para = _build_match_paragraphs(
        matched_skills=["Python", "Django"],
        missing_skills=["Kubernetes"],
        reason="Strong backend overlap.",
        is_match=True,
    )
    assert "Python" in match_para
    assert "Kubernetes" in mismatch_para


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
