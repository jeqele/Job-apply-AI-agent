"""Tests for chat prompt context helpers."""

from job_apply_ai.cv_modifier.chat_context import build_job_context, build_profile_context
from job_apply_ai.storage.user_profile import normalize_profile


def test_build_job_context_includes_description():
    job = {
        "title": "Backend Engineer",
        "company": "Acme",
        "location": "London",
        "description": "Build APIs with Python and PostgreSQL.",
    }
    context = build_job_context(job)
    assert "Backend Engineer" in context
    assert "Build APIs with Python and PostgreSQL." in context


def test_build_profile_context_includes_skills_and_experience():
    profile = normalize_profile({
        "full_name": "Jane Doe",
        "professional_title": "Software Engineer",
        "technical_skills": ["Python", "SQL"],
        "work_experience": [
            {
                "role": "Developer",
                "company": "Acme",
                "period": "2020-2024",
                "bullets": ["Built APIs"],
            }
        ],
    })
    context = build_profile_context(profile)
    assert "Jane Doe" in context
    assert "Python" in context
    assert "Built APIs" in context
