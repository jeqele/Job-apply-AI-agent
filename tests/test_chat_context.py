"""Tests for chat prompt context helpers."""

from job_apply_ai.cv_modifier.chat_context import (
    build_job_context,
    build_profile_context,
    cv_content_to_preview_lines,
    format_numbered_cv_preview,
)
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


def test_cv_content_to_preview_lines_assigns_stable_line_numbers():
    content = {
        "professional_title": "Backend Engineer",
        "professional_summary": "Builds APIs with Python.",
        "job_matched_skills": ["Python", "PostgreSQL"],
        "experience_highlights": [
            {
                "role": "Developer",
                "company": "Acme",
                "period": "2020-2024",
                "bullets": ["Built APIs"],
            }
        ],
    }
    lines = cv_content_to_preview_lines(content, "Jane Doe")
    numbered = format_numbered_cv_preview(lines)

    assert lines[0]["text"] == "Jane Doe"
    assert lines[1]["text"] == "Backend Engineer"
    assert lines[2]["kind"] == "section"
    assert "Builds APIs with Python." in lines[3]["text"]
    assert " 1 | Jane Doe" in numbered
    assert numbered.strip().endswith("22 | None")
    assert any(line["kind"] == "bullet" and "Built APIs" in line["text"] for line in lines)
