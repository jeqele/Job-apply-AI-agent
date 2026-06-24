"""Tests for chat prompt context helpers."""

from job_apply_ai.cv_modifier.chat_context import (
    build_job_context,
    build_profile_context,
    cv_content_to_preview_lines,
    format_numbered_cv_preview,
    normalize_preview_line,
    normalize_preview_lines,
    preview_lines_to_content,
    resolve_cv_preview_lines,
    resolve_effective_tailored_content,
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


def test_resolve_cv_preview_lines_keeps_custom_order_when_content_matches():
    content = {
        "professional_title": "Backend Engineer",
        "professional_summary": "Builds APIs with Python.",
    }
    generated = cv_content_to_preview_lines(content, "Jane Doe")
    reordered = list(reversed(generated))
    resolved = resolve_cv_preview_lines(content, "Jane Doe", stored_lines=reordered)
    assert resolved == reordered


def test_resolve_cv_preview_lines_falls_back_when_lines_change():
    content = {
        "professional_title": "Backend Engineer",
        "professional_summary": "Builds APIs with Python.",
    }
    generated = cv_content_to_preview_lines(content, "Jane Doe")
    stale = generated + [{"text": "Extra line", "kind": "text"}]
    resolved = resolve_cv_preview_lines(content, "Jane Doe", stored_lines=stale)
    assert resolved == generated


def test_resolve_cv_preview_lines_uses_customized_lines():
    content = {
        "professional_title": "Backend Engineer",
        "professional_summary": "Builds APIs with Python.",
    }
    custom = [
        {"text": "Custom Name", "kind": "name"},
        {"text": "Edited summary only.", "kind": "text"},
    ]
    resolved = resolve_cv_preview_lines(
        content,
        "Jane Doe",
        stored_lines=custom,
        customized=True,
    )
    assert resolved == custom


def test_resolve_effective_tailored_content_uses_customized_preview():
    content = {
        "professional_title": "Backend Engineer",
        "professional_summary": "Original summary.",
        "technical_skills": ["Python"],
        "tools_platforms": [],
        "experience_highlights": [],
        "personal_projects": [],
        "soft_skills": [],
        "languages": [],
        "job_matched_skills": [],
        "job_skills_not_in_cv": [],
    }
    customized_lines = cv_content_to_preview_lines(content, "Jane Doe")
    for line in customized_lines:
        if line.get("kind") == "text" and "Original summary" in line.get("text", ""):
            line["text"] = "Customized summary from preview."
            break

    effective = resolve_effective_tailored_content(
        content,
        "Jane Doe",
        stored_lines=customized_lines,
        customized=True,
    )
    assert effective["professional_summary"] == "Customized summary from preview."


def test_resolve_effective_tailored_content_returns_copy_when_not_customized():
    content = {"professional_summary": "Same summary."}
    effective = resolve_effective_tailored_content(content, "Jane Doe")
    assert effective == content
    assert effective is not content


def test_normalize_preview_line_sanitizes_kind_and_variant():
    line = normalize_preview_line({
        "text": "Python",
        "kind": "skills",
        "variant": "matched",
        "extra": "ignored",
    })
    assert line == {"text": "Python", "kind": "skills", "variant": "matched"}

    invalid = normalize_preview_line({"text": "X", "kind": "unknown"})
    assert invalid == {"text": "X", "kind": "text"}


def test_normalize_preview_lines_skips_invalid_entries():
    lines = normalize_preview_lines([
        {"text": "Valid", "kind": "text"},
        None,
        "bad",
        {"text": "Also valid", "kind": "bullet"},
    ])
    assert lines == [
        {"text": "Valid", "kind": "text"},
        {"text": "Also valid", "kind": "bullet"},
    ]


def test_preview_lines_to_content_updates_summary_and_skills():
    content = {
        "professional_title": "Old Title",
        "professional_summary": "Old summary.",
        "technical_skills": ["Java"],
        "experience_highlights": [],
    }
    preview_lines = [
        {"text": "Jane Doe", "kind": "name"},
        {"text": "Backend Engineer", "kind": "title"},
        {"text": "Professional Summary", "kind": "section"},
        {"text": "New summary text.", "kind": "text"},
        {"text": "Technical Skills", "kind": "section"},
        {"text": "• Python • Flask", "kind": "skills"},
        {"text": "Experience Highlights", "kind": "section"},
        {"text": "Developer", "kind": "role"},
        {"text": "Acme · 2020-2024", "kind": "meta"},
        {"text": "• Built APIs", "kind": "bullet"},
    ]
    updated = preview_lines_to_content(content, preview_lines, "Jane Doe")
    assert updated["professional_title"] == "Backend Engineer"
    assert updated["professional_summary"] == "New summary text."
    assert updated["technical_skills"] == ["Python", "Flask"]
    assert updated["experience_highlights"][0]["role"] == "Developer"
    assert updated["experience_highlights"][0]["bullets"] == ["Built APIs"]
