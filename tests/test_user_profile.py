"""Tests for user profile parsing and readiness checks."""

from job_apply_ai.storage.user_profile import (
    parse_multiline_list,
    parse_professional_titles,
    pick_professional_title,
    parse_projects_text,
    parse_work_experience_text,
    profile_from_form,
    profile_is_ready,
    profile_to_text,
)


def test_parse_multiline_list():
    assert parse_multiline_list("Python\nJava, SQL") == ["Python", "Java", "SQL"]


def test_parse_professional_titles():
    assert parse_professional_titles("") == []
    assert parse_professional_titles("Developer") == ["Developer"]
    assert parse_professional_titles("Full-Stack Developer, Backend Engineer") == [
        "Full-Stack Developer",
        "Backend Engineer",
    ]


def test_pick_professional_title():
    titles = ["Full-Stack Developer", "Backend Engineer", "DevOps Engineer"]
    job = {"title": "Senior Backend Engineer", "description": "Python APIs and PostgreSQL"}
    assert pick_professional_title(titles, job) == "Backend Engineer"
    assert pick_professional_title(["Solo Title"], job) == "Solo Title"


def test_profile_to_text_lists_multiple_titles():
    profile = profile_from_form(
        {
            "full_name": "Jane Doe",
            "professional_title": "Full-Stack Developer, Backend Engineer",
        }
    )
    text = profile_to_text(profile)
    assert "Professional Titles (choose best fit for the job)" in text
    assert "Full-Stack Developer, Backend Engineer" in text


def test_parse_work_experience_text():
    text = """
Senior Developer | Acme Corp | 2021 - Present
- Built APIs
- Led team

Junior Developer | Beta Inc | 2018 - 2020
- Fixed bugs
"""
    entries = parse_work_experience_text(text)
    assert len(entries) == 2
    assert entries[0]["role"] == "Senior Developer"
    assert entries[0]["company"] == "Acme Corp"
    assert entries[0]["bullets"] == ["Built APIs", "Led team"]


def test_profile_is_ready_requires_name_and_content():
    assert not profile_is_ready({"full_name": "Jane Doe"})
    assert profile_is_ready(
        {
            "full_name": "Jane Doe",
            "technical_skills": ["Python"],
        }
    )


def test_profile_to_text_includes_sections():
    profile = profile_from_form(
        {
            "full_name": "Jane Doe",
            "professional_title": "Developer",
            "technical_skills": "Python\nFlask",
            "work_experience_text": "Developer | Acme | 2020\n- Built APIs",
        }
    )
    text = profile_to_text(profile)
    assert "Jane Doe" in text
    assert "Technical Skills" in text
    assert "Work Experience" in text
