"""Tests for user profile parsing and readiness checks."""

from job_apply_ai.storage.user_profile import (
    DEFAULT_FAMILIARITY,
    parse_multiline_list,
    parse_professional_titles,
    parse_smtp_accounts_from_form,
    pick_professional_title,
    parse_projects_text,
    parse_work_experience_text,
    profile_from_form,
    profile_is_ready,
    profile_to_form_fields,
    profile_to_text,
    skill_names,
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


def test_profile_to_text_includes_familiarity():
    profile = profile_from_form(
        {
            "full_name": "Jane Doe",
            "technical_skills_json": '[{"name": "Python", "familiarity": 85}]',
        }
    )
    text = profile_to_text(profile)
    assert "Python (85%)" in text


def test_profile_from_form_legacy_strings_get_default_familiarity():
    profile = profile_from_form(
        {
            "full_name": "Jane Doe",
            "technical_skills": "Python\nFlask",
        }
    )
    assert skill_names(profile["technical_skills"]) == ["Python", "Flask"]
    assert all(item["familiarity"] == DEFAULT_FAMILIARITY for item in profile["technical_skills"])


def test_parse_smtp_accounts_from_form_preserves_existing_password():
    class FakeForm(dict):
        def getlist(self, key):
            values = {
                "smtp_id": ["acc1"],
                "smtp_provider": ["gmail"],
                "smtp_email": ["user@gmail.com"],
                "smtp_password": [""],
                "smtp_label": ["Work"],
                "smtp_host": [],
                "smtp_port": [],
                "smtp_use_tls": [],
            }
            return values.get(key, [])

    existing = {
        "smtp_accounts": [
            {
                "id": "acc1",
                "provider": "gmail",
                "email": "user@gmail.com",
                "password": "stored-secret",
                "label": "Work",
                "is_default": True,
            }
        ]
    }
    accounts = parse_smtp_accounts_from_form(FakeForm({"smtp_default_id": "acc1"}), existing)
    assert len(accounts) == 1
    assert accounts[0]["password"] == "stored-secret"


def test_profile_from_form_parses_json_fields():
    profile = profile_from_form(
        {
            "full_name": "Jane Doe",
            "technical_skills_json": '[{"name": "Python", "familiarity": 90}, {"name": "Flask", "familiarity": 75}]',
            "minor_skills_json": '[{"name": "Redis", "familiarity": 60}, "Celery"]',
            "disqualifying_tools_platforms_json": '[{"name": "Salesforce", "familiarity": 100}]',
            "disqualifying_stacks_json": '[{"name": "LAMP", "familiarity": 100}]',
            "work_experience_json": '[{"role": "Developer", "company": "Acme", "period": "2020", "bullets": ["Built APIs"]}]',
            "personal_projects_json": '[{"name": "Side Project", "description": "A demo", "bullets": ["Used Flask"]}]',
        }
    )
    assert profile["technical_skills"] == [
        {"name": "Python", "familiarity": 90},
        {"name": "Flask", "familiarity": 75},
    ]
    assert profile["minor_skills"] == [
        {"name": "Redis", "familiarity": 60},
        {"name": "Celery", "familiarity": DEFAULT_FAMILIARITY},
    ]
    assert profile["disqualifying_tools_platforms"] == [
        {"name": "Salesforce", "familiarity": 100},
    ]
    assert profile["disqualifying_stacks"] == [
        {"name": "LAMP", "familiarity": 100},
    ]
    assert profile["work_experience"][0]["role"] == "Developer"
    assert profile["work_experience"][0]["bullets"] == ["Built APIs"]
    assert profile["personal_projects"][0]["name"] == "Side Project"


def test_profile_to_form_fields_returns_lists():
    profile = profile_from_form(
        {
            "full_name": "Jane Doe",
            "technical_skills_json": '["Python"]',
            "work_experience_json": '[{"role": "Dev", "company": "", "period": "", "bullets": []}]',
        }
    )
    form = profile_to_form_fields(profile)
    assert form["technical_skills_list"] == [{"name": "Python", "familiarity": DEFAULT_FAMILIARITY}]
    assert form["work_experience_list"][0]["role"] == "Dev"


def test_profile_from_form_saves_multiple_smtp_accounts():
    class FakeForm(dict):
        def getlist(self, key):
            values = {
                "smtp_id": ["acc1", "acc2"],
                "smtp_provider": ["gmail", "hotmail"],
                "smtp_email": ["a@gmail.com", "b@hotmail.com"],
                "smtp_password": ["pass1", "pass2"],
                "smtp_label": ["Gmail", "Hotmail"],
                "smtp_host": [],
                "smtp_port": [],
                "smtp_use_tls": [],
            }
            return values.get(key, [])

    profile = profile_from_form(
        FakeForm(
            {
                "full_name": "Jane Doe",
                "smtp_default_id": "acc2",
            }
        )
    )
    assert len(profile["smtp_accounts"]) == 2
    assert profile["smtp_accounts"][1]["is_default"] is True
    assert profile["smtp_accounts"][1]["email"] == "b@hotmail.com"


def test_parse_smtp_accounts_from_form_keeps_oauth_accounts():
    class FakeForm(dict):
        def getlist(self, key):
            return []

    existing = {
        "smtp_accounts": [
            {
                "id": "oauth1",
                "provider": "gmail",
                "auth_type": "oauth",
                "email": "oauth@gmail.com",
                "oauth_refresh_token": "refresh",
                "is_default": True,
            }
        ]
    }
    accounts = parse_smtp_accounts_from_form(FakeForm({}), existing)
    assert len(accounts) == 1
    assert accounts[0]["auth_type"] == "oauth"
