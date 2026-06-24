"""Tests for AI job title suggestions."""

from job_apply_ai.cv_modifier.job_title_suggester import (
    heuristic_job_title_suggestions,
    suggest_job_titles,
)


class FakeLLM:
    provider_label = "FakeLLM"
    fast_model = "fast"
    main_model = "main"
    num_predict = 1024

    def __init__(self, *, available: bool = True, titles: list[str] | None = None, error: Exception | None = None):
        self.available = available
        self.titles = titles or []
        self.error = error

    def is_available(self) -> bool:
        return self.available

    def validate_models(self) -> dict[str, str]:
        return {"fast": self.fast_model, "main": self.main_model}

    def generate_json(self, prompt, **kwargs):
        if self.error:
            raise self.error
        return {"titles": self.titles}


def test_heuristic_job_title_suggestions_uses_profile_titles_and_roles():
    profile = {
        "full_name": "Jane Doe",
        "professional_title": "Backend Engineer, Python Developer",
        "work_experience": [
            {"role": "Senior Software Engineer", "company": "Acme", "period": "2020–2024", "bullets": []},
        ],
    }
    titles = heuristic_job_title_suggestions(profile)
    assert titles == [
        "Backend Engineer",
        "Python Developer",
        "Senior Software Engineer",
    ]


def test_suggest_job_titles_skips_incomplete_profile():
    result = suggest_job_titles({"full_name": "Jane Doe"}, llm=FakeLLM())
    assert result["titles"] == []
    assert result["method"] == "skipped"
    assert "profile" in result["error"].lower()


def test_suggest_job_titles_uses_ai_when_available():
    profile = {
        "full_name": "Jane Doe",
        "personal_summary": "Backend engineer with Python experience.",
        "technical_skills": [{"name": "Python", "familiarity": 90}],
    }
    llm = FakeLLM(titles=["Python Developer", "Backend Engineer"])
    result = suggest_job_titles(profile, llm=llm)
    assert result["method"] == "ai"
    assert result["titles"] == ["Python Developer", "Backend Engineer"]


def test_suggest_job_titles_falls_back_when_llm_unavailable():
    profile = {
        "full_name": "Jane Doe",
        "personal_summary": "Backend engineer.",
        "professional_title": "Software Engineer",
        "technical_skills": [{"name": "Python", "familiarity": 90}],
    }
    result = suggest_job_titles(profile, llm=FakeLLM(available=False))
    assert result["method"] == "heuristic"
    assert result["titles"] == ["Software Engineer"]


def test_suggest_job_titles_falls_back_on_ai_error():
    profile = {
        "full_name": "Jane Doe",
        "personal_summary": "Backend engineer.",
        "professional_title": "Software Engineer",
        "technical_skills": [{"name": "Python", "familiarity": 90}],
    }
    llm = FakeLLM(error=RuntimeError("model offline"))
    result = suggest_job_titles(profile, llm=llm)
    assert result["method"] == "heuristic"
    assert result["titles"] == ["Software Engineer"]
    assert "model offline" in result["error"]
