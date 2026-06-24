"""Tests for ATS-friendly CV analysis."""

from unittest.mock import MagicMock

from job_apply_ai.cv_modifier.ats_friendly_analyzer import (
    apply_suggestion_to_content,
    normalize_ats_analysis,
    normalize_ats_score,
    update_suggestion_status,
    ATSFriendlyAnalyzer,
)


def test_normalize_ats_score_clamps():
    assert normalize_ats_score(105) == 100
    assert normalize_ats_score(-5) == 0
    assert normalize_ats_score(72.4) == 72
    assert normalize_ats_score("bad") == 0


def test_normalize_ats_analysis_assigns_suggestion_ids():
    raw = {
        "ats_score": 68,
        "score_summary": "Good keyword coverage.",
        "keyword_bank": ["Python"],
        "matched_keywords": ["Python"],
        "missing_keywords": ["Kubernetes"],
        "suggestions": [
            {
                "title": "Strengthen summary",
                "description": "Add cloud keywords.",
                "rationale": "ATS match",
                "category": "summary",
                "priority": "high",
                "changes": {"professional_summary": "Updated summary."},
            }
        ],
    }
    analysis = normalize_ats_analysis(raw)
    assert analysis["ats_score"] == 68
    assert len(analysis["suggestions"]) == 1
    assert analysis["suggestions"][0]["id"]
    assert analysis["suggestions"][0]["status"] == "pending"


def test_apply_suggestion_to_content_updates_summary():
    current = {
        "professional_title": "Engineer",
        "professional_summary": "Old summary.",
        "technical_skills": ["Python"],
        "tools_platforms": [],
        "experience_highlights": [],
        "personal_projects": [],
        "soft_skills": [],
        "languages": [],
        "job_matched_skills": [],
        "job_skills_not_in_cv": [],
    }
    suggestion = {
        "changes": {"professional_summary": "Results-driven engineer with Python expertise."},
    }
    updated = apply_suggestion_to_content(current, suggestion)
    assert updated["professional_summary"] == "Results-driven engineer with Python expertise."
    assert updated["professional_title"] == "Engineer"


def test_update_suggestion_status():
    analysis = normalize_ats_analysis(
        {
            "ats_score": 50,
            "score_summary": "Needs work",
            "keyword_bank": [],
            "matched_keywords": [],
            "missing_keywords": [],
            "suggestions": [
                {
                    "id": "abc123",
                    "title": "Fix bullets",
                    "description": "Add metrics",
                    "rationale": "ATS",
                    "category": "experience",
                    "changes": {},
                }
            ],
        }
    )
    updated = update_suggestion_status(analysis, "abc123", status="denied")
    assert updated["suggestions"][0]["status"] == "denied"


def test_analyzer_analyze_calls_ollama():
    ollama = MagicMock()
    ollama.is_available.return_value = True
    ollama.validate_models.return_value = {"main": "test-model"}
    ollama.generate_json.return_value = {
        "ats_score": 81,
        "score_summary": "Strong alignment.",
        "keyword_bank": ["Python", "SQL"],
        "matched_keywords": ["Python"],
        "missing_keywords": ["SQL"],
        "formatting_notes": ["Use standard headings"],
        "trade_offs": "",
        "suggestions": [],
    }

    analyzer = ATSFriendlyAnalyzer(llm=ollama)
    result = analyzer.analyze(
        job={"title": "Developer", "description": "Python and SQL required"},
        cv_content={"professional_summary": "Python developer."},
        profile={"full_name": "Jane Doe", "technical_skills": ["Python"]},
    )

    assert result["ats_score"] == 81
    assert result["method"] == "ai"
    ollama.generate_json.assert_called_once()


def test_analyzer_apply_suggestion_calls_llm_and_merges_changes():
    ollama = MagicMock()
    ollama.is_available.return_value = True
    ollama.validate_models.return_value = {"main": "test-model"}
    ollama.generate_json.return_value = {
        "reply": "Updated the professional summary.",
        "changes": {"professional_summary": "Cloud-focused Python engineer."},
    }

    current = {
        "professional_title": "Engineer",
        "professional_summary": "Old summary.",
        "technical_skills": ["Python"],
        "tools_platforms": [],
        "experience_highlights": [],
        "personal_projects": [],
        "soft_skills": [],
        "languages": [],
        "job_matched_skills": [],
        "job_skills_not_in_cv": [],
    }
    suggestion = {
        "id": "s1",
        "title": "Strengthen summary",
        "description": "Add cloud keywords naturally.",
        "rationale": "Better ATS keyword match",
        "category": "summary",
        "changes": {"professional_summary": "Stale hint from analysis."},
    }

    analyzer = ATSFriendlyAnalyzer(llm=ollama)
    updated = analyzer.apply_suggestion(
        job={"title": "Cloud Engineer", "description": "Python and AWS"},
        cv_content=current,
        profile={"full_name": "Jane Doe", "technical_skills": ["Python"]},
        suggestion=suggestion,
    )

    assert updated["professional_summary"] == "Cloud-focused Python engineer."
    assert updated["professional_title"] == "Engineer"
    ollama.generate_json.assert_called_once()
    prompt = ollama.generate_json.call_args[0][0]
    assert "Strengthen summary" in prompt
    assert "CURRENT CV CONTENT" in prompt
