"""Tests for cover letter generation helpers."""

from job_apply_ai.cv_modifier.cover_letter_generator import CoverLetterGenerator


def test_cover_letter_normalize_fills_defaults():
    profile = {"full_name": "Jane Doe", "email": "jane@example.com", "phone": "123"}
    job = {"title": "Engineer", "company": "Acme Corp"}
    content = {
        "greeting": "Dear Team,",
        "body_paragraphs": ["I am excited to apply.", "I bring strong Python skills."],
    }
    normalized = CoverLetterGenerator.normalize(content, profile, job)
    assert normalized["recipient_company"] == "Acme Corp"
    assert normalized["signature_name"] == "Jane Doe"
    assert len(normalized["body_paragraphs"]) == 2
    assert normalized["candidate_email"] == "jane@example.com"
