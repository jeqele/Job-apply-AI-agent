"""Tests for PDF CV and cover letter export."""

from __future__ import annotations

import os

from job_apply_ai.cv_modifier.chat_context import cv_content_to_preview_lines
from job_apply_ai.cv_modifier.pdf_builder import (
    CoverLetterPdfBuilder,
    CVPdfBuilder,
    build_cover_letter_pdf,
    build_cv_pdf,
    pdf_path_for_docx,
)


def test_pdf_path_for_docx_replaces_extension() -> None:
    assert pdf_path_for_docx("/tmp/CV_2026-01-01_Acme.docx") == "/tmp/CV_2026-01-01_Acme.pdf"


def test_build_cv_pdf_creates_file(tmp_path) -> None:
    docx_path = str(tmp_path / "CV_test.docx")
    open(docx_path, "wb").close()
    content = {
        "professional_title": "Software Engineer",
        "professional_summary": "Experienced developer.",
        "technical_skills": ["Python", "Flask"],
    }
    profile = {"full_name": "Jane Doe", "email": "jane@example.com"}
    preview_lines = cv_content_to_preview_lines(content, profile["full_name"])

    pdf_path = build_cv_pdf(docx_path, preview_lines, profile, content)

    assert pdf_path.endswith(".pdf")
    assert os.path.isfile(pdf_path)
    assert os.path.getsize(pdf_path) > 100


def test_build_cover_letter_pdf_creates_file(tmp_path) -> None:
    docx_path = str(tmp_path / "CoverLetter_test.docx")
    open(docx_path, "wb").close()
    content = {
        "date": "2026-01-01",
        "recipient_company": "Acme Corp",
        "greeting": "Dear Hiring Manager,",
        "body_paragraphs": ["I am excited to apply for this role."],
        "closing": "Sincerely,",
        "signature_name": "Jane Doe",
    }

    pdf_path = build_cover_letter_pdf(docx_path, content)

    assert pdf_path.endswith(".pdf")
    assert os.path.isfile(pdf_path)
    assert os.path.getsize(pdf_path) > 100


def test_cv_pdf_builder_skips_preview_only_sections(tmp_path) -> None:
    output_path = str(tmp_path / "cv.pdf")
    preview_lines = [
        {"text": "Jane Doe", "kind": "name"},
        {"text": "Skills Matching Job", "kind": "section"},
        {"text": "• Python", "kind": "skills"},
        {"text": "Technical Skills", "kind": "section"},
        {"text": "• Flask", "kind": "skills"},
    ]
    CVPdfBuilder().build_from_preview_lines(output_path, preview_lines, {"full_name": "Jane Doe"}, {})
    assert os.path.isfile(output_path)


def test_cover_letter_pdf_builder_writes_minimal_letter(tmp_path) -> None:
    output_path = str(tmp_path / "letter.pdf")
    CoverLetterPdfBuilder().build(
        output_path,
        {
            "greeting": "Hello,",
            "body_paragraphs": ["Body text."],
            "closing": "Thanks,",
            "signature_name": "Jane",
        },
    )
    assert os.path.isfile(output_path)
