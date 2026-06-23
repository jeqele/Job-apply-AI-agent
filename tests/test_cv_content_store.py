"""Tests for CV content sidecar storage helpers."""

from __future__ import annotations

from pathlib import Path

from job_apply_ai.cv_modifier.cv_content_store import (
    cv_content_path,
    delete_cv_artifacts,
)


def test_delete_cv_artifacts_removes_cv_cover_letter_and_sidecar(tmp_path: Path) -> None:
    output_dir = tmp_path / "cvs"
    output_dir.mkdir()

    cv_filename = "CV_2026-01-01_Acme_Engineer.docx"
    cover_letter_filename = "CoverLetter_2026-01-01_Acme_Engineer.docx"
    cv_path = output_dir / cv_filename
    cl_path = output_dir / cover_letter_filename
    sidecar_path = Path(cv_content_path(str(output_dir), cv_filename))

    cv_path.write_text("cv", encoding="utf-8")
    cl_path.write_text("letter", encoding="utf-8")
    sidecar_path.write_text("{}", encoding="utf-8")

    delete_cv_artifacts(
        str(output_dir),
        cv_filename,
        cover_letter_filename=cover_letter_filename,
    )

    assert not cv_path.exists()
    assert not cl_path.exists()
    assert not sidecar_path.exists()


def test_delete_cv_artifacts_is_noop_when_files_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "cvs"
    output_dir.mkdir()

    delete_cv_artifacts(
        str(output_dir),
        "missing.docx",
        cover_letter_filename="also-missing.docx",
    )
