"""Shared filesystem paths for the web app and background workers."""

from __future__ import annotations

import os
import tempfile

from job_apply_ai.utils.helpers import ensure_directory_exists


def get_data_dir() -> str:
    """Application data root (uploads, CVs, jobs output)."""
    custom = os.environ.get("JOB_APPLY_AI_DATA_DIR", "").strip()
    if custom:
        return custom
    return os.path.join(tempfile.gettempdir(), "job_apply_ai")


def get_cv_output_dir() -> str:
    """Directory where generated CV and cover letter files are stored."""
    custom = os.environ.get("JOB_APPLY_AI_CV_DIR", "").strip()
    if custom:
        ensure_directory_exists(custom)
        return custom
    cv_dir = os.path.join(get_data_dir(), "cvs")
    ensure_directory_exists(cv_dir)
    return cv_dir
