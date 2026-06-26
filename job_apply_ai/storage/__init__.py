"""SQLite persistence and export helpers for job listings."""

from job_apply_ai.storage.database import get_db_path, init_db
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.storage.user_profile import UserProfileRepository

__all__ = [
    "get_db_path",
    "init_db",
    "JobRepository",
    "UserProfileRepository",
    "export_jobs",
    "export_backup",
    "restore_backup",
]


def __getattr__(name: str):
    if name == "export_jobs":
        from job_apply_ai.storage.exports import export_jobs

        return export_jobs
    if name == "export_backup":
        from job_apply_ai.storage.backup import export_backup

        return export_backup
    if name == "restore_backup":
        from job_apply_ai.storage.backup import restore_backup

        return restore_backup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
