"""Tests for full backup and restore."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from job_apply_ai.cv_modifier.cv_content_store import cv_content_path, save_cv_content
from job_apply_ai.storage.app_settings import AppSettingsRepository
from job_apply_ai.storage.backup import (
    BACKUP_FORMAT,
    BACKUP_VERSION,
    FILES_CVS_PREFIX,
    FILES_JOBS_PREFIX,
    MANIFEST_NAME,
    export_backup,
    restore_backup,
)
from job_apply_ai.storage.database import get_connection, init_db
from job_apply_ai.storage.dev_log import DevLogRepository
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.storage.user_profile import UserProfileRepository, skill_names


class BackupRestoreTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self._tmpdir.name, "job_apply_ai")
        self.db_path = os.path.join(self._tmpdir.name, "test_jobs.db")
        self.cv_dir = os.path.join(self.data_dir, "cvs")
        self.jobs_dir = os.path.join(self.data_dir, "jobs")
        os.makedirs(self.cv_dir, exist_ok=True)
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._env_patch = patch.dict(os.environ, {"JOB_APPLY_AI_DB": self.db_path})
        self._env_patch.start()
        init_db(self.db_path)
        self.profile_repo = UserProfileRepository()
        self.job_repo = JobRepository()
        self.settings_repo = AppSettingsRepository()
        self.dev_log_repo = DevLogRepository()

    def tearDown(self):
        self._env_patch.stop()
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def _seed_data(self) -> None:
        self.profile_repo.save_profile(
            {
                "full_name": "Jane Doe",
                "email": "jane@example.com",
                "technical_skills": ["Python"],
            }
        )
        self.settings_repo.save_llm_settings(
            {
                "llm_provider": "alibaba",
                "fast_model_provider": "alibaba",
                "main_model_provider": "alibaba",
                "dev_mode": True,
                "alibaba": {
                    "api_key": "secret-backup-key",
                    "base_url": "https://example.test/v1",
                    "fast_model": "qwen-turbo",
                    "main_model": "qwen-plus",
                },
            }
        )
        self.dev_log_repo.add_log(
            category="system",
            event="test_event",
            message="seeded log",
        )
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO batch_search_jobs (
                    name, status, schedule_type, titles_json, locations_json,
                    shuffle_queue, max_jobs, sources, mode, search_filters_json,
                    total_combinations, task_id
                ) VALUES (?, 'pending', 'once', ?, ?, 0, 5, 'linkedin', 'both', '{}', 2, ?)
                """,
                ("Nightly search", json.dumps(["Engineer"]), json.dumps(["Berlin"]), "task-123"),
            )

        search_run_id = self.job_repo.create_search_run("Engineer", "Berlin", "linkedin", "both")
        cv_filename = "CV_2026-01-01_Acme_Engineer.docx"
        save_cv_content(
            self.cv_dir,
            cv_filename,
            {"summary": "Tailored summary"},
            chat_history=[{"role": "user", "content": "Make it shorter"}],
        )
        with open(os.path.join(self.cv_dir, cv_filename), "wb") as handle:
            handle.write(b"fake-docx")

        orphan_cv = "CV_orphan_role.docx"
        with open(os.path.join(self.cv_dir, orphan_cv), "wb") as handle:
            handle.write(b"orphan-docx")
        save_cv_content(self.cv_dir, orphan_cv, {"summary": "Orphan CV"})

        with open(os.path.join(self.jobs_dir, "jobs_export.xlsx"), "wb") as handle:
            handle.write(b"fake-xlsx")

        job_ids = self.job_repo.upsert_jobs(
            [
                {
                    "title": "Backend Engineer",
                    "company": "Acme",
                    "location": "Berlin",
                    "description": "Python APIs",
                    "link": "https://example.com/jobs/1",
                    "workflow_status": "shortlisted",
                }
            ],
            search_run_id=search_run_id,
        )
        self.job_repo.update_job(job_ids[0], {"cv_filename": cv_filename})

    def test_export_contains_full_manifest_and_files(self):
        self._seed_data()
        buffer = export_backup(data_dir=self.data_dir, db_path=self.db_path)

        with zipfile.ZipFile(buffer) as archive:
            self.assertIn(MANIFEST_NAME, archive.namelist())
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            self.assertEqual(manifest["format"], BACKUP_FORMAT)
            self.assertEqual(manifest["version"], BACKUP_VERSION)
            self.assertEqual(manifest["profile"]["full_name"], "Jane Doe")
            self.assertEqual(len(manifest["jobs"]), 1)
            self.assertEqual(len(manifest["search_runs"]), 1)
            self.assertIn("CV_2026-01-01_Acme_Engineer.docx", manifest["cv_content"])
            self.assertIn("CV_orphan_role.docx", manifest["cv_content"])
            self.assertEqual(manifest["app_settings"]["dev_mode"], True)
            self.assertEqual(manifest["app_settings"]["alibaba"]["api_key"], "")
            self.assertEqual(len(manifest["dev_logs"]), 1)
            self.assertEqual(len(manifest["batch_search_jobs"]), 1)
            names = archive.namelist()
            self.assertTrue(any(name.startswith(FILES_CVS_PREFIX) for name in names))
            self.assertTrue(any(name.startswith(FILES_JOBS_PREFIX) for name in names))

    def test_restore_merge_adds_jobs_and_profile_fields(self):
        self._seed_data()
        buffer = export_backup(data_dir=self.data_dir, db_path=self.db_path)

        self.profile_repo.save_profile({"full_name": "Existing User", "email": "existing@example.com"})
        self.job_repo.upsert_jobs(
            [
                {
                    "title": "Other Role",
                    "company": "Beta",
                    "location": "Remote",
                    "link": "https://example.com/jobs/2",
                }
            ]
        )

        stats = restore_backup(
            buffer.getvalue(),
            data_dir=self.data_dir,
            merge_profile=True,
            db_path=self.db_path,
        )
        self.assertEqual(stats["jobs_restored"], 1)
        self.assertGreaterEqual(stats["dev_logs_restored"], 1)
        self.assertGreaterEqual(stats["batch_jobs_restored"], 1)
        self.assertGreaterEqual(stats["files_restored"], 1)

        profile = self.profile_repo.get_profile()
        self.assertEqual(profile["full_name"], "Existing User")
        self.assertEqual(profile["email"], "existing@example.com")
        self.assertEqual(skill_names(profile["technical_skills"]), ["Python"])

        settings = self.settings_repo.get_settings()
        self.assertTrue(settings["dev_mode"])
        self.assertEqual(settings["alibaba"]["api_key"], "secret-backup-key")

        jobs = self.job_repo.list_jobs()
        self.assertEqual(len(jobs), 2)
        titles = {job["title"] for job in jobs}
        self.assertIn("Backend Engineer", titles)
        self.assertIn("Other Role", titles)

        sidecar = cv_content_path(self.cv_dir, "CV_2026-01-01_Acme_Engineer.docx")
        self.assertTrue(os.path.isfile(sidecar))
        with open(sidecar, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["tailored_content"]["summary"], "Tailored summary")
        self.assertTrue(os.path.isfile(os.path.join(self.cv_dir, "CV_orphan_role.docx")))
        self.assertTrue(os.path.isfile(os.path.join(self.jobs_dir, "jobs_export.xlsx")))

    def test_restore_replace_clears_existing_jobs(self):
        self._seed_data()
        buffer = export_backup(data_dir=self.data_dir, db_path=self.db_path)

        self.job_repo.upsert_jobs(
            [
                {
                    "title": "Temporary Role",
                    "company": "TempCo",
                    "location": "London",
                    "link": "https://example.com/jobs/temp",
                }
            ]
        )
        self.assertEqual(self.job_repo.count_jobs(), 2)

        stats = restore_backup(
            buffer.getvalue(),
            data_dir=self.data_dir,
            replace=True,
            db_path=self.db_path,
        )
        self.assertEqual(stats["jobs_restored"], 1)
        self.assertEqual(self.job_repo.count_jobs(), 1)
        self.assertEqual(self.job_repo.list_jobs()[0]["title"], "Backend Engineer")

    def test_restore_rejects_invalid_backup(self):
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            restore_backup(b"not-a-zip", data_dir=self.data_dir, db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
