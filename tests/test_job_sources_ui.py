"""Tests for shared job source UI defaults."""

import os
import tempfile
import unittest
from unittest.mock import patch

from job_apply_ai.job_sources import (
    UI_DEFAULT_JOB_SOURCES,
    UI_JOB_SOURCE_LABELS,
    UI_JOB_SOURCE_OPTIONS,
    format_sources_csv,
    job_source_options_for_ui,
    parse_sources_csv,
    selected_source_ids_from_csv,
)
from job_apply_ai.scraper.aggregator import AVAILABLE_SOURCES
from job_apply_ai.storage.batch_queue_repository import BatchQueueRepository
from job_apply_ai.storage.database import init_db


class JobSourceDefaultsTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "sources.db")
        self._env_patch = patch.dict(os.environ, {"JOB_APPLY_AI_DB": self.db_path})
        self._env_patch.start()
        init_db(self.db_path)
        self.repo = BatchQueueRepository()

    def tearDown(self):
        self._env_patch.stop()
        self.repo = None
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def test_ui_options_match_available_sources(self):
        self.assertEqual(set(UI_JOB_SOURCE_OPTIONS), set(AVAILABLE_SOURCES.keys()))

    def test_default_includes_all_sources(self):
        self.assertEqual(parse_sources_csv(UI_DEFAULT_JOB_SOURCES), list(UI_JOB_SOURCE_OPTIONS))

    def test_arbeitnow_in_default(self):
        self.assertIn("arbeitnow", UI_DEFAULT_JOB_SOURCES.split(","))

    def test_repository_default_includes_all_sources(self):
        job = self.repo.create_job(
            name="Defaults",
            titles=["Engineer"],
            locations=["Berlin"],
        )
        self.assertEqual(job["sources"], UI_DEFAULT_JOB_SOURCES)
        self.assertIn("arbeitnow", job["sources"])

    def test_selected_source_ids_defaults_to_all(self):
        self.assertEqual(selected_source_ids_from_csv(None), set(UI_JOB_SOURCE_OPTIONS))

    def test_selected_source_ids_from_saved_csv(self):
        self.assertEqual(
            selected_source_ids_from_csv("adzuna,reed"),
            {"adzuna", "reed"},
        )

    def test_job_source_options_for_ui(self):
        options = job_source_options_for_ui()
        self.assertEqual(len(options), len(UI_JOB_SOURCE_OPTIONS))
        self.assertEqual(options[0]["label"], UI_JOB_SOURCE_LABELS[options[0]["id"]])


class BatchQueueFormSourcesTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "ui.db")
        self._env_patch = patch.dict(os.environ, {"JOB_APPLY_AI_DB": self.db_path})
        self._env_patch.start()
        init_db(self.db_path)
        self.repo = BatchQueueRepository()

    def tearDown(self):
        self._env_patch.stop()
        self.repo = None
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def test_new_batch_queue_form_renders_source_checkboxes(self):
        from job_apply_ai.ui.app import app

        with patch("job_apply_ai.ui.app.batch_queue_repo", self.repo):
            response = app.test_client().get("/batch-queue/new")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for source_id in UI_JOB_SOURCE_OPTIONS:
            self.assertIn(f'value="{source_id}"', html)
            self.assertIn(f'id="queue_source_{source_id}"', html)
            self.assertIn("checked", html)
        self.assertIn("Arbeitnow", html)
        self.assertIn('name="job_sources_field"', html)

    def test_batch_queue_create_persists_default_sources_without_posting_checkboxes(self):
        from job_apply_ai.ui.app import app

        with patch("job_apply_ai.ui.app.batch_queue_repo", self.repo):
            response = app.test_client().post(
                "/batch-queue/new",
                data={
                    "name": "Arbeitnow sweep",
                    "titles_text": "Software Engineer",
                    "locations_text": "Berlin",
                    "schedule_type": "once",
                    "max_jobs": "5",
                    "mode": "both",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        jobs = self.repo.list_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["sources"], UI_DEFAULT_JOB_SOURCES)
        self.assertIn("arbeitnow", jobs[0]["sources"])

    def test_batch_queue_create_persists_selected_sources(self):
        from job_apply_ai.ui.app import app

        selected = ["adzuna", "arbeitnow"]
        with patch("job_apply_ai.ui.app.batch_queue_repo", self.repo):
            response = app.test_client().post(
                "/batch-queue/new",
                data={
                    "name": "Subset",
                    "titles_text": "Engineer",
                    "locations_text": "Berlin",
                    "schedule_type": "once",
                    "max_jobs": "5",
                    "mode": "both",
                    "job_sources_field": "1",
                    "sources": selected,
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        jobs = self.repo.list_jobs()
        self.assertEqual(jobs[0]["sources"], format_sources_csv(selected))

    def test_batch_queue_create_rejects_empty_source_selection(self):
        from job_apply_ai.ui.app import app

        with patch("job_apply_ai.ui.app.batch_queue_repo", self.repo):
            response = app.test_client().post(
                "/batch-queue/new",
                data={
                    "name": "No sources",
                    "titles_text": "Engineer",
                    "locations_text": "Berlin",
                    "schedule_type": "once",
                    "max_jobs": "5",
                    "mode": "both",
                    "job_sources_field": "1",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Select at least one job source", response.data)
        self.assertEqual(self.repo.list_jobs(), [])
