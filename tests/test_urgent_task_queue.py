"""Tests for the urgent UI I/O task queue."""

import os
import tempfile
import unittest
from unittest.mock import patch

from job_apply_ai.storage.database import init_db
from job_apply_ai.storage.urgent_task_queue_repository import (
    UrgentTaskQueueRepository,
    to_urgent_task_snapshot,
)
from job_apply_ai.urgent_task_runner import UrgentQueueTaskStopped, urgent_queue_control_checkpoint


class UrgentTaskQueueRepositoryTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "test_jobs.db")
        self._env_patch = patch.dict(os.environ, {"JOB_APPLY_AI_DB": self.db_path})
        self._env_patch.start()
        init_db(self.db_path)
        self.repo = UrgentTaskQueueRepository()

    def tearDown(self):
        self._env_patch.stop()
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def test_create_and_claim_single_search(self):
        job = self.repo.create_job(
            task_type="single_search",
            payload={
                "keyword": "Engineer",
                "location": "Berlin",
                "meta": {"keyword": "Engineer", "location": "Berlin"},
            },
        )
        self.assertEqual(job["status"], "pending")
        claimed = self.repo.claim_next_pending(max_concurrent=2)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["id"], job["id"])
        self.assertEqual(claimed["status"], "running")

    def test_linkedin_import_snapshot(self):
        job = self.repo.create_job(
            task_type="linkedin_job_import",
            payload={"linkedin_url": "https://linkedin.com/jobs/view/1", "meta": {}},
        )
        self.repo.claim_next_pending()
        self.repo.complete_job(
            job["id"],
            result={"job": {"title": "Dev"}},
            message="Done",
        )
        snapshot = to_urgent_task_snapshot(self.repo.get_job(job["id"]))
        self.assertEqual(snapshot["task_type"], "linkedin_job_import")
        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["result"]["job"]["title"], "Dev")

    def test_stop_raises_in_checkpoint(self):
        job = self.repo.create_job(
            task_type="single_search",
            payload={"keyword": "QA", "location": "Remote"},
        )
        self.repo.claim_next_pending()
        self.assertTrue(self.repo.request_stop(job["id"]))

        with self.assertRaises(UrgentQueueTaskStopped):
            urgent_queue_control_checkpoint(job["id"], self.repo)


if __name__ == "__main__":
    unittest.main()
