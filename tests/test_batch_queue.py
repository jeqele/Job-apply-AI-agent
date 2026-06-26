"""Tests for the batch search job queue."""

import os
import tempfile
import unittest
from unittest.mock import patch

from job_apply_ai.batch_search_runner import QueueTaskStopped, queue_control_checkpoint
from job_apply_ai.storage.batch_queue_repository import (
    BatchQueueRepository,
    to_task_snapshot,
)
from job_apply_ai.storage.database import init_db


class BatchQueueRepositoryTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "test_jobs.db")
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

    def test_create_and_list_job(self):
        job = self.repo.create_job(
            name="Test batch",
            titles=["Engineer"],
            locations=["Berlin", "Remote"],
            schedule_type="once",
        )
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["total_combinations"], 2)
        self.assertTrue(job["task_id"])

        jobs = self.repo.list_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["name"], "Test batch")

    def test_claim_next_pending(self):
        created = self.repo.create_job(
            name="Claim me",
            titles=["Dev"],
            locations=["London"],
        )
        claimed = self.repo.claim_next_pending()
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["id"], created["id"])
        self.assertEqual(claimed["status"], "running")
        self.assertIsNone(self.repo.claim_next_pending())

    def test_pause_resume_and_stop(self):
        job = self.repo.create_job(
            name="Controls",
            titles=["QA"],
            locations=["Remote"],
        )
        self.repo.claim_next_pending()
        self.assertTrue(self.repo.pause_job(job["id"]))
        self.assertTrue(self.repo.resume_job(job["id"]))
        self.assertTrue(self.repo.request_stop(job["id"]))

        status, control = self.repo.get_control_state(job["id"])
        self.assertEqual(status, "running")
        self.assertEqual(control, "stop")

    def test_recurring_job_keeps_completed_with_next_run(self):
        job = self.repo.create_job(
            name="Daily sweep",
            titles=["Engineer"],
            locations=["Remote"],
            schedule_type="daily",
        )
        self.repo.claim_next_pending()
        self.repo.complete_job(
            job["id"],
            search_run_id=42,
            result={"search_run_id": 42, "total_jobs": 3},
            message="Done",
            reschedule=True,
        )
        completed = self.repo.get_job(job["id"])
        self.assertEqual(completed["status"], "completed")
        self.assertIsNotNone(completed["next_run_at"])

    def test_to_task_snapshot_maps_completed(self):
        snapshot = to_task_snapshot(
            {
                "id": 1,
                "task_id": "abc123",
                "status": "completed",
                "progress_step": "complete",
                "progress_message": "Finished",
                "progress_percent": 100,
                "total_combinations": 4,
                "current_index": 4,
                "result": {"search_run_id": 9, "total_jobs": 2},
                "last_error": "",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:01:00",
            }
        )
        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["result"]["total_jobs"], 2)


class QueueCheckpointTests(unittest.TestCase):
    def test_stop_raises(self):
        with patch("job_apply_ai.batch_search_runner.BatchQueueRepository") as repo_cls:
            repo = repo_cls.return_value
            repo.get_control_state.return_value = ("running", "stop")
            with self.assertRaises(QueueTaskStopped):
                queue_control_checkpoint(1, repo=repo)


if __name__ == "__main__":
    unittest.main()
