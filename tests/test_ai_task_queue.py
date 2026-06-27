"""Tests for the AI task queue."""

import os
import tempfile
import unittest
from unittest.mock import patch

from job_apply_ai.ai_task_runner import AiQueueTaskStopped, ai_queue_control_checkpoint
from job_apply_ai.storage.ai_task_queue_repository import (
    AiTaskQueueRepository,
    to_ai_task_snapshot,
)
from job_apply_ai.storage.database import init_db


class AiTaskQueueRepositoryTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "test_jobs.db")
        self._env_patch = patch.dict(os.environ, {"JOB_APPLY_AI_DB": self.db_path})
        self._env_patch.start()
        init_db(self.db_path)
        self.repo = AiTaskQueueRepository()

    def tearDown(self):
        self._env_patch.stop()
        self.repo = None
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def test_create_and_claim_job(self):
        job = self.repo.create_job(
            task_type="single_cv",
            payload={"job_id": 42},
            job_id=42,
        )
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["task_type"], "single_cv")
        self.assertTrue(job["task_id"])

        claimed = self.repo.claim_next_pending(max_concurrent=3)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["id"], job["id"])
        self.assertEqual(claimed["status"], "running")

    def test_concurrent_claim_respects_limit(self):
        for index in range(3):
            self.repo.create_job(
                task_type="batch_cv",
                payload={"job_ids": [index + 1]},
            )

        claimed = []
        for _ in range(3):
            job = self.repo.claim_next_pending(max_concurrent=2)
            if job:
                claimed.append(job)

        self.assertEqual(len(claimed), 2)
        self.assertIsNone(self.repo.claim_next_pending(max_concurrent=2))

    def test_pause_resume_and_stop(self):
        job = self.repo.create_job(task_type="ats_friendly", payload={"job_id": 1}, job_id=1)
        self.repo.claim_next_pending()
        self.assertTrue(self.repo.pause_job(job["id"]))
        self.assertTrue(self.repo.resume_job(job["id"]))
        self.assertTrue(self.repo.request_stop(job["id"]))

        status, control = self.repo.get_control_state(job["id"])
        self.assertEqual(status, "running")
        self.assertEqual(control, "stop")

    def test_to_ai_task_snapshot_maps_completed(self):
        job = self.repo.create_job(
            task_type="job_match_analyze",
            payload={"meta": {"total_jobs": 5}},
        )
        self.repo.claim_next_pending()
        self.repo.complete_job(job["id"], result={"stats": {"analyzed": 5}}, message="Done")
        snapshot = to_ai_task_snapshot(self.repo.get_job(job["id"]))
        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["task_type"], "job_match_analyze")
        self.assertEqual(snapshot["meta"]["total_jobs"], 5)
        self.assertEqual(snapshot["result"]["stats"]["analyzed"], 5)


class AiQueueCheckpointTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "test_jobs.db")
        self._env_patch = patch.dict(os.environ, {"JOB_APPLY_AI_DB": self.db_path})
        self._env_patch.start()
        init_db(self.db_path)
        self.repo = AiTaskQueueRepository()

    def tearDown(self):
        self._env_patch.stop()
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def test_stop_raises_in_checkpoint(self):
        job = self.repo.create_job(task_type="batch_cv", payload={"job_ids": [1]})
        self.repo.claim_next_pending()
        self.assertTrue(self.repo.request_stop(job["id"]))

        with self.assertRaises(AiQueueTaskStopped):
            ai_queue_control_checkpoint(job["id"], self.repo)


if __name__ == "__main__":
    unittest.main()
