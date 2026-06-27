"""Tests for the batch search job queue."""

import os
import tempfile
import unittest
from unittest.mock import patch

from job_apply_ai.batch_search import split_batch_inputs
from job_apply_ai.batch_search_runner import QueueTaskStopped, queue_control_checkpoint
from job_apply_ai.storage.batch_queue_repository import (
    BatchQueueRepository,
    to_task_snapshot,
)
from job_apply_ai.storage.database import init_db
from job_apply_ai.scraper.search_filters import SearchFilters


class SplitBatchInputsTests(unittest.TestCase):
    def test_no_split_when_under_limit(self):
        titles = ["Engineer", "Analyst"]
        locations = ["Berlin", "Remote"]
        parts = split_batch_inputs(titles, locations, max_combinations=50)
        self.assertEqual(parts, [(titles, locations)])

    def test_splits_by_titles_when_locations_fit(self):
        titles = [f"Title {index}" for index in range(26)]
        locations = ["Berlin", "Remote"]
        parts = split_batch_inputs(titles, locations, max_combinations=50)
        self.assertEqual(len(parts), 2)
        self.assertEqual(sum(len(t) * len(l) for t, l in parts), 52)
        for part_titles, part_locations in parts:
            self.assertLessEqual(len(part_titles) * len(part_locations), 50)

    def test_splits_by_locations_when_titles_fit(self):
        titles = ["Engineer"]
        locations = [f"City {index}" for index in range(120)]
        parts = split_batch_inputs(titles, locations, max_combinations=50)
        self.assertEqual(len(parts), 3)
        self.assertEqual(sum(len(t) * len(l) for t, l in parts), 120)


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

    def test_create_jobs_splits_large_batch(self):
        titles = [f"Title {index}" for index in range(20)]
        locations = [f"City {index}" for index in range(5)]
        with patch(
            "job_apply_ai.batch_search.get_batch_queue_max_combinations_per_job",
            return_value=50,
        ):
            jobs = self.repo.create_jobs(
                name="Weekly sweep",
                titles=titles,
                locations=locations,
                schedule_type="weekly",
                sources="adzuna,reed",
                mode="remote",
                search_filters=SearchFilters(remote=True),
            )
        self.assertEqual(len(jobs), 2)
        self.assertEqual(
            sum(job["total_combinations"] for job in jobs),
            100,
        )
        for index, job in enumerate(jobs, start=1):
            self.assertIn(f"(part {index}/2)", job["name"])
            self.assertEqual(job["schedule_type"], "weekly")
            self.assertEqual(job["sources"], "adzuna,reed")
            self.assertEqual(job["mode"], "remote")
            self.assertTrue(job["search_filters"]["remote"])
            self.assertLessEqual(job["total_combinations"], 50)

    def test_create_job_returns_first_when_split(self):
        titles = [f"Title {index}" for index in range(20)]
        locations = [f"City {index}" for index in range(5)]
        with patch(
            "job_apply_ai.batch_search.get_batch_queue_max_combinations_per_job",
            return_value=50,
        ):
            job = self.repo.create_job(
                name="Split batch",
                titles=titles,
                locations=locations,
            )
            all_jobs = self.repo.list_jobs()
        self.assertEqual(len(all_jobs), 2)
        self.assertEqual(job["id"], all_jobs[1]["id"])
        self.assertIn("(part 1/2)", job["name"])

    def test_create_jobs_single_item_keeps_name(self):
        job = self.repo.create_jobs(
            name="Single batch",
            titles=["Engineer"],
            locations=["Berlin"],
        )[0]
        self.assertEqual(job["name"], "Single batch")
        self.assertNotIn("part", job["name"])

    def test_clear_finished_jobs_removes_terminal_only(self):
        running = self.repo.create_job(
            name="In progress",
            titles=["Engineer"],
            locations=["Berlin"],
        )
        failed = self.repo.create_job(
            name="Broken",
            titles=["Analyst"],
            locations=["Remote"],
        )
        pending = self.repo.create_job(
            name="Waiting",
            titles=["QA"],
            locations=["London"],
        )
        claimed = self.repo.claim_next_pending()
        self.assertEqual(claimed["id"], running["id"])
        self.repo.complete_job(
            running["id"],
            search_run_id=1,
            result={"search_run_id": 1, "total_jobs": 1},
            message="Done",
            reschedule=False,
        )
        self.repo.fail_job(failed["id"], "Worker error")

        deleted = self.repo.clear_finished_jobs()
        self.assertEqual(deleted, 2)
        remaining = self.repo.list_jobs()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], pending["id"])
        self.assertEqual(remaining[0]["status"], "pending")


class QueueCheckpointTests(unittest.TestCase):
    def test_stop_raises(self):
        with patch("job_apply_ai.batch_search_runner.BatchQueueRepository") as repo_cls:
            repo = repo_cls.return_value
            repo.get_control_state.return_value = ("running", "stop")
            with self.assertRaises(QueueTaskStopped):
                queue_control_checkpoint(1, repo=repo)


if __name__ == "__main__":
    unittest.main()
