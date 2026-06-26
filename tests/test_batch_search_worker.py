"""Tests for the batch search queue worker."""

import os
import unittest
from unittest.mock import MagicMock, patch

from job_apply_ai.worker import batch_search_worker as worker


class InterJobIntervalTests(unittest.TestCase):
    def test_default_interval_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(worker._inter_job_interval(), 3600.0)

    def test_parses_valid_interval(self):
        with patch.dict(os.environ, {"BATCH_WORKER_INTER_JOB_INTERVAL_SECONDS": "120"}):
            self.assertEqual(worker._inter_job_interval(), 120.0)

    def test_zero_disables_cooldown(self):
        with patch.dict(os.environ, {"BATCH_WORKER_INTER_JOB_INTERVAL_SECONDS": "0"}):
            self.assertEqual(worker._inter_job_interval(), 0.0)

    def test_invalid_value_falls_back_to_default(self):
        with patch.dict(os.environ, {"BATCH_WORKER_INTER_JOB_INTERVAL_SECONDS": "bad"}):
            self.assertEqual(worker._inter_job_interval(), 3600.0)


class SecondsUntilNextClaimTests(unittest.TestCase):
    def test_no_wait_on_first_job(self):
        self.assertEqual(worker._seconds_until_next_claim(None, 3600.0), 0.0)

    def test_no_wait_when_interval_disabled(self):
        self.assertEqual(worker._seconds_until_next_claim(100.0, 0.0, now=500.0), 0.0)

    def test_full_wait_immediately_after_finish(self):
        self.assertEqual(
            worker._seconds_until_next_claim(1000.0, 3600.0, now=1000.0),
            3600.0,
        )

    def test_partial_wait_after_elapsed_time(self):
        self.assertEqual(
            worker._seconds_until_next_claim(1000.0, 3600.0, now=2800.0),
            1800.0,
        )

    def test_no_wait_after_interval_elapsed(self):
        self.assertEqual(
            worker._seconds_until_next_claim(1000.0, 3600.0, now=4600.0),
            0.0,
        )


class RunWorkerCooldownTests(unittest.TestCase):
    @patch("job_apply_ai.worker.batch_search_worker.init_db")
    @patch("job_apply_ai.worker.batch_search_worker.run_batch_search_queue_job")
    @patch("job_apply_ai.worker.batch_search_worker.load_dotenv")
    def test_worker_waits_before_second_claim(
        self,
        _load_dotenv,
        mock_run_job,
        _init_db,
    ):
        repo = MagicMock()
        job_a = {"id": 1, "name": "Job A", "task_id": "a", "total_combinations": 1}
        job_b = {"id": 2, "name": "Job B", "task_id": "b", "total_combinations": 1}
        claim_times: list[float] = []
        clock = {"t": 0.0}

        def monotonic() -> float:
            return clock["t"]

        def claim_next_pending():
            claim_times.append(clock["t"])
            if len(claim_times) == 1:
                return job_a
            if len(claim_times) == 2:
                return job_b
            return None

        run_count = {"n": 0}

        def run_job(*args, **kwargs):
            run_count["n"] += 1
            if run_count["n"] >= 2:
                raise KeyboardInterrupt

        mock_run_job.side_effect = run_job

        def sleep(seconds: float) -> None:
            clock["t"] += seconds

        repo.claim_next_pending.side_effect = claim_next_pending

        with patch.dict(
            os.environ,
            {
                "BATCH_WORKER_INTER_JOB_INTERVAL_SECONDS": "3600",
                "BATCH_WORKER_POLL_SECONDS": "5",
            },
        ), patch(
            "job_apply_ai.worker.batch_search_worker.BatchQueueRepository",
            return_value=repo,
        ), patch(
            "job_apply_ai.worker.batch_search_worker.time.monotonic",
            side_effect=monotonic,
        ), patch(
            "job_apply_ai.worker.batch_search_worker.time.sleep",
            side_effect=sleep,
        ):
            with self.assertRaises(KeyboardInterrupt):
                worker.run_worker(once=False)

        self.assertEqual(mock_run_job.call_count, 2)
        self.assertEqual(len(claim_times), 2)
        self.assertGreaterEqual(claim_times[1] - claim_times[0], 3600.0)

    @patch("job_apply_ai.worker.batch_search_worker.init_db")
    @patch("job_apply_ai.worker.batch_search_worker.run_batch_search_queue_job")
    @patch("job_apply_ai.worker.batch_search_worker.time.sleep")
    @patch("job_apply_ai.worker.batch_search_worker.time.monotonic")
    @patch("job_apply_ai.worker.batch_search_worker.load_dotenv")
    def test_worker_claims_immediately_when_cooldown_disabled(
        self,
        _load_dotenv,
        mock_monotonic,
        mock_sleep,
        mock_run_job,
        _init_db,
    ):
        repo = MagicMock()
        job = {"id": 1, "name": "Job A", "task_id": "a", "total_combinations": 1}
        repo.claim_next_pending.side_effect = [job, None]
        mock_monotonic.side_effect = [0.0, 0.0, 10.0, 10.0]

        with patch.dict(
            os.environ,
            {
                "BATCH_WORKER_INTER_JOB_INTERVAL_SECONDS": "0",
                "BATCH_WORKER_POLL_SECONDS": "5",
            },
        ), patch(
            "job_apply_ai.worker.batch_search_worker.BatchQueueRepository",
            return_value=repo,
        ):
            worker.run_worker(once=True)

        self.assertEqual(mock_run_job.call_count, 1)
        self.assertEqual(repo.claim_next_pending.call_count, 1)
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
