"""Standalone worker that polls and executes batch search queue jobs."""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv

from job_apply_ai.batch_search_runner import QueueTaskStopped, run_batch_search_queue_job
from job_apply_ai.storage.batch_queue_repository import BatchQueueRepository
from job_apply_ai.storage.database import init_db

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 5.0


def _poll_interval() -> float:
    raw = os.environ.get("BATCH_WORKER_POLL_SECONDS", "").strip()
    if not raw:
        return DEFAULT_POLL_SECONDS
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_POLL_SECONDS


def run_worker(*, once: bool = False) -> None:
    """Poll the batch search queue and process jobs until interrupted."""
    load_dotenv()
    init_db()
    repo = BatchQueueRepository()
    poll_seconds = _poll_interval()
    logger.info("Batch search worker started (poll every %.1fs)", poll_seconds)

    while True:
        job = repo.claim_next_pending()
        if not job:
            if once:
                logger.info("No pending jobs; exiting (--once).")
                return
            time.sleep(poll_seconds)
            continue

        job_id = job["id"]
        logger.info(
            "Running batch search job %s (%s, %d combinations)",
            job_id,
            job.get("name") or job["task_id"],
            job.get("total_combinations", 0),
        )
        try:
            run_batch_search_queue_job(job_id, queue_repo=repo)
        except QueueTaskStopped as exc:
            logger.info("Batch search job %s stopped: %s", job_id, exc)
            current = repo.get_job(job_id)
            if current and current["status"] == "running":
                repo.mark_cancelled(
                    job_id,
                    message=str(exc),
                    result=current.get("result") or {},
                )
        except Exception as exc:
            logger.exception("Batch search job %s failed", job_id)
            repo.fail_job(job_id, str(exc))

        if once:
            return


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Batch search queue worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one job then exit",
    )
    args = parser.parse_args()
    run_worker(once=args.once)


if __name__ == "__main__":
    main()
