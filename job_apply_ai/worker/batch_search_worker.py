"""Standalone worker that polls and executes batch search queue jobs."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv

from job_apply_ai.batch_search_runner import QueueTaskStopped, run_batch_search_queue_job
from job_apply_ai.storage.batch_queue_repository import BatchQueueRepository
from job_apply_ai.storage.database import init_db

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 5.0
DEFAULT_INTER_JOB_INTERVAL_SECONDS = 3600.0


def _poll_interval() -> float:
    raw = os.environ.get("BATCH_WORKER_POLL_SECONDS", "").strip()
    if not raw:
        return DEFAULT_POLL_SECONDS
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_POLL_SECONDS


def _inter_job_interval() -> float:
    raw = os.environ.get("BATCH_WORKER_INTER_JOB_INTERVAL_SECONDS", "").strip()
    if not raw:
        return DEFAULT_INTER_JOB_INTERVAL_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_INTER_JOB_INTERVAL_SECONDS


def _seconds_until_next_claim(
    last_job_finished_at: float | None,
    interval_seconds: float,
    *,
    now: float | None = None,
) -> float:
    """Return seconds to wait before claiming the next queue job."""
    if interval_seconds <= 0 or last_job_finished_at is None:
        return 0.0
    elapsed = (now if now is not None else time.monotonic()) - last_job_finished_at
    return max(0.0, interval_seconds - elapsed)


def _format_next_claim_time(wait_seconds: float) -> str:
    when = datetime.utcnow() + timedelta(seconds=wait_seconds)
    return when.strftime("%Y-%m-%d %H:%M:%S UTC")


def run_worker(*, once: bool = False) -> None:
    """Poll the batch search queue and process jobs until interrupted."""
    load_dotenv()
    init_db()
    repo = BatchQueueRepository()
    poll_seconds = _poll_interval()
    inter_job_seconds = _inter_job_interval()
    last_job_finished_at: float | None = None
    cooldown_logged = False

    if inter_job_seconds > 0:
        logger.info(
            "Batch search worker started (poll every %.1fs, inter-job cooldown %.0fs)",
            poll_seconds,
            inter_job_seconds,
        )
    else:
        logger.info("Batch search worker started (poll every %.1fs)", poll_seconds)

    while True:
        cooldown_wait = _seconds_until_next_claim(last_job_finished_at, inter_job_seconds)
        if cooldown_wait > 0:
            if not cooldown_logged:
                logger.info(
                    "Inter-job cooldown active; next job claim eligible in %.0fs (~%s)",
                    cooldown_wait,
                    _format_next_claim_time(cooldown_wait),
                )
                cooldown_logged = True
            time.sleep(min(poll_seconds, cooldown_wait))
            continue
        cooldown_logged = False

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

        last_job_finished_at = time.monotonic()
        cooldown_logged = False

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
