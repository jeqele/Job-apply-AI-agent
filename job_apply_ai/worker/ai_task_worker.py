"""Standalone worker that polls and executes AI task queue jobs concurrently."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor

from dotenv import load_dotenv

from job_apply_ai.storage.app_settings import (
    AppSettingsRepository,
    WORKER_SETTING_LIMITS,
    normalize_worker_settings,
)
from job_apply_ai.ai_task_runner import AiQueueTaskStopped, run_ai_task_queue_job
from job_apply_ai.storage.ai_task_queue_repository import AiTaskQueueRepository
from job_apply_ai.storage.database import init_db

logger = logging.getLogger(__name__)

_MAX_EXECUTOR_THREADS = WORKER_SETTING_LIMITS["ai_worker_concurrency"][1]


def _load_worker_config() -> tuple[float, int]:
    """Read AI worker settings from UI-backed app settings."""
    try:
        settings = AppSettingsRepository().get_worker_settings()
    except Exception:
        settings = normalize_worker_settings(None)
    return settings["ai_worker_poll_seconds"], settings["ai_worker_concurrency"]


def _handle_finished_future(future: Future[None], job_id: int, repo: AiTaskQueueRepository) -> None:
    try:
        future.result()
    except AiQueueTaskStopped as exc:
        logger.info("AI task job %s stopped: %s", job_id, exc)
        current = repo.get_job(job_id)
        if current and current["status"] == "running":
            repo.mark_cancelled(
                job_id,
                message=str(exc),
                result=current.get("result") or {},
            )
    except Exception as exc:
        logger.exception("AI task job %s failed", job_id)
        repo.fail_job(job_id, str(exc))


def run_worker(*, once: bool = False) -> None:
    """Poll the AI task queue and process jobs with bounded concurrency."""
    load_dotenv()
    init_db()
    repo = AiTaskQueueRepository()
    poll_seconds, max_concurrent = _load_worker_config()
    executor = ThreadPoolExecutor(
        max_workers=_MAX_EXECUTOR_THREADS,
        thread_name_prefix="ai-worker",
    )
    active: dict[int, Future[None]] = {}

    logger.info(
        "AI task worker started (poll every %.1fs, concurrency %d from Settings)",
        poll_seconds,
        max_concurrent,
    )

    try:
        while True:
            poll_seconds, max_concurrent = _load_worker_config()

            finished_ids = [job_id for job_id, future in active.items() if future.done()]
            for job_id in finished_ids:
                _handle_finished_future(active.pop(job_id), job_id, repo)

            slots = max_concurrent - len(active)
            claimed_any = False
            for _ in range(slots):
                job = repo.claim_next_pending(max_concurrent=max_concurrent)
                if not job:
                    break
                claimed_any = True
                job_id = job["id"]
                logger.info(
                    "Running AI task job %s (%s, type=%s)",
                    job_id,
                    job.get("task_id"),
                    job.get("task_type"),
                )
                active[job_id] = executor.submit(run_ai_task_queue_job, job_id, queue_repo=repo)

            if once and not active:
                if not claimed_any:
                    logger.info("No pending AI tasks; exiting (--once).")
                return

            time.sleep(poll_seconds)
    finally:
        for job_id, future in list(active.items()):
            _handle_finished_future(future, job_id, repo)
        executor.shutdown(wait=True)


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="AI task queue worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process until the queue is idle then exit",
    )
    args = parser.parse_args()
    run_worker(once=args.once)


if __name__ == "__main__":
    main()
