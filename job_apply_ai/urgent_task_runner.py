"""Execute urgent UI I/O queue jobs (shared by worker and tests)."""

from __future__ import annotations

import logging
import time
from typing import Any

from job_apply_ai.io_workflows import run_linkedin_import_workflow, run_single_search_workflow
from job_apply_ai.storage.urgent_task_queue_repository import UrgentTaskQueueRepository
from job_apply_ai.storage.user_profile import UserProfileRepository
from job_apply_ai.ui.cv_tasks import TaskStopped

logger = logging.getLogger(__name__)


class UrgentQueueTaskStopped(TaskStopped):
    """Raised when an urgent queue job should exit early."""


class QueueTaskProgress:
    """Bridge urgent queue repository updates to io_workflows progress protocol."""

    def __init__(
        self,
        queue_job_id: int,
        task_id: str,
        repo: UrgentTaskQueueRepository,
    ) -> None:
        self.queue_job_id = queue_job_id
        self.task_id = task_id
        self._repo = repo

    def update(
        self,
        *,
        status: str | None = None,
        step: str | None = None,
        message: str | None = None,
        percent: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        payload_patch = None
        if meta:
            job = self._repo.get_job(self.queue_job_id)
            existing_meta = (job.get("payload") or {}).get("meta") or {}
            payload_patch = {"meta": {**existing_meta, **meta}}
        self._repo.update_progress(
            self.queue_job_id,
            step=step,
            message=message,
            percent=percent,
            payload_patch=payload_patch,
        )

    def complete(self, result: dict[str, Any], *, message: str = "") -> None:
        self._repo.complete_job(
            self.queue_job_id,
            result=result,
            message=message or "Task complete",
        )

    def fail(self, error: str, *, result: dict | None = None) -> None:
        self._repo.fail_job(self.queue_job_id, error, result=result)

    def checkpoint(self) -> None:
        urgent_queue_control_checkpoint(self.queue_job_id, self._repo)

    def is_stop_requested(self) -> bool:
        status, control = self._repo.get_control_state(self.queue_job_id)
        return control == "stop" or status == "cancelled"


def urgent_queue_control_checkpoint(
    job_id: int,
    repo: UrgentTaskQueueRepository | None = None,
) -> None:
    """Block while paused; raise UrgentQueueTaskStopped if stop was requested."""
    repository = repo or UrgentTaskQueueRepository()
    while True:
        status, control = repository.get_control_state(job_id)
        if control == "stop" or status == "cancelled":
            raise UrgentQueueTaskStopped("Stopped by user")
        if status != "paused":
            return
        time.sleep(0.25)


def run_urgent_task_queue_job(
    job_id: int,
    *,
    queue_repo: UrgentTaskQueueRepository | None = None,
    profile_repo: UserProfileRepository | None = None,
) -> None:
    """Run one urgent queue job to completion, failure, or cancellation."""
    queue_repo = queue_repo or UrgentTaskQueueRepository()
    profile_repo = profile_repo or UserProfileRepository()

    job = queue_repo.get_job(job_id)
    if not job:
        raise ValueError(f"Urgent queue job {job_id} not found")

    payload = job.get("payload") or {}
    progress = QueueTaskProgress(job_id, job["task_id"], queue_repo)
    profile = profile_repo.get_profile()
    task_type = job["task_type"]

    try:
        if task_type == "single_search":
            run_single_search_workflow(
                progress,
                keyword=payload["keyword"],
                location=payload["location"],
                max_jobs=int(payload.get("max_jobs", 10)),
                sources=payload.get("sources", ""),
                source_list=list(payload.get("source_list") or []),
                mode=payload.get("mode", "both"),
                profile=profile,
                search_filters=payload.get("search_filters"),
            )
            return

        if task_type == "linkedin_job_import":
            run_linkedin_import_workflow(
                progress,
                linkedin_url=payload["linkedin_url"],
                return_folder=payload.get("return_folder", "all"),
                return_search=payload.get("return_search", ""),
            )
            return

        queue_repo.fail_job(job_id, f"Unknown urgent task type: {task_type}")
    except UrgentQueueTaskStopped as exc:
        current = queue_repo.get_job(job_id)
        if current and current["status"] == "running":
            queue_repo.mark_cancelled(
                job_id,
                message=str(exc),
                result=current.get("result") or {},
            )
    except Exception as exc:
        logger.exception("Urgent queue job %s failed", job_id)
        queue_repo.fail_job(job_id, str(exc))
