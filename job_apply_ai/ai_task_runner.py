"""Execute AI task queue jobs (shared by worker and tests)."""

from __future__ import annotations

import logging
import time
from typing import Any

from job_apply_ai.cv_workflows import (
    run_ats_friendly_workflow,
    run_batch_ats_friendly_workflow,
    run_batch_cv_workflow,
    run_job_match_analyze_workflow,
    run_profile_import_workflow,
    run_single_cv_workflow,
)
from job_apply_ai.storage.ai_task_queue_repository import AiTaskQueueRepository
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.storage.user_profile import UserProfileRepository
from job_apply_ai.ui.cv_tasks import TaskStopped

logger = logging.getLogger(__name__)


class AiQueueTaskStopped(TaskStopped):
    """Raised when an AI queue job should exit early."""


class QueueTaskProgress:
    """Bridge queue repository updates to cv_workflows TaskProgress protocol."""

    def __init__(
        self,
        queue_job_id: int,
        task_id: str,
        repo: AiTaskQueueRepository,
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
        ai_queue_control_checkpoint(self.queue_job_id, self._repo)

    def is_stop_requested(self) -> bool:
        status, control = self._repo.get_control_state(self.queue_job_id)
        return control == "stop" or status == "cancelled"


def ai_queue_control_checkpoint(
    job_id: int,
    repo: AiTaskQueueRepository | None = None,
) -> None:
    """Block while paused; raise AiQueueTaskStopped if stop was requested."""
    repository = repo or AiTaskQueueRepository()
    while True:
        status, control = repository.get_control_state(job_id)
        if control == "stop" or status == "cancelled":
            raise AiQueueTaskStopped("Stopped by user")
        if status != "paused":
            return
        time.sleep(0.25)


def _jobs_for_manage_folder(
    folder: str,
    search: str,
    job_repo: JobRepository,
) -> list[dict]:
    workflow_status = None if folder == "all" else folder
    exclude_statuses = ["archived"] if folder == "all" else None
    return job_repo.list_jobs(
        workflow_status=workflow_status,
        search=search or None,
        exclude_workflow_statuses=exclude_statuses,
    )


def _jobs_by_ids(job_ids: list[int], job_repo: JobRepository) -> list[dict]:
    jobs = []
    for job_id in job_ids:
        job = job_repo.get_job(job_id)
        if job:
            jobs.append(job)
    return jobs


def run_ai_task_queue_job(
    job_id: int,
    *,
    queue_repo: AiTaskQueueRepository | None = None,
    job_repo: JobRepository | None = None,
    profile_repo: UserProfileRepository | None = None,
) -> None:
    """Run one AI queue job to completion, failure, or cancellation."""
    queue_repo = queue_repo or AiTaskQueueRepository()
    job_repo = job_repo or JobRepository()
    profile_repo = profile_repo or UserProfileRepository()

    job = queue_repo.get_job(job_id)
    if not job:
        raise ValueError(f"AI queue job {job_id} not found")

    payload = job.get("payload") or {}
    progress = QueueTaskProgress(job_id, job["task_id"], queue_repo)
    profile = profile_repo.get_profile()
    task_type = job["task_type"]

    try:
        if task_type == "single_cv":
            target_job_id = int(payload.get("job_id") or job.get("job_id") or 0)
            target_job = job_repo.get_job(target_job_id)
            if not target_job:
                queue_repo.fail_job(job_id, f"Job {target_job_id} not found")
                return
            run_single_cv_workflow(
                progress,
                profile=profile,
                job=target_job,
                job_id=target_job_id,
                return_folder=payload.get("return_folder", "all"),
                return_search=payload.get("return_search", ""),
                return_from_manage=bool(payload.get("return_from_manage")),
                return_sort=payload.get("return_sort", ""),
                job_repo=job_repo,
            )
            return

        if task_type == "batch_cv":
            job_ids = payload.get("job_ids") or []
            jobs = _jobs_by_ids([int(item) for item in job_ids], job_repo)
            if not jobs:
                queue_repo.fail_job(job_id, "No jobs found for batch CV generation")
                return
            run_batch_cv_workflow(progress, profile=profile, jobs=jobs, job_repo=job_repo)
            return

        if task_type == "ats_friendly":
            target_job_id = int(payload.get("job_id") or job.get("job_id") or 0)
            target_job = job_repo.get_job(target_job_id)
            if not target_job:
                queue_repo.fail_job(job_id, f"Job {target_job_id} not found")
                return
            cv_filename = target_job.get("cv_filename") or payload.get("cv_filename", "")
            if not cv_filename:
                queue_repo.fail_job(job_id, "No CV found for this job. Generate a CV first.")
                return
            run_ats_friendly_workflow(
                progress,
                job=target_job,
                job_id=target_job_id,
                profile=profile,
                cv_filename=cv_filename,
                return_folder=payload.get("return_folder", "all"),
                return_search=payload.get("return_search", ""),
                return_from_manage=bool(payload.get("return_from_manage")),
                return_sort=payload.get("return_sort", ""),
            )
            return

        if task_type == "batch_ats_friendly":
            return_folder = payload.get("return_folder", "all")
            return_search = payload.get("return_search", "")
            jobs = _jobs_for_manage_folder(return_folder, return_search, job_repo)
            run_batch_ats_friendly_workflow(
                progress,
                jobs=jobs,
                profile=profile,
                return_folder=return_folder,
                return_search=return_search,
                return_sort=payload.get("return_sort", ""),
                job_repo=job_repo,
            )
            return

        if task_type == "job_match_analyze":
            return_folder = payload.get("return_folder", "all")
            return_search = payload.get("return_search", "")
            jobs = _jobs_for_manage_folder(return_folder, return_search, job_repo)
            if not jobs:
                queue_repo.fail_job(job_id, "No jobs to analyze in this folder.")
                return
            run_job_match_analyze_workflow(
                progress,
                jobs=jobs,
                profile=profile,
                min_match_score=float(payload.get("min_match_score", 50)),
                return_folder=return_folder,
                return_search=return_search,
                return_sort=payload.get("return_sort", ""),
                job_repo=job_repo,
            )
            return

        if task_type == "profile_import":
            cv_path = payload.get("cv_path", "")
            if not cv_path:
                queue_repo.fail_job(job_id, "Missing CV file path for profile import")
                return
            run_profile_import_workflow(
                progress,
                cv_path=cv_path,
                current_profile=profile_repo.get_profile(),
            )
            return

        queue_repo.fail_job(job_id, f"Unknown AI task type: {task_type}")
    except AiQueueTaskStopped as exc:
        current = queue_repo.get_job(job_id)
        if current and current["status"] == "running":
            queue_repo.mark_cancelled(
                job_id,
                message=str(exc),
                result=current.get("result") or {},
            )
    except Exception as exc:
        logger.exception("AI queue job %s failed", job_id)
        queue_repo.fail_job(job_id, str(exc))
