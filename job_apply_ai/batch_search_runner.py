"""Execute batch search queue jobs (shared by worker and tests)."""

from __future__ import annotations

import logging
from typing import Callable

from job_apply_ai.batch_search import batch_search_pause, build_search_queue, shuffle_search_queue
from job_apply_ai.cv_modifier.cv_analyzer import CVAnalyzer
from job_apply_ai.cv_modifier.job_match_analyzer import classify_jobs_by_profile_match
from job_apply_ai.scraper.aggregator import search_jobs as aggregate_search_jobs
from job_apply_ai.scraper.search_filters import SearchFilters
from job_apply_ai.storage.batch_queue_repository import (
    BatchQueueRepository,
    filters_from_dict,
)
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.storage.user_profile import UserProfileRepository
from job_apply_ai.ui.cv_tasks import TaskStopped

logger = logging.getLogger(__name__)


class QueueTaskStopped(TaskStopped):
    """Raised when a queue job should exit early."""


def enrich_jobs_with_skills(jobs: list[dict]) -> list[dict]:
    """Extract matched skills from job descriptions."""
    analyzer = CVAnalyzer()
    enriched = []
    for job in jobs:
        if job.get("description"):
            matched_skills, _, matched_categories = analyzer.extract_skills_from_description(
                job["description"]
            )
            job["matched_skills"] = matched_skills
            job["matched_categories"] = matched_categories
        enriched.append(job)
    return enriched


def queue_control_checkpoint(
    job_id: int,
    repo: BatchQueueRepository | None = None,
) -> None:
    """Block while paused; raise QueueTaskStopped if stop was requested."""
    repository = repo or BatchQueueRepository()
    import time

    while True:
        status, control = repository.get_control_state(job_id)
        if control == "stop" or status == "cancelled":
            raise QueueTaskStopped("Stopped by user")
        if status != "paused":
            return
        time.sleep(0.25)


def run_batch_search_queue_job(
    job_id: int,
    *,
    queue_repo: BatchQueueRepository | None = None,
    job_repo: JobRepository | None = None,
    profile_repo: UserProfileRepository | None = None,
    checkpoint: Callable[[int], None] | None = None,
) -> None:
    """Run one batch search queue job to completion, failure, or cancellation."""
    queue_repo = queue_repo or BatchQueueRepository()
    job_repo = job_repo or JobRepository()
    profile_repo = profile_repo or UserProfileRepository()
    checkpoint_fn = checkpoint or queue_control_checkpoint

    job = queue_repo.get_job(job_id)
    if not job:
        raise ValueError(f"Queue job {job_id} not found")

    queue = build_search_queue(job["titles"], job["locations"])
    if job["shuffle_queue"]:
        queue = shuffle_search_queue(queue)

    total = len(queue)
    unique_titles = len({keyword for keyword, _ in queue})
    unique_locations = len({location for _, location in queue})
    source_list = [source.strip() for source in job["sources"].split(",") if source.strip()]
    search_filters = filters_from_dict(job.get("search_filters"))
    profile = profile_repo.get_profile()

    search_run_id = job_repo.create_search_run(
        f"batch: {unique_titles} title(s)",
        f"batch: {unique_locations} location(s)",
        job["sources"],
        job["mode"],
    )

    total_jobs_saved = 0
    failed_searches: list[dict] = []
    stopped = False
    searches_completed = 0

    queue_repo.update_progress(
        job_id,
        step="searching",
        message=f"Starting batch search ({total} combinations)…",
        percent=1,
    )

    for index, (keyword, location) in enumerate(queue, start=1):
        try:
            checkpoint_fn(job_id)
        except QueueTaskStopped:
            stopped = True
            break

        percent = max(1, min(99, int(((index - 1) / total) * 100)))
        queue_repo.update_progress(
            job_id,
            step="searching",
            message=f"Searching {index} of {total}",
            percent=percent,
            current_index=index,
            result_patch={
                "current_keyword": keyword,
                "current_location": location,
            },
        )

        try:
            jobs = aggregate_search_jobs(
                keyword,
                location,
                max_jobs=job["max_jobs"],
                sources=source_list,
                mode=job["mode"],
                enrich_details=True,
                search_filters=search_filters,
            )
            if jobs:
                processed_jobs = enrich_jobs_with_skills(jobs)
                processed_jobs = classify_jobs_by_profile_match(processed_jobs, profile)
                job_repo.upsert_jobs(processed_jobs, search_run_id=search_run_id)
                total_jobs_saved += len(processed_jobs)
            searches_completed += 1
        except Exception as exc:
            logger.error(
                "Batch search failed for %r in %r: %s",
                keyword,
                location,
                exc,
            )
            failed_searches.append(
                {
                    "keyword": keyword,
                    "location": location,
                    "error": str(exc),
                }
            )
            searches_completed += 1

        if index < total:
            batch_search_pause(source_list)

    result = {
        "search_run_id": search_run_id,
        "total_jobs": total_jobs_saved,
        "searches_run": searches_completed,
        "failed_searches": failed_searches,
    }

    if stopped:
        if total_jobs_saved == 0:
            queue_repo.mark_cancelled(
                job_id,
                message="Batch search stopped before saving any jobs.",
                result={**result, "stopped": True},
            )
            return

        message = (
            f"Batch search stopped — saved {total_jobs_saved} jobs "
            f"after {searches_completed} of {total} searches"
        )
        if failed_searches:
            message += f" ({len(failed_searches)} searches failed)"
        queue_repo.complete_job(
            job_id,
            search_run_id=search_run_id,
            result={**result, "stopped": True},
            message=message,
            reschedule=False,
        )
        return

    if total_jobs_saved == 0:
        detail = (
            f"{len(failed_searches)} of {total} searches failed."
            if failed_searches
            else "No jobs matched any title/location combination."
        )
        queue_repo.fail_job(job_id, f"Batch search found no jobs. {detail}", result=result)
        return

    message = f"Batch search complete — saved {total_jobs_saved} jobs"
    if failed_searches:
        message += f" ({len(failed_searches)} searches failed)"

    queue_repo.complete_job(
        job_id,
        search_run_id=search_run_id,
        result={**result, "searches_run": total},
        message=message,
        reschedule=job["schedule_type"] in {"daily", "weekly"},
    )
