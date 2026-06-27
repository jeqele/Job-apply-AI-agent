"""Urgent UI-bound I/O workflows: single search and LinkedIn scrape."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from job_apply_ai.batch_search_runner import enrich_jobs_with_skills
from job_apply_ai.cv_modifier.job_match_analyzer import classify_jobs_by_profile_match
from job_apply_ai.cv_workflows import TaskProgress
from job_apply_ai.dev_logging import dev_agent, dev_task
from job_apply_ai.job_schema import JOB_COLUMNS
from job_apply_ai.job_status import DEFAULT_JOB_STATUS
from job_apply_ai.scraper.aggregator import search_jobs as aggregate_search_jobs
from job_apply_ai.scraper.linkedin import LinkedInScraper
from job_apply_ai.scraper.linkedin_job_url import parse_linkedin_job_url
from job_apply_ai.scraper.search_filters import SearchFilters
from job_apply_ai.storage.batch_queue_repository import filters_from_dict
from job_apply_ai.storage.job_repository import JobRepository

logger = logging.getLogger(__name__)


class UrgentTaskProgress(Protocol):
    task_id: str

    def update(
        self,
        *,
        status: str | None = None,
        step: str | None = None,
        message: str | None = None,
        percent: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None: ...

    def complete(self, result: dict[str, Any], *, message: str = "") -> None: ...

    def fail(self, error: str, *, result: dict | None = None) -> None: ...

    def checkpoint(self) -> None: ...

    def is_stop_requested(self) -> bool: ...


def build_job_from_linkedin_scrape(url: str, details: dict) -> dict:
    """Merge scraped LinkedIn details into a job record for create/edit forms."""
    job = {column: details.get(column, "") or "" for column in JOB_COLUMNS}
    job["link"] = parse_linkedin_job_url(url) or url.strip()
    job["source"] = "LinkedIn"
    job["fetch_method"] = "scrape"
    job["workflow_status"] = DEFAULT_JOB_STATUS
    return job


def run_single_search_workflow(
    progress: TaskProgress | UrgentTaskProgress,
    *,
    keyword: str,
    location: str,
    max_jobs: int,
    sources: str,
    source_list: list[str],
    mode: str,
    profile: dict,
    search_filters: SearchFilters | dict | None = None,
    job_repo: JobRepository | None = None,
) -> None:
    repository = job_repo or JobRepository()
    filters = (
        search_filters
        if isinstance(search_filters, SearchFilters)
        else filters_from_dict(search_filters if isinstance(search_filters, dict) else None)
    )

    progress.update(
        status="running",
        step="searching",
        message=f"Searching for {keyword} in {location}…",
        percent=5,
        meta={"keyword": keyword, "location": location},
    )

    try:
        progress.checkpoint()
    except Exception:
        progress.fail("Job search stopped before fetching results.")
        return

    search_run_id = repository.create_search_run(keyword, location, sources, mode)

    progress.update(
        status="running",
        step="fetching",
        message="Fetching jobs from sources…",
        percent=20,
        meta={"keyword": keyword, "location": location},
    )

    try:
        jobs = aggregate_search_jobs(
            keyword,
            location,
            max_jobs=max_jobs,
            sources=source_list,
            mode=mode,
            enrich_details=True,
            search_filters=filters,
        )
    except Exception as exc:
        logger.error("Single search failed for %r in %r: %s", keyword, location, exc)
        progress.fail(f"Job search failed: {exc}")
        return

    try:
        progress.checkpoint()
    except Exception:
        if not jobs:
            progress.fail("Job search stopped before saving any jobs.")
            return

        processed_jobs = enrich_jobs_with_skills(jobs)
        processed_jobs = classify_jobs_by_profile_match(processed_jobs, profile)
        repository.upsert_jobs(processed_jobs, search_run_id=search_run_id)
        progress.complete(
            {
                "search_run_id": search_run_id,
                "total_jobs": len(processed_jobs),
                "stopped": True,
            },
            message=f"Search stopped — saved {len(processed_jobs)} jobs",
        )
        return

    if not jobs:
        progress.fail("No jobs found. Try different search terms or adjust filters.")
        return

    progress.update(
        status="running",
        step="processing",
        message="Matching skills and profile…",
        percent=75,
        meta={"keyword": keyword, "location": location},
    )

    processed_jobs = enrich_jobs_with_skills(jobs)
    processed_jobs = classify_jobs_by_profile_match(processed_jobs, profile)
    repository.upsert_jobs(processed_jobs, search_run_id=search_run_id)

    progress.complete(
        {
            "search_run_id": search_run_id,
            "total_jobs": len(processed_jobs),
        },
        message=f"Search complete — saved {len(processed_jobs)} jobs",
    )


def run_linkedin_import_workflow(
    progress: TaskProgress | UrgentTaskProgress,
    *,
    linkedin_url: str,
    return_folder: str,
    return_search: str,
) -> None:
    with dev_task(progress.task_id, "linkedin_job_import"):
        progress.update(
            status="running",
            step="opening",
            message="Opening the LinkedIn job page…",
            percent=15,
            meta={"linkedin_url": linkedin_url},
        )
        scraper = LinkedInScraper(headless=True)
        with dev_agent("LinkedInScraper"):
            details = scraper.fetch_job_details(linkedin_url)

        progress.update(
            step="building",
            message="Building job details…",
            percent=85,
        )
        job = build_job_from_linkedin_scrape(linkedin_url, details)
        if not job.get("title"):
            progress.fail(
                "Could not extract a job title from that LinkedIn page. "
                "The listing may require sign-in or the page layout may have changed."
            )
            return

        progress.complete(
            {
                "job": job,
                "return_folder": return_folder,
                "return_search": return_search,
            },
            message="LinkedIn job imported successfully",
        )
