"""Base class for job listing sources."""

import logging
from abc import ABC, abstractmethod

from job_apply_ai.scraper.email_extractor import enrich_jobs_with_emails
from job_apply_ai.scraper.jobs_io import dedupe_jobs, enrich_job_metadata, normalize_job

logger = logging.getLogger(__name__)


class JobSource(ABC):
    """Common interface for API and scrape-based job sources."""

    source_name = "Unknown"
    supports_api = False
    supports_scrape = False

    def __init__(self, headless: bool = True):
        self.headless = headless

    @abstractmethod
    def fetch_via_api(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        """Fetch jobs using an official or public API."""

    @abstractmethod
    def fetch_via_scrape(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        """Fetch jobs by scraping the website."""

    def search(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
        mode: str = "both",
    ) -> list[dict]:
        """Search jobs using API, scrape, or both."""
        jobs: list[dict] = []

        if mode in {"api", "both"} and self.supports_api:
            try:
                api_jobs = self.fetch_via_api(keyword, location, max_jobs, max_days_old)
                jobs.extend(
                    normalize_job(job, self.source_name, fetch_method="api")
                    for job in api_jobs
                )
                logger.info("%s API returned %s jobs", self.source_name, len(api_jobs))
            except Exception as exc:
                logger.warning("%s API failed: %s", self.source_name, exc)

        if mode in {"scrape", "both"} and self.supports_scrape:
            try:
                scrape_jobs = self.fetch_via_scrape(keyword, location, max_jobs, max_days_old)
                jobs.extend(
                    normalize_job(job, self.source_name, fetch_method="scrape")
                    for job in scrape_jobs
                )
                logger.info("%s scrape returned %s jobs", self.source_name, len(scrape_jobs))
            except Exception as exc:
                logger.warning("%s scrape failed: %s", self.source_name, exc)

        jobs = dedupe_jobs(jobs)[:max_jobs]
        for job in jobs:
            enrich_job_metadata(job)
        return jobs

    def enrich_jobs(self, jobs: list[dict], deep_fetch: bool = True) -> list[dict]:
        """Fetch extra details and contact emails for jobs from this source."""
        if deep_fetch and hasattr(self, "fetch_job_details_batch"):
            jobs = self.fetch_job_details_batch(jobs)  # type: ignore[attr-defined]
        for job in jobs:
            enrich_job_metadata(job)
        enrich_jobs_with_emails(jobs, fetch_pages=deep_fetch)
        return jobs
