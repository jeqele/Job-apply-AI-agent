"""LinkedIn job source wrapper around the existing scraper."""

from job_apply_ai.scraper.base import JobSource
from job_apply_ai.scraper.email_extractor import enrich_job_emails
from job_apply_ai.scraper.linkedin import LinkedInScraper


class LinkedInJobSource(JobSource):
    source_name = "LinkedIn"
    supports_api = False
    supports_scrape = True

    def __init__(self, headless: bool = True):
        super().__init__(headless=headless)
        self._scraper = LinkedInScraper(headless=headless)

    def fetch_via_api(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
        **kwargs,
    ) -> list[dict]:
        return []

    def fetch_via_scrape(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
        **kwargs,
    ) -> list[dict]:
        return self._scraper.scrape_job_listings(
            keyword,
            location,
            max_jobs=max_jobs,
            max_days_old=max_days_old,
            search_filters=kwargs.get("search_filters"),
        )

    def fetch_job_details_batch(self, jobs: list[dict]) -> list[dict]:
        jobs = self._scraper.fetch_job_descriptions(jobs)
        for job in jobs:
            enrich_job_emails(job, fetch_page=True)
        return jobs
