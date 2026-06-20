"""Totaljobs job source (scrape-based)."""

import logging
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from job_apply_ai.scraper.base import JobSource
from job_apply_ai.scraper.email_extractor import enrich_job_emails
from job_apply_ai.scraper.job_metadata import infer_work_type

logger = logging.getLogger(__name__)


class TotaljobsJobSource(JobSource):
    source_name = "Totaljobs"
    supports_api = False
    supports_scrape = True

    BASE_URL = "https://www.totaljobs.com/jobs"

    def fetch_via_api(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        return []

    def fetch_via_scrape(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        slug = f"{quote_plus(keyword)}/in-{quote_plus(location)}"
        url = f"{self.BASE_URL}/{slug}"
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JobApplyAI/1.0)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        jobs = []

        for card in soup.select("article[data-at='job-item'], div.job, div[data-job-id]")[:max_jobs]:
            title_elem = card.select_one("h2 a, h3 a, a[data-at='job-title-link']")
            if not title_elem:
                continue

            company_elem = card.select_one("[data-at='job-company-name'], .company")
            location_elem = card.select_one("[data-at='job-location'], .location")
            salary_elem = card.select_one("[data-at='job-salary'], .salary")

            link = title_elem.get("href", "")
            if link and link.startswith("/"):
                link = f"https://www.totaljobs.com{link}"

            title = title_elem.get_text(" ", strip=True)
            location_text = location_elem.get_text(" ", strip=True) if location_elem else location
            job = {
                "title": title,
                "company": company_elem.get_text(" ", strip=True) if company_elem else "",
                "location": location_text,
                "salary": salary_elem.get_text(" ", strip=True) if salary_elem else "",
                "work_type": infer_work_type(title, location_text),
                "link": link,
            }
            enrich_job_emails(job, html=str(card), fetch_page=False)
            jobs.append(job)
        return jobs

    def fetch_job_details_batch(self, jobs: list[dict]) -> list[dict]:
        for job in jobs:
            if job.get("description") or not job.get("link"):
                continue
            try:
                response = requests.get(
                    job["link"],
                    timeout=15,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; JobApplyAI/1.0)"},
                )
                if not response.ok:
                    continue
                soup = BeautifulSoup(response.text, "html.parser")
                description_elem = soup.select_one(
                    "[data-at='job-description'], .job-description, .description"
                )
                if description_elem:
                    job["description"] = description_elem.get_text("\n", strip=True)
                enrich_job_emails(job, html=response.text, fetch_page=False)
            except requests.RequestException as exc:
                logger.debug("Totaljobs detail fetch failed for %s: %s", job.get("link"), exc)
        return jobs
