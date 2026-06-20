"""Reed.co.uk job source (API + scrape fallback)."""

import base64
import logging
import os
from datetime import datetime
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from job_apply_ai.scraper.base import JobSource
from job_apply_ai.scraper.email_extractor import enrich_job_emails
from job_apply_ai.scraper.job_metadata import extract_salary, infer_work_type

logger = logging.getLogger(__name__)


class ReedJobSource(JobSource):
    source_name = "Reed"
    supports_api = True
    supports_scrape = True

    API_URL = "https://www.reed.co.uk/api/1.0/search"
    WEB_URL = "https://www.reed.co.uk/jobs"

    def _api_headers(self) -> dict[str, str]:
        api_key = os.environ.get("REED_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Reed API key missing. Set REED_API_KEY in your environment.")
        token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}"}

    def fetch_via_api(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        params = {
            "keywords": keyword,
            "locationName": location,
            "resultsToTake": max_jobs,
        }
        response = requests.get(
            self.API_URL,
            params=params,
            headers=self._api_headers(),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()

        jobs = []
        for result in payload.get("results", [])[:max_jobs]:
            description = result.get("jobDescription", "")
            posted_date = (result.get("date") or "")[:10]
            posted_days_ago = "Unknown"
            if posted_date:
                posted_days_ago = (datetime.today() - datetime.strptime(posted_date, "%Y-%m-%d")).days

            job = {
                "title": result.get("jobTitle", ""),
                "company": result.get("employerName", ""),
                "location": result.get("locationName", location),
                "work_type": infer_work_type(result.get("jobTitle", ""), location, description),
                "salary": extract_salary(result.get("maximumSalary", ""), description),
                "employment_type": result.get("employmentType", ""),
                "link": result.get("jobUrl", ""),
                "description": description,
                "posted_days_ago": posted_days_ago,
                "posted_date": posted_date,
            }
            enrich_job_emails(job, fetch_page=False)
            jobs.append(job)
        return jobs

    def fetch_via_scrape(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        slug = f"{keyword.strip().lower().replace(' ', '-')}-jobs-in-{location.strip().lower().replace(' ', '-')}"
        url = f"{self.WEB_URL}/{quote_plus(slug)}"
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JobApplyAI/1.0)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        jobs = []

        for card in soup.select("article.job-card, article[data-id]")[:max_jobs]:
            title_elem = card.select_one("h3 a, h2 a, .job-card__title a")
            if not title_elem:
                continue

            company_elem = card.select_one(".job-card__title + p, .job-card__company, [data-qa='company-name']")
            location_elem = card.select_one(".job-card__location, [data-qa='location']")
            salary_elem = card.select_one(".job-card__salary, [data-qa='salary']")

            link = title_elem.get("href", "")
            if link and link.startswith("/"):
                link = f"https://www.reed.co.uk{link}"

            job = {
                "title": title_elem.get_text(" ", strip=True),
                "company": company_elem.get_text(" ", strip=True) if company_elem else "",
                "location": location_elem.get_text(" ", strip=True) if location_elem else location,
                "salary": salary_elem.get_text(" ", strip=True) if salary_elem else "",
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
                    ".description, .job-description, [data-qa='job-description']"
                )
                if description_elem:
                    job["description"] = description_elem.get_text("\n", strip=True)
                enrich_job_emails(job, html=response.text, fetch_page=False)
            except requests.RequestException as exc:
                logger.debug("Reed detail fetch failed for %s: %s", job.get("link"), exc)
        return jobs
