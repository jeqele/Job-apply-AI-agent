"""Adzuna job source (API + scrape fallback)."""

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


class AdzunaJobSource(JobSource):
    source_name = "Adzuna"
    supports_api = True
    supports_scrape = True

    API_BASE = "https://api.adzuna.com/v1/api/jobs/gb/search/1"
    WEB_BASE = "https://www.adzuna.co.uk/search"

    def _credentials(self) -> tuple[str, str]:
        app_id = os.environ.get("ADZUNA_APP_ID", "").strip()
        app_key = os.environ.get("ADZUNA_APP_KEY", "").strip()
        if not app_id or not app_key:
            raise RuntimeError(
                "Adzuna API credentials missing. Set ADZUNA_APP_ID and ADZUNA_APP_KEY."
            )
        return app_id, app_key

    def fetch_via_api(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
        **kwargs,
    ) -> list[dict]:
        app_id, app_key = self._credentials()
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": keyword,
            "where": location,
            "results_per_page": max_jobs,
            "max_days_old": max_days_old,
            "content-type": "application/json",
        }
        response = requests.get(self.API_BASE, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        jobs = []
        for result in payload.get("results", [])[:max_jobs]:
            description = result.get("description", "")
            salary_min = result.get("salary_min")
            salary_max = result.get("salary_max")
            salary = ""
            if salary_min and salary_max:
                salary = f"£{int(salary_min):,} - £{int(salary_max):,}"
            elif salary_min:
                salary = f"£{int(salary_min):,}+"
            elif salary_max:
                salary = f"Up to £{int(salary_max):,}"

            created = result.get("created", "")
            posted_days_ago = "Unknown"
            posted_date = created[:10] if created else ""
            if posted_date:
                posted_days_ago = (datetime.today() - datetime.strptime(posted_date, "%Y-%m-%d")).days

            job = {
                "title": result.get("title", ""),
                "company": result.get("company", {}).get("display_name", ""),
                "location": result.get("location", {}).get("display_name", location),
                "work_type": infer_work_type(result.get("title", ""), location, description),
                "salary": salary or extract_salary(description),
                "employment_type": result.get("contract_type", ""),
                "link": result.get("redirect_url") or result.get("url", ""),
                "company_url": "",
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
        **kwargs,
    ) -> list[dict]:
        url = (
            f"{self.WEB_BASE}?q={quote_plus(keyword)}"
            f"&loc={quote_plus(location)}"
        )
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JobApplyAI/1.0)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        jobs = []

        for article in soup.select("article[data-aid]")[:max_jobs]:
            title_elem = article.select_one("h2 a")
            if not title_elem:
                continue
            company = ""
            company_elem = article.select_one("[data-testid='company-name'], .company")
            if company_elem:
                company = company_elem.get_text(" ", strip=True)

            location_text = ""
            location_elem = article.select_one("[data-testid='location'], .location")
            if location_elem:
                location_text = location_elem.get_text(" ", strip=True)

            salary = ""
            salary_elem = article.select_one(".salary, [data-testid='salary']")
            if salary_elem:
                salary = salary_elem.get_text(" ", strip=True)

            link = title_elem.get("href", "")
            if link and link.startswith("/"):
                link = f"https://www.adzuna.co.uk{link}"

            job = {
                "title": title_elem.get_text(" ", strip=True),
                "company": company,
                "location": location_text or location,
                "salary": salary,
                "link": link,
            }
            enrich_job_emails(job, html=str(article), fetch_page=False)
            jobs.append(job)
        return jobs
