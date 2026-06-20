"""Remote OK job source (public JSON API + scrape fallback)."""

import logging
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from job_apply_ai.scraper.base import JobSource
from job_apply_ai.scraper.email_extractor import enrich_job_emails
from job_apply_ai.scraper.job_metadata import extract_salary, infer_work_type

logger = logging.getLogger(__name__)


class RemoteOKJobSource(JobSource):
    source_name = "RemoteOK"
    supports_api = True
    supports_scrape = True

    API_URL = "https://remoteok.com/api"
    WEB_URL = "https://remoteok.com/remote-dev-jobs"

    def fetch_via_api(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        response = requests.get(
            self.API_URL,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JobApplyAI/1.0)"},
        )
        response.raise_for_status()
        payload = response.json()

        keyword_terms = [term.lower() for term in keyword.replace("/", " ").split() if term]
        jobs = []
        for result in payload:
            if not isinstance(result, dict) or not result.get("url"):
                continue
            if "legal" in result:
                continue

            searchable = " ".join(
                [
                    result.get("position", ""),
                    result.get("company", ""),
                    result.get("description", ""),
                    " ".join(result.get("tags") or []),
                ]
            ).lower()
            if keyword_terms and not all(term in searchable for term in keyword_terms[:2]):
                continue

            description = result.get("description", "")
            tags = ", ".join(result.get("tags") or [])
            job = {
                "title": result.get("position", result.get("title", "")),
                "company": result.get("company", ""),
                "location": result.get("location") or "Remote",
                "work_type": "Remote",
                "salary": extract_salary(str(result.get("salary_min", "")), str(result.get("salary_max", "")), description),
                "employment_type": result.get("employment_type", ""),
                "link": result.get("url", ""),
                "company_url": result.get("company_url", ""),
                "description": description,
                "industry": tags,
                "posted_date": result.get("date", ""),
            }
            enrich_job_emails(job, fetch_page=False)
            jobs.append(job)
            if len(jobs) >= max_jobs:
                break
        return jobs

    def fetch_via_scrape(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
    ) -> list[dict]:
        response = requests.get(
            self.WEB_URL,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JobApplyAI/1.0)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        jobs = []

        for row in soup.select("tr.job")[:max_jobs]:
            title_elem = row.select_one("h2[itemprop='title'], .company_and_position h2")
            company_elem = row.select_one("h3[itemprop='name'], .companyLink")
            link_elem = row.select_one("a.preventLink")
            if not title_elem:
                continue

            link = link_elem.get("href", "") if link_elem else ""
            if link and link.startswith("/"):
                link = f"https://remoteok.com{link}"

            job = {
                "title": title_elem.get_text(" ", strip=True),
                "company": company_elem.get_text(" ", strip=True) if company_elem else "",
                "location": "Remote",
                "work_type": "Remote",
                "link": link,
            }
            enrich_job_emails(job, html=str(row), fetch_page=False)
            jobs.append(job)
        return jobs
