"""Arbeitnow job source (public JSON API, Europe-focused)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from job_apply_ai.scraper.base import JobSource
from job_apply_ai.scraper.email_extractor import enrich_job_emails
from job_apply_ai.scraper.http_client import get_with_retry
from job_apply_ai.scraper.job_metadata import extract_salary, infer_work_type, parse_visa_sponsorship
from job_apply_ai.scraper.search_filters import SearchFilters

logger = logging.getLogger(__name__)


def _posted_from_timestamp(timestamp: int | str | None) -> tuple[str, int | str]:
    if not timestamp:
        return "", "Unknown"
    try:
        posted_at = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return "", "Unknown"
    posted_date = posted_at.strftime("%Y-%m-%d")
    posted_days_ago = (datetime.now(tz=timezone.utc).date() - posted_at.date()).days
    return posted_date, posted_days_ago


def _normalize_list_field(value: list | str | None) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _parse_arbeitnow_job(result: dict, fallback_location: str) -> dict:
    description = result.get("description", "") or ""
    location = result.get("location") or fallback_location
    is_remote = bool(result.get("remote"))
    posted_date, posted_days_ago = _posted_from_timestamp(result.get("created_at"))
    tags = _normalize_list_field(result.get("tags"))
    job_types = _normalize_list_field(result.get("job_types"))

    job = {
        "title": result.get("title", ""),
        "company": result.get("company_name", ""),
        "location": location,
        "work_type": "Remote" if is_remote else infer_work_type(result.get("title", ""), location, description),
        "salary": extract_salary(description),
        "employment_type": job_types,
        "link": result.get("url", ""),
        "company_url": "",
        "description": description,
        "industry": tags,
        "posted_date": posted_date,
        "posted_days_ago": posted_days_ago,
        "visa_sponsorship": parse_visa_sponsorship(description),
    }
    enrich_job_emails(job, fetch_page=False)
    return job


class ArbeitnowJobSource(JobSource):
    source_name = "Arbeitnow"
    supports_api = True
    supports_scrape = False

    API_URL = "https://www.arbeitnow.com/api/job-board-api"
    PER_PAGE = 100

    def fetch_via_api(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
        **kwargs,
    ) -> list[dict]:
        filters: SearchFilters = kwargs.get("search_filters") or SearchFilters()
        params: dict[str, str | int] = {"page": 1}
        keyword = (keyword or "").strip()
        location = (location or "").strip()
        if keyword:
            params["search"] = keyword
        if location:
            params["location"] = location
        if filters.remote:
            params["remote"] = "true"
        if filters.visa_sponsorship:
            params["visa_sponsorship"] = "true"

        jobs: list[dict] = []
        page = 1
        while len(jobs) < max_jobs:
            params["page"] = page
            response = get_with_retry(self.API_URL, params=params, timeout=20)
            payload = response.json()
            results = payload.get("data") or []
            if not results:
                break

            for result in results:
                if not isinstance(result, dict) or not result.get("url"):
                    continue
                posted_date, posted_days_ago = _posted_from_timestamp(result.get("created_at"))
                if (
                    max_days_old
                    and isinstance(posted_days_ago, int)
                    and posted_days_ago > max_days_old
                ):
                    continue
                jobs.append(_parse_arbeitnow_job(result, location))
                if len(jobs) >= max_jobs:
                    break

            if len(jobs) >= max_jobs or not payload.get("links", {}).get("next"):
                break
            page += 1

        return jobs[:max_jobs]

    def fetch_via_scrape(
        self,
        keyword: str,
        location: str,
        max_jobs: int = 10,
        max_days_old: int = 30,
        **kwargs,
    ) -> list[dict]:
        return []
