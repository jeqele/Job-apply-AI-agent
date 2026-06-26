"""LinkedIn job source backed by the local linkedin-mcp-server sidecar."""

from __future__ import annotations

import logging
import os
import time

from job_apply_ai.scraper.base import JobSource
from job_apply_ai.scraper.linkedin_mcp_client import LinkedInMcpError, call_linkedin_mcp_tool
from job_apply_ai.scraper.linkedin_mcp_parser import (
    job_from_details_payload,
    job_id_from_url,
    jobs_from_search_payload,
    map_date_posted_filter,
)
from job_apply_ai.scraper.linkedin_job_url import parse_linkedin_job_url
from job_apply_ai.scraper.search_filters import SearchFilters

logger = logging.getLogger(__name__)


def _max_search_pages(max_jobs: int) -> int:
    configured = os.environ.get("LINKEDIN_MCP_MAX_PAGES", "").strip()
    if configured:
        return max(1, min(10, int(configured)))
    return max(1, min(3, (max_jobs + 24) // 25))


def _detail_pause() -> None:
    delay = float(os.environ.get("LINKEDIN_MCP_DETAIL_DELAY_SECONDS", "2.0"))
    if delay > 0:
        time.sleep(delay)


def _work_type_filter(search_filters: SearchFilters | None) -> str | None:
    if search_filters and search_filters.remote:
        return "remote"
    return None


class LinkedInMcpJobSource(JobSource):
    """Search LinkedIn via authenticated browser session exposed by linkedin-mcp-server."""

    source_name = "LinkedIn (MCP)"
    supports_api = False
    supports_scrape = True

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
        search_filters = kwargs.get("search_filters")
        arguments: dict = {
            "keywords": keyword,
            "location": location or None,
            "max_pages": _max_search_pages(max_jobs),
            "date_posted": map_date_posted_filter(max_days_old),
            "sort_by": "date",
        }
        work_type = _work_type_filter(search_filters)
        if work_type:
            arguments["work_type"] = work_type

        logger.info(
            "LinkedIn MCP search: keywords=%r location=%r max_pages=%s",
            keyword,
            location,
            arguments["max_pages"],
        )
        payload = call_linkedin_mcp_tool("search_jobs", arguments)
        jobs = jobs_from_search_payload(
            payload,
            keyword=keyword,
            location=location,
            max_jobs=max_jobs,
        )
        if not jobs and payload.get("section_errors"):
            raise LinkedInMcpError(f"LinkedIn MCP search failed: {payload['section_errors']}")
        return jobs

    def fetch_job_details_batch(self, jobs: list[dict]) -> list[dict]:
        enriched: list[dict] = []
        for index, job in enumerate(jobs):
            link = job.get("link") or ""
            canonical = parse_linkedin_job_url(link) or link
            job_id = None
            if canonical:
                job_id = job_id_from_url(canonical)
            if not job_id:
                enriched.append(job)
                continue
            try:
                payload = call_linkedin_mcp_tool("get_job_details", {"job_id": job_id})
                details = job_from_details_payload(payload)
                merged = {**job, **{k: v for k, v in details.items() if v}}
                if not merged.get("title"):
                    merged["title"] = job.get("title", "")
                enriched.append(merged)
            except LinkedInMcpError as exc:
                logger.warning("LinkedIn MCP detail fetch failed for %s: %s", job_id, exc)
                enriched.append(job)
            if index + 1 < len(jobs):
                _detail_pause()
        return enriched
