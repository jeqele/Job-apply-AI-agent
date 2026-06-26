"""Parse linkedin-mcp-server tool payloads into HermesHire job dicts."""

from __future__ import annotations

import re
from typing import Any

from job_apply_ai.scraper.job_metadata import empty_job_details, infer_work_type
from job_apply_ai.scraper.linkedin_job_url import parse_linkedin_job_url

_JOB_ID_RE = re.compile(r"/jobs/view/(\d+)")
_SKIP_TITLE_PREFIXES = (
    "show more",
    "show less",
    "apply",
    "save",
    "share",
    "report",
    "linkedin",
    "sign in",
)


def job_id_from_url(url: str) -> str | None:
    match = _JOB_ID_RE.search(url or "")
    return match.group(1) if match else None


def _canonical_job_url(job_id: str) -> str:
    return parse_linkedin_job_url(f"https://www.linkedin.com/jobs/view/{job_id}") or (
        f"https://www.linkedin.com/jobs/view/{job_id}"
    )


def _reference_titles(payload: dict[str, Any]) -> dict[str, str]:
    titles: dict[str, str] = {}
    references = payload.get("references") or {}
    for section_refs in references.values():
        if not isinstance(section_refs, list):
            continue
        for ref in section_refs:
            if not isinstance(ref, dict) or ref.get("kind") != "job":
                continue
            job_id = job_id_from_url(ref.get("url", ""))
            text = (ref.get("text") or "").strip()
            if job_id and text:
                titles[job_id] = text
    return titles


def _usable_title(line: str) -> bool:
    cleaned = line.strip()
    if len(cleaned) < 3:
        return False
    lower = cleaned.lower()
    return not any(lower.startswith(prefix) for prefix in _SKIP_TITLE_PREFIXES)


def _parse_job_posting_text(text: str, url: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = ""
    company = ""
    location = ""

    for line in lines[:12]:
        if not title and _usable_title(line) and len(line) < 120:
            title = line
            continue
        if title and not company and _usable_title(line) and len(line) < 120:
            if "·" in line:
                parts = [part.strip() for part in line.split("·") if part.strip()]
                if parts:
                    company = parts[0]
                if len(parts) > 1:
                    location = parts[-1]
            else:
                company = line
            continue
        if title and company and not location and _usable_title(line) and len(line) < 120:
            location = line
            break

    return {
        "title": title,
        "company": company,
        "location": location,
        "description": text.strip(),
    }


def map_date_posted_filter(max_days_old: int) -> str | None:
    """Map HermesHire max_days_old to linkedin-mcp date_posted values."""
    if max_days_old <= 1:
        return "past_24_hours"
    if max_days_old <= 7:
        return "past_week"
    if max_days_old <= 30:
        return "past_month"
    return None


def jobs_from_search_payload(
    payload: dict[str, Any],
    *,
    keyword: str,
    location: str,
    max_jobs: int,
) -> list[dict]:
    """Build stub job rows from a search_jobs MCP payload."""
    titles_by_id = _reference_titles(payload)
    job_ids = payload.get("job_ids") or []
    jobs: list[dict] = []

    for job_id in job_ids:
        if len(jobs) >= max_jobs:
            break
        if not str(job_id).isdigit():
            continue
        job_id = str(job_id)
        link = _canonical_job_url(job_id)
        title = titles_by_id.get(job_id) or f"{keyword} (LinkedIn)"
        jobs.append(
            {
                **empty_job_details(),
                "title": title,
                "company": "",
                "location": location,
                "work_type": infer_work_type(title, location),
                "link": link,
                "posted_days_ago": "Unknown",
                "fetch_method": "scrape",
            }
        )
    return jobs


def job_from_details_payload(payload: dict[str, Any]) -> dict:
    """Merge a get_job_details MCP payload into a job dict."""
    url = payload.get("url") or ""
    text = (payload.get("sections") or {}).get("job_posting", "")
    parsed = _parse_job_posting_text(text, url)
    job_id = job_id_from_url(url)
    link = parse_linkedin_job_url(url) or url

    title = parsed["title"]
    if not title and job_id:
        title = f"LinkedIn job {job_id}"

    location = parsed["location"]
    description = parsed["description"]

    posted_date = ""
    posted_days_ago: int | str = "Unknown"
    for line in text.splitlines()[:20]:
        lower = line.lower()
        if "reposted" in lower or "posted" in lower:
            if "week" in lower:
                posted_days_ago = 7
            elif "day" in lower:
                match = re.search(r"(\d+)\s+day", lower)
                posted_days_ago = int(match.group(1)) if match else 1
            elif "hour" in lower:
                posted_days_ago = 0
            break

    return {
        **empty_job_details(),
        "title": title,
        "company": parsed["company"],
        "location": location,
        "work_type": infer_work_type(title, location, description),
        "description": description,
        "link": link,
        "posted_date": posted_date,
        "posted_days_ago": posted_days_ago,
        "fetch_method": "scrape",
    }
