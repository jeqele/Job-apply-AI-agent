"""Shared helpers for detecting and keying duplicate job listings."""

import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "refid",
    "trk",
    "trackingid",
    "source",
    "src",
    "campaign",
    "medium",
}


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _canonical_job_path(netloc: str, path: str, query: str) -> str | None:
    """Return a source-specific canonical path when the job ID is embedded in the URL."""
    netloc_lower = netloc.lower()

    linkedin_match = re.search(r"/jobs/view/(\d+)", path, re.IGNORECASE)
    if linkedin_match and "linkedin." in netloc_lower:
        return f"/jobs/view/{linkedin_match.group(1)}"

    if "indeed." in netloc_lower:
        job_key = parse_qs(query).get("jk", [None])[0]
        if job_key:
            return f"/viewjob?jk={job_key}"

    reed_match = re.search(r"/jobs/[^/]+/(\d+)", path, re.IGNORECASE)
    if reed_match and "reed.co.uk" in netloc_lower:
        return path.rstrip("/")

    adzuna_match = re.search(r"/details/(\d+)", path, re.IGNORECASE)
    if adzuna_match and "adzuna." in netloc_lower:
        return f"/details/{adzuna_match.group(1)}"

    return None


def normalize_job_link(link: str) -> str:
    """Normalize a job URL for stable duplicate comparison."""
    raw = (link or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    if not parsed.netloc:
        return raw.lower().rstrip("/")

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"

    canonical_path = _canonical_job_path(netloc, path, parsed.query)
    if canonical_path:
        return f"{scheme}://{netloc}{canonical_path}".lower()

    filtered_query = ""
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=False)
        kept = {
            key: value
            for key, value in params.items()
            if key.lower() not in TRACKING_QUERY_PARAMS
        }
        if kept:
            filtered_query = urlencode(sorted(kept.items()), doseq=True)

    normalized = urlunparse((scheme, netloc, path, "", filtered_query, ""))
    return normalized.lower().rstrip("/")


def compute_dedupe_key(job: dict) -> str:
    """Build a stable dedupe key from a normalized link or job identity fields."""
    normalized_link = normalize_job_link(job.get("link") or "")
    if normalized_link:
        return f"link:{normalized_link}"

    title = _normalize_text(job.get("title", ""))
    company = _normalize_text(job.get("company", ""))
    location = _normalize_text(job.get("location", ""))

    if title:
        return f"identity:{title}|{company}|{location}"

    return ""


def dedupe_jobs(jobs: list[dict]) -> list[dict]:
    """Remove duplicate jobs using normalized links or title/company/location."""
    seen: set[str] = set()
    unique_jobs: list[dict] = []
    for job in jobs:
        key = compute_dedupe_key(job)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_jobs.append(job)
    return unique_jobs
