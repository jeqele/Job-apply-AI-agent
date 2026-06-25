"""Parse and normalize LinkedIn job posting URLs."""

import re
from urllib.parse import parse_qs, urlparse

from job_apply_ai.job_dedupe import normalize_job_link

_LINKEDIN_JOB_PATH = re.compile(r"/jobs/view/(\d+)", re.IGNORECASE)
_LINKEDIN_NETLOC = re.compile(r"(^|\.)linkedin\.(com|cn)$", re.IGNORECASE)


def is_linkedin_job_url(url: str) -> bool:
    """Return True when the URL refers to a LinkedIn job posting."""
    return parse_linkedin_job_url(url) is not None


def parse_linkedin_job_url(url: str) -> str | None:
    """
    Return a canonical LinkedIn job view URL, or None if the input is invalid.

    Supports direct view links and collection/search URLs that carry currentJobId.
    """
    raw = (url or "").strip()
    if not raw:
        return None

    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    netloc = (parsed.netloc or "").lower()
    if not _LINKEDIN_NETLOC.search(netloc):
        return None

    job_id = None
    path_match = _LINKEDIN_JOB_PATH.search(parsed.path or "")
    if path_match:
        job_id = path_match.group(1)
    else:
        current_job_id = parse_qs(parsed.query).get("currentJobId", [None])[0]
        if current_job_id and str(current_job_id).isdigit():
            job_id = str(current_job_id)

    if not job_id:
        return None

    canonical = f"https://www.linkedin.com/jobs/view/{job_id}"
    return normalize_job_link(canonical) or canonical
