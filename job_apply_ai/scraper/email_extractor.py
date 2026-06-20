"""Extract contact emails from job listings and descriptions."""

import logging
import re
from typing import Iterable
from urllib.parse import unquote

import requests

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
)
MAILTO_PATTERN = re.compile(
    r"mailto:([^\s\"'<>?]+)",
    re.IGNORECASE,
)

EMAIL_BLOCKLIST = (
    "noreply",
    "no-reply",
    "donotreply",
    "do-not-reply",
    "mailer-daemon",
    "example.com",
    "email.com",
    "domain.com",
    "sentry.io",
    "w3.org",
    "schema.org",
    "linkedin.com",
    "indeed.com",
    "reed.co.uk",
    "totaljobs.com",
    "cv-library.co.uk",
    "adzuna.",
    "google.com",
    "facebook.com",
    "twitter.com",
    "youtube.com",
    "instagram.com",
    "github.com",
    "privacy@",
    "support@indeed",
    "jobs@indeed",
    "feedback@",
    "newsletter@",
    "marketing@mailchimp",
)


def _normalize_email(email: str) -> str:
    email = unquote(email.strip().lower())
    email = email.rstrip(".,;)")
    return email


def _is_valid_contact_email(email: str) -> bool:
    if not email or "@" not in email:
        return False
    if any(blocked in email for blocked in EMAIL_BLOCKLIST):
        return False
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain:
        return False
    if len(local) > 64 or len(email) > 254:
        return False
    return True


def extract_emails_from_text(text: str) -> list[str]:
    """Return unique contact emails found in plain text."""
    if not text:
        return []

    found = set()
    for match in EMAIL_PATTERN.findall(text):
        email = _normalize_email(match)
        if _is_valid_contact_email(email):
            found.add(email)
    return sorted(found)


def extract_emails_from_html(html: str) -> list[str]:
    """Return unique contact emails found in HTML, including mailto links."""
    if not html:
        return []

    found = set()
    for match in MAILTO_PATTERN.findall(html):
        email = _normalize_email(match.split("?")[0])
        if _is_valid_contact_email(email):
            found.add(email)

    found.update(extract_emails_from_text(html))
    return sorted(found)


def merge_emails(*email_groups: Iterable[str]) -> str:
    """Merge email iterables into a comma-separated string."""
    merged = set()
    for group in email_groups:
        if not group:
            continue
        if isinstance(group, str):
            parts = re.split(r"[,;\s]+", group)
        else:
            parts = group
        for part in parts:
            email = _normalize_email(str(part))
            if _is_valid_contact_email(email):
                merged.add(email)
    return ", ".join(sorted(merged))


def infer_application_method(job: dict) -> str:
    """Infer how to apply based on available contact channels."""
    if job.get("emails"):
        return "email"
    if job.get("link"):
        return "url"
    return "unknown"


def enrich_job_emails(job: dict, html: str = "", fetch_page: bool = True) -> dict:
    """Populate emails and application_method on a job record."""
    description = job.get("description", "")
    emails = extract_emails_from_html(html)
    if not emails:
        emails = extract_emails_from_text(description)

    if not emails and fetch_page and job.get("link"):
        try:
            response = requests.get(
                job["link"],
                timeout=12,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            if response.ok:
                emails = extract_emails_from_html(response.text)
        except requests.RequestException as exc:
            logger.debug("Could not fetch job page for emails: %s", exc)

    job["emails"] = merge_emails(emails)
    job["application_method"] = infer_application_method(job)
    return job


def enrich_jobs_with_emails(jobs: list[dict], fetch_pages: bool = True) -> list[dict]:
    """Extract emails for each job in a list."""
    for index, job in enumerate(jobs):
        logger.info(
            "Extracting emails for job %s/%s: %s",
            index + 1,
            len(jobs),
            job.get("title", "Unknown"),
        )
        enrich_job_emails(job, fetch_page=fetch_pages)
    return jobs
