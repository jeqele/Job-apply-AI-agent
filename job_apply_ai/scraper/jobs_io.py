"""Shared helpers for normalizing and saving job listings."""

import logging
from datetime import datetime
from typing import List, Optional

from job_apply_ai.scraper.email_extractor import enrich_job_emails, infer_application_method
from job_apply_ai.scraper.job_metadata import (
    empty_job_details,
    extract_relocation_info,
    extract_salary,
    infer_work_type,
    parse_relocation_support,
    parse_visa_sponsorship,
)

logger = logging.getLogger(__name__)

from job_apply_ai.job_dedupe import dedupe_jobs
from job_apply_ai.job_schema import JOB_COLUMNS


def normalize_job(job: dict, source: str, fetch_method: str = "") -> dict:
    """Ensure a job dict contains the standard schema."""
    normalized = {**empty_job_details(), **job}
    normalized["source"] = source
    normalized["fetch_method"] = fetch_method or job.get("fetch_method", "")
    normalized.setdefault("emails", "")
    normalized.setdefault("application_method", infer_application_method(normalized))
    return normalized


def enrich_job_metadata(job: dict) -> dict:
    """Fill derived metadata fields from title, location, and description."""
    description = job.get("description", "")
    if not job.get("work_type") or job.get("work_type") == "Not specified":
        job["work_type"] = infer_work_type(
            job.get("title", ""),
            job.get("location", ""),
            description,
        )
    if not job.get("salary"):
        job["salary"] = extract_salary(description)
    if not job.get("visa_sponsorship") or job.get("visa_sponsorship") == "Not mentioned":
        job["visa_sponsorship"] = parse_visa_sponsorship(description)
    if not job.get("relocation_support") or job.get("relocation_support") == "Not mentioned":
        job["relocation_support"] = parse_relocation_support(description)
    if not job.get("relocation_info"):
        job["relocation_info"] = extract_relocation_info(description)
    enrich_job_emails(job, fetch_page=False)
    return job


def save_jobs_to_excel(jobs: List[dict], filename: Optional[str] = None) -> Optional[str]:
    """Save job listings to Excel with a consistent column order."""
    if not jobs:
        logger.warning("No jobs to save")
        return None

    from job_apply_ai.storage.exports import export_jobs

    if filename is None:
        today_date = datetime.today().strftime("%Y-%m-%d")
        filename = f"jobs_{today_date}.xlsx"

    export_jobs(jobs, "excel", filename)
    return filename


def save_jobs_to_db(
    jobs: List[dict],
    search_run_id: Optional[int] = None,
) -> List[int]:
    """Persist job listings to SQLite."""
    if not jobs:
        return []

    from job_apply_ai.storage.job_repository import JobRepository

    repo = JobRepository()
    return repo.upsert_jobs(jobs, search_run_id=search_run_id)
