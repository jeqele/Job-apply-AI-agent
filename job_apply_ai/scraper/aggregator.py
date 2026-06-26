"""Aggregate job searches across multiple sources."""

import logging
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from job_apply_ai.scraper.adzuna import AdzunaJobSource
from job_apply_ai.scraper.arbeitnow import ArbeitnowJobSource
from job_apply_ai.scraper.base import JobSource
from job_apply_ai.scraper.cv_library import CVLibraryJobSource
from job_apply_ai.scraper.indeed import IndeedJobSource
from job_apply_ai.scraper.jobs_io import dedupe_jobs, save_jobs_to_db, save_jobs_to_excel
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.scraper.linkedin_mcp_source import LinkedInMcpJobSource
from job_apply_ai.scraper.linkedin_source import LinkedInJobSource
from job_apply_ai.scraper.reed import ReedJobSource
from job_apply_ai.scraper.remoteok import RemoteOKJobSource
from job_apply_ai.scraper.search_filters import SearchFilters
from job_apply_ai.scraper.totaljobs import TotaljobsJobSource

load_dotenv()

logger = logging.getLogger(__name__)

AVAILABLE_SOURCES: dict[str, type[JobSource]] = {
    "linkedin": LinkedInJobSource,
    "linkedin-mcp": LinkedInMcpJobSource,
    "adzuna": AdzunaJobSource,
    "reed": ReedJobSource,
    "indeed": IndeedJobSource,
    "totaljobs": TotaljobsJobSource,
    "cv-library": CVLibraryJobSource,
    "remoteok": RemoteOKJobSource,
    "arbeitnow": ArbeitnowJobSource,
}

DEFAULT_SOURCES = ["linkedin", "adzuna", "reed", "indeed", "totaljobs", "cv-library"]


def resolve_sources(sources: Iterable[str] | None = None) -> list[str]:
    """Resolve requested source names."""
    if not sources:
        return DEFAULT_SOURCES.copy()

    resolved = []
    for source in sources:
        name = source.strip().lower()
        if name == "all":
            return list(AVAILABLE_SOURCES.keys())
        if name in AVAILABLE_SOURCES:
            resolved.append(name)
        else:
            logger.warning("Unknown job source '%s' ignored", source)
    return resolved or DEFAULT_SOURCES.copy()


def search_jobs(
    keyword: str,
    location: str,
    max_jobs: int = 10,
    max_days_old: int = 30,
    sources: Iterable[str] | None = None,
    mode: str = "both",
    headless: bool = True,
    enrich_details: bool = True,
    search_filters: SearchFilters | None = None,
) -> list[dict]:
    """Search jobs across multiple sources."""
    filters = search_filters or SearchFilters()
    keyword, location = filters.augment_query(keyword, location)
    selected_sources = filters.expand_sources(resolve_sources(sources))
    per_source_limit = max(1, max_jobs * filters.fetch_multiplier())

    all_jobs: list[dict] = []
    for source_name in selected_sources:
        source_cls = AVAILABLE_SOURCES[source_name]
        source = source_cls(headless=headless)
        logger.info("Searching %s (%s mode)", source.source_name, mode)
        jobs = source.search(
            keyword,
            location,
            max_jobs=per_source_limit,
            max_days_old=max_days_old,
            mode=mode,
            search_filters=filters,
        )
        if enrich_details:
            jobs = source.enrich_jobs(jobs, deep_fetch=True)
        all_jobs.extend(jobs)

    filtered_jobs = filters.filter_jobs(dedupe_jobs(all_jobs))
    return filtered_jobs[:max_jobs]


def search_and_save(
    keyword: str,
    location: str,
    output_file: str | None = None,
    save_excel: bool = True,
    **kwargs,
) -> Tuple[List[dict], Optional[str]]:
    """Search jobs, persist to SQLite, and optionally export to Excel."""
    sources = kwargs.get("sources")
    mode = kwargs.get("mode", "both")
    sources_str = ",".join(sources) if isinstance(sources, (list, tuple)) else str(sources or "")

    jobs = search_jobs(keyword, location, **kwargs)

    repo = JobRepository()
    search_run_id = repo.create_search_run(keyword, location, sources_str, mode)
    save_jobs_to_db(jobs, search_run_id=search_run_id)

    filename = save_jobs_to_excel(jobs, output_file) if save_excel and jobs else None
    return jobs, filename
