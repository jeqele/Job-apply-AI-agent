"""Job scraper package."""

from job_apply_ai.scraper.aggregator import AVAILABLE_SOURCES, search_and_save, search_jobs

__all__ = ["AVAILABLE_SOURCES", "search_jobs", "search_and_save"]
