"""Search preference filters for job queries and result matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SearchFilters:
    """Optional filters for remote work, relocation, and visa sponsorship."""

    remote: bool = False
    relocation: bool = False
    visa_sponsorship: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, str] | None) -> SearchFilters:
        """Build filters from a form mapping or dict-like object."""
        if not data:
            return cls()
        return cls(
            remote=_is_checked(data, "filter_remote"),
            relocation=_is_checked(data, "filter_relocation"),
            visa_sponsorship=_is_checked(data, "filter_visa_sponsorship"),
        )

    def any_active(self) -> bool:
        return self.remote or self.relocation or self.visa_sponsorship

    def augment_query(self, keyword: str, location: str) -> tuple[str, str]:
        """Bias search queries toward jobs matching active filters."""
        keyword = (keyword or "").strip()
        location = (location or "").strip()
        extras: list[str] = []

        if self.visa_sponsorship and "visa" not in keyword.lower():
            extras.append("visa sponsorship")
        if self.relocation and "relocation" not in keyword.lower():
            extras.append("relocation")
        if self.remote:
            keyword_lower = keyword.lower()
            location_lower = location.lower()
            if "remote" not in keyword_lower and "remote" not in location_lower:
                extras.append("remote")

        if extras:
            keyword = f"{keyword} {' '.join(extras)}".strip()
        return keyword, location

    def expand_sources(self, sources: list[str]) -> list[str]:
        """Add remote- or visa-focused sources when matching filters are enabled."""
        if "all" in sources:
            return sources

        expanded = list(sources)
        if self.remote and "remoteok" not in expanded:
            expanded.append("remoteok")
        if (self.remote or self.visa_sponsorship) and "arbeitnow" not in expanded:
            expanded.append("arbeitnow")
        return expanded

    def matches_job(self, job: dict) -> bool:
        """Return True when a job satisfies all active filters."""
        if self.remote and not _job_is_remote(job):
            return False
        if self.visa_sponsorship and job.get("visa_sponsorship") != "Yes":
            return False
        if self.relocation and job.get("relocation_support") not in {"Yes", "Mentioned"}:
            return False
        return True

    def filter_jobs(self, jobs: list[dict]) -> list[dict]:
        if not self.any_active():
            return jobs
        return [job for job in jobs if self.matches_job(job)]

    def fetch_multiplier(self) -> int:
        """Fetch extra listings when post-filtering is enabled."""
        return 3 if self.any_active() else 1


def _is_checked(data: Mapping[str, str], field_name: str) -> bool:
    value = data.get(field_name)
    if value is None:
        return False
    return str(value).lower() in {"1", "on", "true", "yes"}


def _job_is_remote(job: dict) -> bool:
    work_type = (job.get("work_type") or "").lower()
    location = (job.get("location") or "").lower()
    if "remote" in work_type or "remote" in location:
        return True
    return False
