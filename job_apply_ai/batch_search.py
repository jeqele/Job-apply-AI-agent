"""Batch job search helpers: parse input files and build title × location queues."""

from __future__ import annotations

import os
import random
import time
from itertools import product
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

DEFAULT_MAX_BATCH_SEARCH_COMBINATIONS = 100
DEFAULT_BATCH_QUEUE_MAX_COMBINATIONS_PER_JOB = 50
_LINKEDIN_SOURCES = frozenset({"linkedin", "linkedin-mcp"})


def get_max_batch_search_combinations() -> int:
    """Return max title × location pairs per batch (from env or default)."""
    load_dotenv()
    raw = os.environ.get("MAX_BATCH_SEARCH_COMBINATIONS", "").strip()
    if not raw:
        return DEFAULT_MAX_BATCH_SEARCH_COMBINATIONS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_BATCH_SEARCH_COMBINATIONS


# Backward-compatible alias for the built-in default.
MAX_BATCH_SEARCH_COMBINATIONS = DEFAULT_MAX_BATCH_SEARCH_COMBINATIONS


def get_batch_queue_max_combinations_per_job() -> int:
    """Return max title × location pairs per queue job when auto-splitting."""
    load_dotenv()
    raw = os.environ.get("BATCH_QUEUE_MAX_COMBINATIONS_PER_JOB", "").strip()
    if not raw:
        return DEFAULT_BATCH_QUEUE_MAX_COMBINATIONS_PER_JOB
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_BATCH_QUEUE_MAX_COMBINATIONS_PER_JOB


def parse_lines(content: str) -> list[str]:
    """Parse newline-separated lines, skipping blanks and # comments."""
    lines: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def parse_lines_from_path(path: str | Path) -> list[str]:
    """Read a UTF-8 text file and return non-empty lines."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return parse_lines(text)


def decode_uploaded_text(raw: bytes) -> str:
    """Decode uploaded file bytes, preferring UTF-8 with BOM support."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def build_search_queue(titles: list[str], locations: list[str]) -> list[tuple[str, str]]:
    """Return every title paired with every location."""
    if not titles or not locations:
        return []
    return [(title, location) for title, location in product(titles, locations)]


def shuffle_search_queue(queue: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return a randomly shuffled copy of the search queue."""
    shuffled = list(queue)
    random.shuffle(shuffled)
    return shuffled


def split_batch_inputs(
    titles: list[str],
    locations: list[str],
    max_combinations: int | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Split titles/locations into chunks each within the combination limit."""
    if not titles or not locations:
        return []
    limit = (
        max_combinations
        if max_combinations is not None
        else get_batch_queue_max_combinations_per_job()
    )
    total = len(titles) * len(locations)
    if total <= limit:
        return [(titles, locations)]

    if len(locations) <= limit:
        chunk_size = max(1, limit // len(locations))
        return [
            (titles[index : index + chunk_size], locations)
            for index in range(0, len(titles), chunk_size)
        ]

    if len(titles) <= limit:
        chunk_size = max(1, limit // len(titles))
        return [
            (titles, locations[index : index + chunk_size])
            for index in range(0, len(locations), chunk_size)
        ]

    chunk_size = max(1, limit // len(locations))
    parts: list[tuple[list[str], list[str]]] = []
    for index in range(0, len(titles), chunk_size):
        sub_titles = titles[index : index + chunk_size]
        if len(sub_titles) * len(locations) <= limit:
            parts.append((sub_titles, locations))
        else:
            parts.extend(split_batch_inputs(sub_titles, locations, limit))
    return parts


def validate_batch_queue(queue: list[tuple[str, str]]) -> str | None:
    """Return an error message when the queue is invalid, else None."""
    if not queue:
        return "Provide at least one job title and one location."
    limit = get_max_batch_search_combinations()
    if len(queue) > limit:
        return (
            f"Too many search combinations ({len(queue)}). "
            f"Maximum is {limit} (titles × locations)."
        )
    return None


def _uses_linkedin_sources(sources: Iterable[str] | None) -> bool:
    if not sources:
        return False
    return any(source.strip().lower() in _LINKEDIN_SOURCES for source in sources)


def batch_search_pause(sources: Iterable[str] | None = None) -> None:
    """Sleep between batch search iterations to reduce upstream rate limits."""
    if _uses_linkedin_sources(sources):
        default = os.environ.get("LINKEDIN_MCP_BATCH_DELAY_SECONDS", "30")
    else:
        default = "1.0"
    configured = os.environ.get("BATCH_SEARCH_DELAY_SECONDS", "").strip()
    delay = float(configured if configured else default)
    if delay > 0:
        time.sleep(delay)
