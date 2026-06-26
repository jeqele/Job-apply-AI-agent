"""HTTP helpers with per-host rate limiting and retry for scrapers."""

from __future__ import annotations

import logging
import os
import threading
import time
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; JobApplyAI/1.0)"
DEFAULT_MIN_INTERVAL = float(os.environ.get("SCRAPE_MIN_INTERVAL_SECONDS", "2.0"))
DEFAULT_MAX_RETRIES = int(os.environ.get("SCRAPE_MAX_RETRIES", "3"))
DEFAULT_BACKOFF_BASE = float(os.environ.get("SCRAPE_BACKOFF_BASE_SECONDS", "2.0"))

_lock = threading.Lock()
_last_request_at: dict[str, float] = {}


def _host_key(url: str) -> str:
    return urlparse(url).netloc.lower()


def _wait_for_rate_limit(host: str, min_interval: float) -> None:
    if min_interval <= 0:
        return
    with _lock:
        now = time.monotonic()
        last = _last_request_at.get(host, 0.0)
        wait = min_interval - (now - last)
        if wait > 0:
            time.sleep(wait)
        _last_request_at[host] = time.monotonic()


def _retry_wait_seconds(
    response: requests.Response | None,
    attempt: int,
    backoff_base: float,
) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), backoff_base)
            except ValueError:
                pass
    return backoff_base * (2**attempt)


def get_with_retry(
    url: str,
    *,
    timeout: float = 20,
    headers: dict | None = None,
    params: dict | None = None,
    min_interval: float | None = None,
    max_retries: int | None = None,
    backoff_base: float | None = None,
    **kwargs,
) -> requests.Response:
    """GET with per-host spacing and exponential backoff on 429/503."""
    interval = DEFAULT_MIN_INTERVAL if min_interval is None else min_interval
    retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
    backoff = DEFAULT_BACKOFF_BASE if backoff_base is None else backoff_base

    merged_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged_headers.update(headers)

    host = _host_key(url)
    last_error: requests.RequestException | None = None

    for attempt in range(retries + 1):
        _wait_for_rate_limit(host, interval)
        try:
            response = requests.get(
                url,
                params=params,
                headers=merged_headers,
                timeout=timeout,
                **kwargs,
            )
            if response.status_code in (429, 503) and attempt < retries:
                wait = _retry_wait_seconds(response, attempt, backoff)
                logger.warning(
                    "%s returned %s; retrying in %.1fs (attempt %d/%d)",
                    host,
                    response.status_code,
                    wait,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (429, 503) and attempt < retries:
                wait = _retry_wait_seconds(exc.response, attempt, backoff)
                logger.warning(
                    "%s returned %s; retrying in %.1fs (attempt %d/%d)",
                    host,
                    status,
                    wait,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait)
                last_error = exc
                continue
            raise
        except requests.RequestException as exc:
            if attempt < retries:
                wait = backoff * (2**attempt)
                logger.warning(
                    "Request to %s failed: %s; retrying in %.1fs",
                    host,
                    exc,
                    wait,
                )
                time.sleep(wait)
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise last_error
    raise requests.RequestException(f"Failed to GET {url} after {retries} retries")
