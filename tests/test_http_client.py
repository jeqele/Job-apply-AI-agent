"""Tests for scraper HTTP helpers and API/scrape fallback behavior."""

import time
import unittest
from unittest.mock import MagicMock, patch

import requests

from job_apply_ai.scraper.base import JobSource
from job_apply_ai.scraper.http_client import get_with_retry


class StubJobSource(JobSource):
    source_name = "Stub"
    supports_api = True
    supports_scrape = True

    def __init__(self, api_jobs=None, scrape_jobs=None, api_error=None):
        super().__init__()
        self.api_jobs = api_jobs or []
        self.scrape_jobs = scrape_jobs or []
        self.api_error = api_error

    def fetch_via_api(self, keyword, location, max_jobs=10, max_days_old=30, **kwargs):
        if self.api_error:
            raise self.api_error
        return self.api_jobs

    def fetch_via_scrape(self, keyword, location, max_jobs=10, max_days_old=30, **kwargs):
        return self.scrape_jobs


class HttpClientTests(unittest.TestCase):
    def test_get_with_retry_retries_on_429(self):
        too_many = MagicMock()
        too_many.status_code = 429
        too_many.headers = {"Retry-After": "0"}
        too_many.raise_for_status.side_effect = requests.HTTPError(response=too_many)

        ok = MagicMock()
        ok.status_code = 200
        ok.raise_for_status.return_value = None

        with patch("job_apply_ai.scraper.http_client.requests.get", side_effect=[too_many, ok]) as get:
            with patch("job_apply_ai.scraper.http_client.time.sleep"):
                response = get_with_retry(
                    "https://www.adzuna.co.uk/search",
                    min_interval=0,
                    max_retries=1,
                )

        self.assertIs(response, ok)
        self.assertEqual(get.call_count, 2)

    def test_get_with_retry_rate_limits_same_host(self):
        with patch("job_apply_ai.scraper.http_client.requests.get") as get:
            ok = MagicMock()
            ok.status_code = 200
            ok.raise_for_status.return_value = None
            get.return_value = ok

            with patch("job_apply_ai.scraper.http_client._last_request_at", {}):
                with patch("job_apply_ai.scraper.http_client.time.sleep") as sleep:
                    get_with_retry("https://www.adzuna.co.uk/a", min_interval=2.0, max_retries=0)
                    get_with_retry("https://www.adzuna.co.uk/b", min_interval=2.0, max_retries=0)

        self.assertTrue(sleep.called)


class JobSourceSearchTests(unittest.TestCase):
    def test_both_mode_skips_scrape_when_api_succeeds(self):
        source = StubJobSource(
            api_jobs=[{"title": "API Job", "company": "Co", "location": "London", "link": "http://a"}],
            scrape_jobs=[{"title": "Scrape Job", "company": "Co", "location": "London", "link": "http://b"}],
        )

        jobs = source.search("engineer", "London", mode="both")

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["title"], "API Job")
        self.assertEqual(jobs[0]["fetch_method"], "api")

    def test_both_mode_scrapes_when_api_fails(self):
        source = StubJobSource(
            api_error=RuntimeError("missing credentials"),
            scrape_jobs=[{"title": "Scrape Job", "company": "Co", "location": "London", "link": "http://b"}],
        )

        jobs = source.search("engineer", "London", mode="both")

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["title"], "Scrape Job")
        self.assertEqual(jobs[0]["fetch_method"], "scrape")


if __name__ == "__main__":
    unittest.main()
