"""Tests for LinkedIn MCP parser and batch pause behavior."""

import unittest
from unittest.mock import patch

from job_apply_ai.batch_search import batch_search_pause, validate_batch_queue
from job_apply_ai.scraper.linkedin_mcp_parser import (
    job_from_details_payload,
    jobs_from_search_payload,
    map_date_posted_filter,
)


class LinkedInMcpParserTests(unittest.TestCase):
    def test_map_date_posted_filter(self):
        self.assertEqual(map_date_posted_filter(1), "past_24_hours")
        self.assertEqual(map_date_posted_filter(7), "past_week")
        self.assertEqual(map_date_posted_filter(30), "past_month")
        self.assertIsNone(map_date_posted_filter(90))

    def test_jobs_from_search_payload_uses_references(self):
        payload = {
            "job_ids": ["123", "456"],
            "references": {
                "search_results": [
                    {"kind": "job", "url": "/jobs/view/123/", "text": "Backend Engineer"},
                    {"kind": "job", "url": "/jobs/view/456/", "text": "Platform Engineer"},
                ]
            },
        }
        jobs = jobs_from_search_payload(
            payload,
            keyword="engineer",
            location="Berlin",
            max_jobs=5,
        )
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["title"], "Backend Engineer")
        self.assertEqual(jobs[0]["link"], "https://www.linkedin.com/jobs/view/123")

    def test_job_from_details_payload_parses_title_and_description(self):
        payload = {
            "url": "https://www.linkedin.com/jobs/view/999/",
            "sections": {
                "job_posting": (
                    "Senior Python Developer\n"
                    "Acme GmbH\n"
                    "Berlin, Germany\n\n"
                    "We build APIs and data pipelines."
                )
            },
        }
        job = job_from_details_payload(payload)
        self.assertEqual(job["title"], "Senior Python Developer")
        self.assertEqual(job["company"], "Acme GmbH")
        self.assertIn("APIs", job["description"])


class BatchSearchPauseTests(unittest.TestCase):
    @patch("job_apply_ai.batch_search.time.sleep")
    def test_pause_uses_linkedin_delay_for_mcp_source(self, sleep_mock):
        with patch.dict("os.environ", {"BATCH_SEARCH_DELAY_SECONDS": "", "LINKEDIN_MCP_BATCH_DELAY_SECONDS": "25"}):
            batch_search_pause(["linkedin-mcp", "adzuna"])
        sleep_mock.assert_called_once_with(25.0)

    @patch("job_apply_ai.batch_search.time.sleep")
    def test_pause_uses_default_delay_without_linkedin(self, sleep_mock):
        with patch.dict("os.environ", {"BATCH_SEARCH_DELAY_SECONDS": "2.5"}):
            batch_search_pause(["adzuna", "reed"])
        sleep_mock.assert_called_once_with(2.5)

    def test_validate_batch_queue_enforces_limit(self):
        queue = [("a", "b")] * 3
        with patch("job_apply_ai.batch_search.get_max_batch_search_combinations", return_value=2):
            error = validate_batch_queue(queue)
        self.assertIn("Too many search combinations", error or "")


if __name__ == "__main__":
    unittest.main()
