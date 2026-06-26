"""Tests for the Arbeitnow job source."""

import unittest
from unittest.mock import MagicMock, patch

from job_apply_ai.scraper.arbeitnow import ArbeitnowJobSource, _parse_arbeitnow_job, _posted_from_timestamp
from job_apply_ai.scraper.search_filters import SearchFilters


SAMPLE_PAYLOAD = {
    "data": [
        {
            "slug": "python-developer-berlin-123",
            "company_name": "Example GmbH",
            "title": "Python Developer",
            "description": "Build APIs with Python. Visa sponsorship available.",
            "remote": True,
            "url": "https://www.arbeitnow.com/jobs/companies/example-gmbh/python-developer-berlin-123",
            "tags": ["Python", "Remote"],
            "job_types": ["Full-time"],
            "location": "Berlin, Germany",
            "created_at": 1782495041,
        },
        {
            "slug": "old-role-456",
            "company_name": "Legacy Co",
            "title": "Old Role",
            "description": "Outdated posting",
            "remote": False,
            "url": "https://www.arbeitnow.com/jobs/companies/legacy-co/old-role-456",
            "tags": [],
            "job_types": [],
            "location": "Munich",
            "created_at": 1,
        },
    ],
    "links": {"next": None},
    "meta": {"current_page": 1, "per_page": 100},
}


class ArbeitnowHelperTests(unittest.TestCase):
    def test_posted_from_timestamp(self):
        posted_date, posted_days_ago = _posted_from_timestamp(1700000000)
        self.assertEqual(posted_date, "2023-11-14")
        self.assertIsInstance(posted_days_ago, int)

    def test_parse_arbeitnow_job_maps_fields(self):
        job = _parse_arbeitnow_job(SAMPLE_PAYLOAD["data"][0], "Berlin")
        self.assertEqual(job["title"], "Python Developer")
        self.assertEqual(job["company"], "Example GmbH")
        self.assertEqual(job["location"], "Berlin, Germany")
        self.assertEqual(job["work_type"], "Remote")
        self.assertEqual(job["employment_type"], "Full-time")
        self.assertEqual(job["industry"], "Python, Remote")
        self.assertEqual(job["link"], SAMPLE_PAYLOAD["data"][0]["url"])
        self.assertEqual(job["visa_sponsorship"], "Yes")


class ArbeitnowJobSourceTests(unittest.TestCase):
    def test_fetch_via_api_maps_results_and_filters_age(self):
        source = ArbeitnowJobSource()
        response = MagicMock()
        response.json.return_value = SAMPLE_PAYLOAD

        with patch("job_apply_ai.scraper.arbeitnow.get_with_retry", return_value=response) as get:
            jobs = source.fetch_via_api(
                "python",
                "Berlin",
                max_jobs=10,
                max_days_old=30,
            )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["title"], "Python Developer")
        get.assert_called_once()
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["search"], "python")
        self.assertEqual(params["location"], "Berlin")
        self.assertEqual(params["page"], 1)

    def test_fetch_via_api_passes_remote_and_visa_filters(self):
        source = ArbeitnowJobSource()
        response = MagicMock()
        response.json.return_value = {"data": [], "links": {}}

        filters = SearchFilters(remote=True, visa_sponsorship=True)
        with patch("job_apply_ai.scraper.arbeitnow.get_with_retry", return_value=response) as get:
            source.fetch_via_api(
                "engineer",
                "Remote",
                max_jobs=5,
                search_filters=filters,
            )

        params = get.call_args.kwargs["params"]
        self.assertEqual(params["remote"], "true")
        self.assertEqual(params["visa_sponsorship"], "true")

    def test_fetch_via_api_paginates_until_max_jobs(self):
        source = ArbeitnowJobSource()
        page_one = MagicMock()
        page_one.json.return_value = {
            "data": [
                {
                    **SAMPLE_PAYLOAD["data"][0],
                    "slug": "job-1",
                    "url": "https://www.arbeitnow.com/jobs/job-1",
                    "created_at": 1782495041,
                }
            ],
            "links": {"next": "https://www.arbeitnow.com/api/job-board-api?page=2"},
        }
        page_two = MagicMock()
        page_two.json.return_value = {
            "data": [
                {
                    **SAMPLE_PAYLOAD["data"][0],
                    "slug": "job-2",
                    "title": "Second Job",
                    "url": "https://www.arbeitnow.com/jobs/job-2",
                    "created_at": 1782495041,
                }
            ],
            "links": {"next": None},
        }

        with patch(
            "job_apply_ai.scraper.arbeitnow.get_with_retry",
            side_effect=[page_one, page_two],
        ) as get:
            jobs = source.fetch_via_api("python", "Berlin", max_jobs=2, max_days_old=9999)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[1].kwargs["params"]["page"], 2)

    def test_search_uses_api_only(self):
        source = ArbeitnowJobSource()
        with patch.object(
            source,
            "fetch_via_api",
            return_value=[{"title": "API Job", "company": "Co", "location": "Berlin", "link": "http://a"}],
        ) as fetch_api:
            with patch.object(source, "fetch_via_scrape") as fetch_scrape:
                jobs = source.search("python", "Berlin", mode="both")

        fetch_api.assert_called_once()
        fetch_scrape.assert_not_called()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["fetch_method"], "api")
        self.assertEqual(jobs[0]["source"], "Arbeitnow")


if __name__ == "__main__":
    unittest.main()
