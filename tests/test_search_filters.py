"""Tests for search preference filters."""

import unittest

from job_apply_ai.scraper.search_filters import SearchFilters


class SearchFiltersTests(unittest.TestCase):
    def test_from_mapping_reads_checkboxes(self):
        filters = SearchFilters.from_mapping(
            {
                "filter_remote": "on",
                "filter_visa_sponsorship": "true",
            }
        )
        self.assertTrue(filters.remote)
        self.assertFalse(filters.relocation)
        self.assertTrue(filters.visa_sponsorship)

    def test_augment_query_adds_terms(self):
        filters = SearchFilters(remote=True, relocation=True, visa_sponsorship=True)
        keyword, location = filters.augment_query("Engineer", "Berlin")
        self.assertIn("remote", keyword.lower())
        self.assertIn("relocation", keyword.lower())
        self.assertIn("visa sponsorship", keyword.lower())
        self.assertEqual(location, "Berlin")

    def test_augment_query_skips_duplicate_remote(self):
        filters = SearchFilters(remote=True)
        keyword, location = filters.augment_query("Remote Engineer", "Berlin")
        self.assertEqual(keyword, "Remote Engineer")

    def test_expand_sources_adds_remoteok(self):
        filters = SearchFilters(remote=True)
        sources = filters.expand_sources(["linkedin", "indeed"])
        self.assertEqual(sources, ["linkedin", "indeed", "remoteok"])

    def test_filter_jobs_requires_all_active_filters(self):
        filters = SearchFilters(remote=True, visa_sponsorship=True)
        jobs = [
            {
                "title": "Remote role",
                "work_type": "Remote",
                "visa_sponsorship": "Yes",
            },
            {
                "title": "Remote no visa",
                "work_type": "Remote",
                "visa_sponsorship": "No",
            },
            {
                "title": "On-site with visa",
                "work_type": "On-site",
                "visa_sponsorship": "Yes",
            },
        ]
        filtered = filters.filter_jobs(jobs)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["title"], "Remote role")

    def test_relocation_filter_accepts_mentioned(self):
        filters = SearchFilters(relocation=True)
        jobs = [
            {"relocation_support": "Yes"},
            {"relocation_support": "Mentioned"},
            {"relocation_support": "No"},
        ]
        filtered = filters.filter_jobs(jobs)
        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
