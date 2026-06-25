import unittest

from job_apply_ai.scraper.linkedin_job_url import is_linkedin_job_url, parse_linkedin_job_url


class LinkedInJobUrlTests(unittest.TestCase):
    def test_parse_direct_view_link(self):
        url = "https://www.linkedin.com/jobs/view/4431490621/"
        self.assertEqual(
            parse_linkedin_job_url(url),
            "https://www.linkedin.com/jobs/view/4431490621",
        )
        self.assertTrue(is_linkedin_job_url(url))

    def test_parse_view_link_without_scheme(self):
        url = "www.linkedin.com/jobs/view/4431490621"
        self.assertEqual(
            parse_linkedin_job_url(url),
            "https://www.linkedin.com/jobs/view/4431490621",
        )

    def test_parse_view_link_strips_tracking_params(self):
        url = "https://www.linkedin.com/jobs/view/4431490621/?trackingId=abc&utm_source=share"
        self.assertEqual(
            parse_linkedin_job_url(url),
            "https://www.linkedin.com/jobs/view/4431490621",
        )

    def test_parse_current_job_id_from_collection_url(self):
        url = "https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4431490621"
        self.assertEqual(
            parse_linkedin_job_url(url),
            "https://www.linkedin.com/jobs/view/4431490621",
        )

    def test_rejects_non_linkedin_urls(self):
        self.assertIsNone(parse_linkedin_job_url("https://example.com/jobs/view/123"))
        self.assertFalse(is_linkedin_job_url("https://example.com/jobs/view/123"))

    def test_rejects_linkedin_non_job_urls(self):
        self.assertIsNone(parse_linkedin_job_url("https://www.linkedin.com/in/someone"))
        self.assertFalse(is_linkedin_job_url("https://www.linkedin.com/company/acme"))


if __name__ == "__main__":
    unittest.main()
