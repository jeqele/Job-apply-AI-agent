import os
import tempfile
import unittest

from job_apply_ai.job_dedupe import compute_dedupe_key, dedupe_jobs, normalize_job_link
from job_apply_ai.storage.database import init_db
from job_apply_ai.storage.job_repository import JobRepository


class JobDedupeTests(unittest.TestCase):
    def test_normalize_job_link_strips_tracking_params(self):
        raw = "https://www.linkedin.com/jobs/view/123456?utm_source=google&ref=abc"
        self.assertEqual(
            normalize_job_link(raw),
            "https://www.linkedin.com/jobs/view/123456",
        )

    def test_normalize_job_link_canonicalizes_indeed(self):
        first = "https://uk.indeed.com/viewjob?jk=abc123&utm_source=newsletter"
        second = "https://uk.indeed.com/viewjob?jk=abc123&from=share"
        self.assertEqual(normalize_job_link(first), normalize_job_link(second))

    def test_compute_dedupe_key_uses_identity_without_link(self):
        job = {
            "title": "Python Developer",
            "company": "Acme Ltd",
            "location": "London",
            "source": "linkedin",
        }
        same_job_other_source = {**job, "source": "indeed"}
        self.assertEqual(
            compute_dedupe_key(job),
            compute_dedupe_key(same_job_other_source),
        )

    def test_dedupe_jobs_merges_cross_source_matches(self):
        jobs = [
            {
                "title": "Python Developer",
                "company": "Acme Ltd",
                "location": "London",
                "source": "linkedin",
            },
            {
                "title": "Python Developer",
                "company": "Acme Ltd",
                "location": "London",
                "source": "indeed",
            },
        ]
        self.assertEqual(len(dedupe_jobs(jobs)), 1)


class JobRepositoryDedupeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = os.path.join(self.temp_dir.name, "jobs.db")
        os.environ["JOB_APPLY_AI_DB"] = self.db_path
        init_db(self.db_path)
        self.repo = JobRepository()

    def tearDown(self):
        os.environ.pop("JOB_APPLY_AI_DB", None)
        self.temp_dir.cleanup()

    def test_upsert_jobs_updates_existing_row_by_dedupe_key(self):
        first_id = self.repo.upsert_jobs(
            [
                {
                    "title": "Backend Engineer",
                    "company": "Example Co",
                    "location": "Remote",
                    "link": "https://www.linkedin.com/jobs/view/999?utm_source=x",
                    "description": "First pass",
                }
            ],
            search_run_id=1,
        )[0]
        second_id = self.repo.upsert_jobs(
            [
                {
                    "title": "Backend Engineer",
                    "company": "Example Co",
                    "location": "Remote",
                    "link": "https://www.linkedin.com/jobs/view/999?ref=y",
                    "description": "Updated pass",
                }
            ],
            search_run_id=2,
        )[0]

        self.assertEqual(first_id, second_id)
        job = self.repo.get_job(first_id)
        self.assertEqual(job["description"], "Updated pass")
        self.assertEqual(job["search_run_id"], 2)
        self.assertEqual(self.repo.count_jobs(), 1)

    def test_create_job_reuses_existing_dedupe_key(self):
        first_id = self.repo.create_job(
            {
                "title": "Manual Role",
                "company": "Manual Co",
                "location": "Leeds",
            }
        )
        second_id = self.repo.create_job(
            {
                "title": "Manual Role",
                "company": "Manual Co",
                "location": "Leeds",
            }
        )
        self.assertEqual(first_id, second_id)
        self.assertEqual(self.repo.count_jobs(), 1)


if __name__ == "__main__":
    unittest.main()
