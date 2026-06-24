"""Tests for batch job search queue helpers."""

import tempfile
import unittest
from pathlib import Path

from job_apply_ai.batch_search import (
    build_search_queue,
    get_max_batch_search_combinations,
    parse_lines,
    parse_lines_from_path,
    shuffle_search_queue,
    validate_batch_queue,
)


class BatchSearchTests(unittest.TestCase):
    def test_parse_lines_skips_blank_and_comments(self):
        content = """
        Software Engineer

        # comment line
        Data Scientist
        """
        self.assertEqual(
            parse_lines(content),
            ["Software Engineer", "Data Scientist"],
        )

    def test_parse_lines_from_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "titles.txt"
            path.write_text("Backend Developer\n\nFrontend Developer\n", encoding="utf-8")
            self.assertEqual(
                parse_lines_from_path(path),
                ["Backend Developer", "Frontend Developer"],
            )

    def test_build_search_queue_cartesian_product(self):
        queue = build_search_queue(
            ["Engineer", "Analyst"],
            ["Berlin", "Remote"],
        )
        self.assertEqual(
            queue,
            [
                ("Engineer", "Berlin"),
                ("Engineer", "Remote"),
                ("Analyst", "Berlin"),
                ("Analyst", "Remote"),
            ],
        )

    def test_validate_batch_queue_rejects_empty(self):
        self.assertEqual(
            validate_batch_queue([]),
            "Provide at least one job title and one location.",
        )

    def test_validate_batch_queue_rejects_too_many(self):
        titles = [f"Title {index}" for index in range(11)]
        locations = [f"City {index}" for index in range(10)]
        queue = build_search_queue(titles, locations)
        limit = get_max_batch_search_combinations()
        self.assertEqual(len(queue), 110)
        self.assertGreater(len(queue), limit)
        self.assertIn("Too many search combinations", validate_batch_queue(queue))

    def test_shuffle_search_queue_preserves_items(self):
        queue = build_search_queue(["Engineer", "Analyst"], ["Berlin", "Remote"])
        shuffled = shuffle_search_queue(queue)
        self.assertEqual(sorted(shuffled), sorted(queue))
        self.assertEqual(len(shuffled), len(queue))


if __name__ == "__main__":
    unittest.main()
