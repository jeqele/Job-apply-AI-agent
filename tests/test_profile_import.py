"""Tests for profile merge and CV import helpers."""

import os
import unittest

from job_apply_ai.cv_modifier.profile_importer import ProfileImporter
from job_apply_ai.storage.user_profile import merge_profiles, skill_names, summarize_import_changes


class ProfileMergeTests(unittest.TestCase):
    def test_merge_fills_empty_scalar_fields(self):
        base = {"full_name": "", "email": "existing@example.com", "technical_skills": []}
        incoming = {
            "full_name": "Jane Doe",
            "email": "jane@example.com",
            "technical_skills": ["Python"],
        }
        merged, changes = merge_profiles(base, incoming)
        self.assertEqual(merged["full_name"], "Jane Doe")
        self.assertEqual(merged["email"], "existing@example.com")
        self.assertEqual(skill_names(merged["technical_skills"]), ["Python"])
        self.assertIn("full_name", changes["filled_fields"])
        self.assertEqual(changes["added_technical_skills"], ["Python"])

    def test_merge_deduplicates_skills(self):
        base = {"full_name": "Jane", "technical_skills": ["Python", "Flask"]}
        incoming = {"full_name": "Jane Doe", "technical_skills": ["python", "Docker"]}
        merged, changes = merge_profiles(base, incoming)
        self.assertEqual(skill_names(merged["technical_skills"]), ["Python", "Flask", "Docker"])
        self.assertEqual(changes["added_technical_skills"], ["Docker"])

    def test_merge_adds_new_experience_and_bullets(self):
        base = {
            "full_name": "Jane",
            "work_experience": [
                {
                    "role": "Developer",
                    "company": "Acme",
                    "period": "2020",
                    "bullets": ["Built APIs"],
                }
            ],
        }
        incoming = {
            "full_name": "Jane Doe",
            "work_experience": [
                {
                    "role": "Developer",
                    "company": "Acme",
                    "period": "2020",
                    "bullets": ["Built APIs", "Led team"],
                },
                {
                    "role": "Intern",
                    "company": "Beta",
                    "period": "2018",
                    "bullets": ["Fixed bugs"],
                },
            ],
        }
        merged, changes = merge_profiles(base, incoming)
        self.assertEqual(len(merged["work_experience"]), 2)
        self.assertEqual(merged["work_experience"][0]["bullets"], ["Built APIs", "Led team"])
        self.assertEqual(len(changes["added_work_experience"]), 1)
        self.assertIn("Led team", changes["added_bullets"])

    def test_summarize_import_changes(self):
        lines = summarize_import_changes(
            {
                "filled_fields": ["email"],
                "added_technical_skills": ["Docker"],
                "added_bullets": ["Led team"],
            }
        )
        self.assertTrue(any("Filled Email" in line for line in lines))
        self.assertTrue(any("Docker" in line for line in lines))


class ProfileImporterTests(unittest.TestCase):
    def test_heuristic_import_from_sample_cv(self):
        sample_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Full_Stack .docx",
        )
        if not os.path.exists(sample_path):
            self.skipTest("Sample CV not available")

        importer = ProfileImporter(llm=object())
        importer.llm = type(
            "UnavailableLLM",
            (),
            {"is_available": lambda self: False},
        )()
        profile = importer.extract_from_docx(sample_path)
        self.assertEqual(profile["full_name"], "Amin Khalili")
        self.assertIn("amin.khalily@hotmail.com", profile["email"])
        self.assertTrue(profile["work_experience"])
        self.assertTrue(profile["technical_skills"] or profile["tools_platforms"])


if __name__ == "__main__":
    unittest.main()
