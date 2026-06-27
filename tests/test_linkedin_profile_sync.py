"""Tests for LinkedIn profile parser and sync diff engine."""

import unittest

from job_apply_ai.scraper.linkedin_profile_parser import profile_from_linkedin_payload
from job_apply_ai.scraper.linkedin_profile_sync import apply_sync_action, compare_profiles, diff_summary
from job_apply_ai.storage.user_profile import normalize_profile


SAMPLE_PAYLOAD = {
    "url": "https://www.linkedin.com/in/jane-doe/",
    "sections": {
        "main_profile": (
            "Jane Doe\n\n"
            "Senior Python Developer | Backend Specialist\n\n"
            "Berlin, Germany\n\n"
            "About\n\n"
            "Backend engineer focused on APIs and data pipelines.\n\n"
            "Top skills\n\nPython • Django"
        ),
        "contact_info": (
            "Contact info\n\n"
            "linkedin.com/in/jane-doe\n\n"
            "Phone\n\n"
            "+49 123 4567890 (Mobile)\n\n"
            "Email\n\n"
            "jane@example.com"
        ),
        "experience": (
            "Experience\n\n"
            "Senior Developer\n\n"
            "Acme GmbH\n\n"
            "Jan 2021 - Present · 4 yrs\n\n"
            "Built REST APIs and led a small team.\n\n"
            "Junior Developer\n\n"
            "Beta Inc\n\n"
            "2018 - 2020 · 2 yrs\n\n"
            "Maintained legacy services."
        ),
        "skills": "Skills\n\nAll\n\nPython\n\nDjango\n\nPostgreSQL\n\nLeadership",
        "languages": "Languages\n\nEnglish\n\nGerman",
        "projects": "Projects\n\nNothing to see for now",
    },
}


class LinkedInProfileParserTests(unittest.TestCase):
    def test_profile_from_linkedin_payload_parses_core_fields(self):
        profile = profile_from_linkedin_payload(SAMPLE_PAYLOAD)
        self.assertEqual(profile["full_name"], "Jane Doe")
        self.assertIn("Python Developer", profile["professional_title"])
        self.assertEqual(profile["email"], "jane@example.com")
        self.assertIn("123", profile["phone"])
        self.assertIn("APIs", profile["personal_summary"])
        self.assertEqual(len(profile["work_experience"]), 2)
        self.assertEqual(profile["work_experience"][0]["role"], "Senior Developer")
        self.assertEqual(profile["work_experience"][0]["company"], "Acme GmbH")
        self.assertIn("Python", [item["name"] if isinstance(item, dict) else item for item in profile["technical_skills"]])

    def test_compare_profiles_finds_scalar_and_skill_diffs(self):
        local = normalize_profile(
            {
                "full_name": "Jane Doe",
                "professional_title": "Python Developer",
                "email": "jane@example.com",
                "technical_skills": ["Python", "Flask"],
                "languages": ["English"],
            }
        )
        linkedin = profile_from_linkedin_payload(SAMPLE_PAYLOAD)
        diffs = compare_profiles(local, linkedin)
        summary = diff_summary(diffs)
        self.assertGreater(summary["total"], 0)
        self.assertTrue(any(item["field"] == "technical_skills" and item["status"] == "linkedin_only" for item in diffs))
        self.assertTrue(any(item["field"] == "technical_skills" and item["status"] == "local_only" for item in diffs))

    def test_apply_sync_action_adds_linkedin_skill_to_profile(self):
        local = normalize_profile({"full_name": "Jane Doe", "technical_skills": ["Python"]})
        linkedin = profile_from_linkedin_payload(SAMPLE_PAYLOAD)
        diffs = compare_profiles(local, linkedin)
        django_diff = next(item for item in diffs if item.get("linkedin_display") == "Django")
        updated, result = apply_sync_action(local, linkedin, django_diff["id"], "add_to_profile")
        self.assertTrue(result["applied"])
        names = [
            item["name"] if isinstance(item, dict) else item for item in updated["technical_skills"]
        ]
        self.assertIn("Django", names)

    def test_apply_sync_action_remove_from_profile(self):
        local = normalize_profile({"full_name": "Jane Doe", "technical_skills": ["Python", "Flask"]})
        linkedin = profile_from_linkedin_payload(SAMPLE_PAYLOAD)
        diffs = compare_profiles(local, linkedin)
        flask_diff = next(item for item in diffs if item.get("local_display") == "Flask")
        updated, result = apply_sync_action(local, linkedin, flask_diff["id"], "remove_from_profile")
        self.assertTrue(result["applied"])
        names = [
            item["name"] if isinstance(item, dict) else item for item in updated["technical_skills"]
        ]
        self.assertNotIn("Flask", names)

    def test_apply_sync_action_linkedin_write_is_manual(self):
        local = normalize_profile({"full_name": "Jane Doe", "technical_skills": ["Flask"]})
        linkedin = profile_from_linkedin_payload(SAMPLE_PAYLOAD)
        diffs = compare_profiles(local, linkedin)
        flask_diff = next(item for item in diffs if item.get("local_display") == "Flask")
        updated, result = apply_sync_action(local, linkedin, flask_diff["id"], "add_to_linkedin")
        self.assertFalse(result["applied"])
        self.assertTrue(result["manual"])
        self.assertIn("edit_url", result)


if __name__ == "__main__":
    unittest.main()
