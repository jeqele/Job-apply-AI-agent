"""Tests for CV document builder job-skill sections."""

import os
import tempfile
import unittest

from job_apply_ai.cv_modifier.docx_builder import CVDocumentBuilder
from job_apply_ai.storage.user_profile import get_default_cv_template_path


class DocxBuilderJobSkillsTests(unittest.TestCase):
    def test_builds_job_skill_sections(self):
        template_path = get_default_cv_template_path()
        self.assertTrue(os.path.exists(template_path))

        content = {
            "professional_summary": "Experienced developer targeting this role.",
            "job_matched_skills": ["Python", "Android", "REST APIs"],
            "job_skills_not_in_cv": ["Kotlin", "GraphQL"],
            "technical_skills": ["Python", "Android", "Java", "Flask"],
            "tools_platforms": ["Git", "Docker"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "tailored_cv.docx")
            builder = CVDocumentBuilder(template_path)
            builder.build(output_path, content)

            from docx import Document

            doc = Document(output_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

            summary_idx = paragraphs.index("Personal Summary")
            matched_idx = paragraphs.index("Skills Matching Job Description")
            missing_idx = paragraphs.index("Job Skills Not In CV")
            technical_idx = paragraphs.index("Technical Skills")

            self.assertLess(summary_idx, matched_idx)
            self.assertLess(matched_idx, missing_idx)
            self.assertLess(missing_idx, technical_idx)
            self.assertIn("Python", paragraphs[matched_idx + 1])
            self.assertIn("Kotlin", paragraphs[missing_idx + 1])


if __name__ == "__main__":
    unittest.main()
