"""Tests for CV document builder job-skill sections."""

import os
import tempfile
import unittest

from job_apply_ai.cv_modifier.docx_builder import CVDocumentBuilder
from job_apply_ai.storage.user_profile import get_default_cv_template_path


class DocxBuilderJobSkillsTests(unittest.TestCase):
    def test_export_omits_preview_only_job_skill_sections(self):
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

            self.assertNotIn("Skills Matching Job Description", paragraphs)
            self.assertNotIn("Job Skills Not In CV", paragraphs)

            technical_idx = paragraphs.index("Technical Skills")
            technical_line = paragraphs[technical_idx + 1]
            self.assertIn("Python", technical_line)
            self.assertIn("REST APIs", technical_line)
            self.assertIn("Flask", technical_line)
            self.assertNotIn("Kotlin", technical_line)
            self.assertNotIn("GraphQL", technical_line)

    def test_export_technical_skills_merges_job_matched_skills(self):
        merged = CVDocumentBuilder._export_technical_skills(
            {
                "job_matched_skills": ["REST APIs", "Python"],
                "technical_skills": ["Java", "Python", "Flask"],
            }
        )
        self.assertEqual(merged, ["REST APIs", "Python", "Java", "Flask"])


if __name__ == "__main__":
    unittest.main()
