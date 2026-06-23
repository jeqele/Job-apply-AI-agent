"""Generate tailored cover letters from job details and CV content."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from job_apply_ai.cv_modifier.ollama_client import OllamaClient, get_ollama_client

logger = logging.getLogger(__name__)

GENERATION_SYSTEM_PROMPT = (
    "You are an expert cover letter writer. Write professional, concise cover letters using ONLY "
    "facts supported by the candidate profile and tailored CV content. Never invent employers, "
    "dates, degrees, or achievements. Match the tone of the target company when possible. "
    "Return valid JSON only."
)


class CoverLetterGenerator:
    """Generate structured cover letter content with Ollama."""

    def __init__(self, ollama: OllamaClient | None = None):
        self.ollama = ollama or get_ollama_client()

    def generate(
        self,
        job: dict[str, Any],
        profile: dict[str, Any],
        tailored_cv_content: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.ollama.is_available():
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama locally to generate cover letters."
            )
        self.ollama.validate_models()

        job_context = self._build_job_context(job)
        cv_summary = self._summarize_cv(tailored_cv_content)
        prompt = f"""
Write a tailored cover letter for this job application.

CANDIDATE:
Name: {profile.get('full_name', '')}
Email: {profile.get('email', '')}
Phone: {profile.get('phone', '')}
LinkedIn: {profile.get('linkedin', '')}

TARGET JOB:
{job_context}

TAILORED CV HIGHLIGHTS:
{cv_summary}

Instructions:
1. Write 3-4 body paragraphs (150-280 words total).
2. Open with a strong hook tied to the role and company.
3. Connect the candidate's proven strengths to the job requirements.
4. Close with enthusiasm and a clear call to action.
5. Use UK/US professional business English.
6. Do not invent facts beyond the supplied profile and CV highlights.

Return JSON with this exact shape:
{{
  "date": "formatted date string",
  "recipient_name": "Hiring Manager or specific name if unknown use Hiring Manager",
  "recipient_company": "company name",
  "greeting": "Dear Hiring Manager,",
  "body_paragraphs": ["paragraph 1", "paragraph 2", "paragraph 3"],
  "closing": "Yours sincerely,",
  "signature_name": "candidate full name"
}}
"""
        result = self.ollama.generate_json(
            prompt,
            model=self.ollama.main_model,
            system=GENERATION_SYSTEM_PROMPT,
            temperature=0.3,
            max_attempts=2,
        )
        return self.normalize(result, profile, job)

    @staticmethod
    def normalize(
        content: dict[str, Any],
        profile: dict[str, Any],
        job: dict[str, Any],
    ) -> dict[str, Any]:
        paragraphs = content.get("body_paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        paragraphs = [str(p).strip() for p in paragraphs if str(p).strip()]

        full_name = str(profile.get("full_name", "")).strip()
        company = str(job.get("company", "")).strip() or str(content.get("recipient_company", "")).strip()

        return {
            "date": str(content.get("date", "")).strip() or datetime.today().strftime("%d %B %Y"),
            "recipient_name": str(content.get("recipient_name", "")).strip() or "Hiring Manager",
            "recipient_company": str(content.get("recipient_company", "")).strip() or company,
            "greeting": str(content.get("greeting", "")).strip() or "Dear Hiring Manager,",
            "body_paragraphs": paragraphs,
            "closing": str(content.get("closing", "")).strip() or "Yours sincerely,",
            "signature_name": str(content.get("signature_name", "")).strip() or full_name,
            "candidate_email": str(profile.get("email", "")).strip(),
            "candidate_phone": str(profile.get("phone", "")).strip(),
        }

    @staticmethod
    def _build_job_context(job: dict[str, Any]) -> str:
        parts = [
            f"Title: {job.get('title', '')}",
            f"Company: {job.get('company', '')}",
            f"Location: {job.get('location', '')}",
            f"Work type: {job.get('work_type', '')}",
            f"Employment type: {job.get('employment_type', '')}",
            f"Seniority: {job.get('seniority_level', '')}",
            f"Industry: {job.get('industry', '')}",
            f"Description:\n{job.get('description', '')}",
        ]
        return "\n".join(part for part in parts if part and not part.endswith(": "))

    @staticmethod
    def _summarize_cv(content: dict[str, Any]) -> str:
        lines = []
        if content.get("professional_title"):
            lines.append(f"Title: {content['professional_title']}")
        if content.get("professional_summary"):
            lines.append(f"Summary: {content['professional_summary']}")
        skills = content.get("technical_skills") or content.get("key_skills") or []
        if skills:
            lines.append("Technical skills: " + ", ".join(str(s) for s in skills[:12]))
        matched = content.get("job_matched_skills") or []
        if matched:
            lines.append("Job-matched skills: " + ", ".join(str(s) for s in matched[:10]))
        for entry in (content.get("experience_highlights") or [])[:3]:
            role = entry.get("role", "")
            company = entry.get("company", "")
            if role or company:
                lines.append(f"Experience: {role} at {company}")
        return "\n".join(lines) or "No CV highlights supplied."
