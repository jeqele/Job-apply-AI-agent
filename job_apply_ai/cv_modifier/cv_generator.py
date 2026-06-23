"""RAG + Ollama powered CV generation pipeline."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable

from job_apply_ai.cv_modifier.docx_builder import CVDocumentBuilder
from job_apply_ai.cv_modifier.llm_client import LLMClient, get_llm_client
from job_apply_ai.dev_logging import dev_llm_context
from job_apply_ai.cv_modifier.rag_system import CVRAGSystem
from job_apply_ai.storage.user_profile import (
    get_default_cv_template_path,
    parse_professional_titles,
    pick_professional_title,
    profile_to_text,
)
from job_apply_ai.utils.helpers import extract_text_from_docx

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert recruiter and CV analyst. Extract only factual requirements "
    "from the job description. Do not invent requirements."
)

GENERATION_SYSTEM_PROMPT = (
    "You are a professional CV writer. Tailor CV content to the target role using ONLY "
    "facts present in the supplied candidate profile excerpts. Never invent employers, dates, "
    "degrees, certifications, achievements, or skills. When the job description asks for a "
    "specific variant the candidate does not have, use the broader truthful term from the profile "
    "(e.g. profile has Android (Java) and job wants Android (Kotlin) -> list Android only). "
    "Do not add skills from the job description to technical_skills unless they are supported by "
    "the profile. Respect each skill's self-rated familiarity percentage: do not present "
    "low-familiarity skills as expert-level strengths, and prioritize higher-familiarity skills "
    "when ordering and emphasizing capabilities. Keep language concise, professional, and "
    "ATS-friendly. Return valid JSON only."
)


class RAGCVGenerator:
    """Generate a job-tailored CV using retrieval-augmented generation and Ollama."""

    def __init__(
        self,
        rag: CVRAGSystem | None = None,
        llm: LLMClient | None = None,
    ):
        self.rag = rag or CVRAGSystem(chunking_strategy="document_aware", chunk_size=300)
        self.llm = llm or get_llm_client()

    def prepare_profile_index(self, profile: dict[str, Any]) -> int:
        """Index stored profile data once for batch generation."""
        cv_text = profile_to_text(profile)
        if not cv_text or not cv_text.strip():
            raise ValueError("Your profile is empty. Add your details before generating a CV.")
        chunk_count = self.rag.index_cv(cv_text, source_name="user_profile")
        if chunk_count == 0:
            raise ValueError("Could not index your profile content.")
        return chunk_count

    def prepare_cv_index(self, cv_template_path: str) -> int:
        """Index an uploaded CV document (legacy fallback)."""
        cv_text = extract_text_from_docx(cv_template_path)
        if not cv_text or not cv_text.strip():
            raise ValueError("Could not extract text from the uploaded CV.")
        chunk_count = self.rag.index_cv(cv_text, source_name="user_cv")
        if chunk_count == 0:
            raise ValueError("The uploaded CV appears to be empty.")
        return chunk_count

    def generate_cv(
        self,
        job: dict[str, Any],
        output_path: str,
        *,
        profile: dict[str, Any] | None = None,
        cv_template_path: str | None = None,
        reindex: bool = True,
        on_progress: Callable[[str, str, int], None] | None = None,
    ) -> dict[str, Any]:
        def report(step: str, message: str, percent: int) -> None:
            if on_progress:
                on_progress(step, message, percent)

        if not profile and not cv_template_path:
            raise ValueError("A user profile or CV template path is required.")

        template_path = cv_template_path or get_default_cv_template_path()
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"CV template not found: {template_path}")

        report("starting", "Preparing AI CV generation…", 2)

        if not self.llm.is_available():
            raise RuntimeError(
                f"{self.llm.provider_label} is not reachable. Check your LLM settings "
                f"({self.llm.fast_model}, {self.llm.main_model})."
            )

        report("validating_ollama", f"Checking {self.llm.provider_label} and models…", 8)
        resolved_models = self.llm.validate_models()

        report("indexing_cv", "Indexing your profile with RAG…", 18)
        if reindex:
            chunk_count = (
                self.prepare_profile_index(profile)
                if profile
                else self.prepare_cv_index(cv_template_path or template_path)
            )
        elif not self.rag.documents:
            chunk_count = (
                self.prepare_profile_index(profile)
                if profile
                else self.prepare_cv_index(cv_template_path or template_path)
            )
        else:
            chunk_count = len(self.rag.documents)

        job_context = self._build_job_context(job)
        report("retrieving_context", "Retrieving relevant profile sections for this job…", 28)
        retrieved_chunks = self.rag.search(job_context, k=10)
        if not retrieved_chunks:
            raise ValueError("RAG retrieval returned no profile content for this job.")

        report(
            "analyzing_job",
            f"Analyzing job requirements with {resolved_models['fast']}…",
            45,
        )
        analysis = self._analyze_job(job_context, retrieved_chunks)

        report(
            "generating_content",
            f"Writing tailored CV content with {resolved_models['main']}…",
            70,
        )
        tailored_content = self._generate_tailored_cv(
            job, job_context, retrieved_chunks, analysis, profile
        )

        report("building_document", "Building the tailored Word document…", 88)
        builder = CVDocumentBuilder(template_path)
        builder.build(output_path, tailored_content, profile=profile)

        report("saving", "Finalizing your CV file…", 96)

        report("complete", "CV generated successfully", 100)

        return {
            "output_path": output_path,
            "chunk_count": chunk_count,
            "retrieved_chunks": retrieved_chunks,
            "analysis": analysis,
            "tailored_content": tailored_content,
            "models": resolved_models,
        }

    def _build_job_context(self, job: dict[str, Any]) -> str:
        parts = [
            f"Job title: {job.get('title', '')}",
            f"Company: {job.get('company', '')}",
            f"Location: {job.get('location', '')}",
            f"Work type: {job.get('work_type', '')}",
            f"Employment type: {job.get('employment_type', '')}",
            f"Seniority: {job.get('seniority_level', '')}",
            f"Industry: {job.get('industry', '')}",
            f"Visa sponsorship: {job.get('visa_sponsorship', '')}",
            f"Description:\n{job.get('description', '')}",
        ]
        return "\n".join(part for part in parts if part and not part.endswith(": "))

    def _analyze_job(self, job_context: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        cv_excerpt = self._format_retrieved_chunks(retrieved_chunks[:6])
        prompt = f"""
Analyze the job and the candidate profile excerpts below.

JOB INFORMATION:
{job_context}

CANDIDATE PROFILE EXCERPTS:
{cv_excerpt}

Return JSON with this exact shape:
{{
  "role_focus": "one sentence describing the target role",
  "priority_requirements": ["requirement 1", "requirement 2"],
  "matched_strengths": ["strength from profile that fits the role"],
  "gaps_or_risks": ["gap or risk if any"],
  "keywords_to_emphasize": ["keyword 1", "keyword 2"],
  "job_skills_in_cv": ["skill from job description that the candidate profile supports"],
  "job_skills_not_in_cv": ["skill from job description not supported by the profile"]
}}

Rules for job_skills_in_cv and job_skills_not_in_cv:
- Extract concrete skills, tools, and technologies from the job description only.
- A skill belongs in job_skills_in_cv only when the profile clearly supports it (exact match or a broader parent the profile proves, e.g. Android for Android Java vs Android Kotlin).
- A skill belongs in job_skills_not_in_cv when the job asks for it but the profile does not support it.
- When the profile lists familiarity percentages, treat low-familiarity overlaps as weaker evidence than high-familiarity matches.
- Use concise labels; generalize when only a parent skill is truthful (Android, not Kotlin).
"""
        with dev_llm_context(
            operation="cv_job_analysis",
            context={
                "rag_chunk_count": len(retrieved_chunks[:6]),
                "job_context_preview": job_context[:500],
            },
        ):
            return self.llm.generate_json(
                prompt,
                model=self.llm.fast_model,
                system=ANALYSIS_SYSTEM_PROMPT,
                temperature=0.1,
                max_attempts=2,
            )

    def _generate_tailored_cv(
        self,
        job: dict[str, Any],
        job_context: str,
        retrieved_chunks: list[dict[str, Any]],
        analysis: dict[str, Any],
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cv_excerpt = self._format_retrieved_chunks(retrieved_chunks)
        available_titles = parse_professional_titles((profile or {}).get("professional_title", ""))
        if len(available_titles) > 1:
            titles_block = (
                "\nAVAILABLE PROFESSIONAL TITLES (pick exactly one for this job):\n"
                + ", ".join(available_titles)
                + "\n"
            )
        elif available_titles:
            titles_block = f"\nPROFESSIONAL TITLE:\n{available_titles[0]}\n"
        else:
            titles_block = ""

        prompt = f"""
Create a tailored CV content package for this application.

TARGET JOB:
Title: {job.get('title', '')}
Company: {job.get('company', '')}

JOB CONTEXT:
{job_context}

JOB ANALYSIS:
{analysis}
{titles_block}
RELEVANT PROFILE EXCERPTS (only use facts from here):
{cv_excerpt}

Instructions:
1. Rewrite the personal summary for this specific role and company tone using job keywords the profile actually supports.
2. Populate job_matched_skills from the job description skills the profile supports (use analysis.job_skills_in_cv as a guide).
3. Populate job_skills_not_in_cv with job-required skills the profile does NOT support (use analysis.job_skills_not_in_cv as a guide). Do not add these to technical_skills.
4. Select and reorder technical_skills from the profile, prioritizing job_matched_skills and higher familiarity ratings. Never invent skills.
5. When a job skill differs in specificity from the profile, use the broader truthful label (e.g. Android instead of Kotlin when only Java is evidenced).
6. Select the most relevant tools and platforms for this role from the profile, favoring items with stronger familiarity.
7. Reframe work experience bullets to emphasize measurable impact aligned with the job, using only facts from the profile.
8. Include relevant personal projects when they strengthen the application.
9. Keep soft skills and languages truthful to the supplied profile excerpts and their familiarity levels.
10. Use strong action verbs and professional UK/US business English.
11. Limit the summary to 3-4 sentences.
12. Provide 8-14 technical skills maximum, ordered by relevance to this job.
13. Include 2-4 experience highlight groups with 2-4 bullets each when available.
14. Set professional_title to the single title that best matches the target job. When AVAILABLE PROFESSIONAL TITLES are listed, choose exactly one from that list; do not combine titles or invent new ones.

Return JSON with this exact shape:
{{
  "professional_title": "string",
  "professional_summary": "string",
  "job_matched_skills": ["skill from job description supported by profile"],
  "job_skills_not_in_cv": ["skill from job description not in profile"],
  "technical_skills": ["skill"],
  "tools_platforms": ["tool or platform"],
  "experience_highlights": [
    {{
      "role": "string",
      "company": "string",
      "period": "string",
      "bullets": ["bullet"]
    }}
  ],
  "personal_projects": [
    {{
      "name": "string",
      "description": "string",
      "bullets": ["bullet"]
    }}
  ],
  "soft_skills": ["skill"],
  "languages": ["language and level"]
}}
"""
        with dev_llm_context(
            operation="cv_content_generate",
            context={
                "job_title": job.get("title", ""),
                "job_company": job.get("company", ""),
                "rag_chunk_count": len(retrieved_chunks),
                "analysis_role_focus": analysis.get("role_focus", ""),
            },
        ):
            content = self.llm.generate_json(
                prompt,
                model=self.llm.main_model,
                system=GENERATION_SYSTEM_PROMPT,
                temperature=0.25,
                max_attempts=3,
            )
        return self._normalize_generated_content(content, analysis, profile, job)

    @staticmethod
    def _format_retrieved_chunks(chunks: list[dict[str, Any]]) -> str:
        formatted = []
        for index, chunk in enumerate(chunks, start=1):
            score = chunk.get("score", 0)
            text = chunk.get("original_text") or chunk.get("text", "")
            formatted.append(f"[Excerpt {index} | relevance={score:.3f}]\n{text}")
        return "\n\n".join(formatted)

    @staticmethod
    def _normalize_skill_list(items: Any) -> list[str]:
        if not items:
            return []
        if isinstance(items, str):
            items = [items]
        return [str(skill).strip() for skill in items if str(skill).strip()]

    @staticmethod
    def _normalize_generated_content(
        content: dict[str, Any],
        analysis: dict[str, Any] | None = None,
        profile: dict[str, Any] | None = None,
        job: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        available_titles = parse_professional_titles((profile or {}).get("professional_title", ""))
        professional_title = str(content.get("professional_title", "")).strip()
        if available_titles:
            matched = next(
                (title for title in available_titles if title.lower() == professional_title.lower()),
                None,
            )
            professional_title = matched or pick_professional_title(available_titles, job)
        elif not professional_title and profile:
            professional_title = str(profile.get("professional_title", "")).strip()

        job_matched_skills = RAGCVGenerator._normalize_skill_list(content.get("job_matched_skills"))
        if not job_matched_skills and analysis:
            job_matched_skills = RAGCVGenerator._normalize_skill_list(analysis.get("job_skills_in_cv"))

        job_skills_not_in_cv = RAGCVGenerator._normalize_skill_list(content.get("job_skills_not_in_cv"))
        if not job_skills_not_in_cv and analysis:
            job_skills_not_in_cv = RAGCVGenerator._normalize_skill_list(analysis.get("job_skills_not_in_cv"))

        normalized = {
            "professional_title": professional_title,
            "professional_summary": str(content.get("professional_summary", "")).strip(),
            "job_matched_skills": job_matched_skills,
            "job_skills_not_in_cv": job_skills_not_in_cv,
            "technical_skills": RAGCVGenerator._normalize_skill_list(
                content.get("technical_skills") or content.get("key_skills")
            ),
            "tools_platforms": RAGCVGenerator._normalize_skill_list(content.get("tools_platforms")),
            "experience_highlights": content.get("experience_highlights") or [],
            "personal_projects": content.get("personal_projects") or [],
            "soft_skills": RAGCVGenerator._normalize_skill_list(content.get("soft_skills")),
            "languages": content.get("languages") or [],
            "education": content.get("education") or [],
            "additional_sections": content.get("additional_sections") or {},
            "key_skills": [],
        }
        normalized["key_skills"] = normalized["technical_skills"]
        if isinstance(normalized["languages"], str):
            normalized["languages"] = [normalized["languages"]] if normalized["languages"] else []
        if isinstance(normalized["education"], str):
            normalized["education"] = [normalized["education"]] if normalized["education"] else []
        return normalized


def batch_generate_cvs(
    profile: dict[str, Any],
    jobs: list[dict[str, Any]],
    output_dir: str,
) -> list[str]:
    """Generate tailored CVs for multiple jobs using one RAG index."""
    generator = RAGCVGenerator()
    generator.prepare_profile_index(profile)
    generated_paths: list[str] = []

    for job in jobs:
        today_date = datetime.today().strftime("%Y-%m-%d")
        company = str(job.get("company", "Company")).replace(" ", "_")
        title = str(job.get("title", "Role")).replace(" ", "_")
        output_path = os.path.join(output_dir, f"CV_{today_date}_{company}_{title}.docx")
        generator.generate_cv(job, output_path, profile=profile, reindex=False)
        generated_paths.append(output_path)

    return generated_paths
