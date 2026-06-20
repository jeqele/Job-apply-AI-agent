"""RAG + Ollama powered CV generation pipeline."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable

from job_apply_ai.cv_modifier.docx_builder import CVDocumentBuilder
from job_apply_ai.cv_modifier.ollama_client import OllamaClient
from job_apply_ai.cv_modifier.rag_system import CVRAGSystem
from job_apply_ai.utils.helpers import extract_text_from_docx

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert recruiter and CV analyst. Extract only factual requirements "
    "from the job description. Do not invent requirements."
)

GENERATION_SYSTEM_PROMPT = (
    "You are a professional CV writer. Tailor CV content to the target role using ONLY "
    "facts present in the supplied CV excerpts. Never invent employers, dates, degrees, "
    "certifications, or achievements. Keep language concise, professional, and ATS-friendly. "
    "Return valid JSON only."
)


class RAGCVGenerator:
    """Generate a job-tailored CV using retrieval-augmented generation and Ollama."""

    def __init__(
        self,
        rag: CVRAGSystem | None = None,
        ollama: OllamaClient | None = None,
    ):
        self.rag = rag or CVRAGSystem(chunking_strategy="document_aware", chunk_size=300)
        self.ollama = ollama or OllamaClient()

    def prepare_cv_index(self, cv_template_path: str) -> int:
        """Index the uploaded CV once for batch generation."""
        cv_text = extract_text_from_docx(cv_template_path)
        if not cv_text or not cv_text.strip():
            raise ValueError("Could not extract text from the uploaded CV.")
        chunk_count = self.rag.index_cv(cv_text, source_name="user_cv")
        if chunk_count == 0:
            raise ValueError("The uploaded CV appears to be empty.")
        return chunk_count

    def generate_cv(
        self,
        cv_template_path: str,
        job: dict[str, Any],
        output_path: str,
        *,
        reindex: bool = True,
        on_progress: Callable[[str, str, int], None] | None = None,
    ) -> dict[str, Any]:
        def report(step: str, message: str, percent: int) -> None:
            if on_progress:
                on_progress(step, message, percent)

        report("starting", "Preparing AI CV generation…", 2)

        if not self.ollama.is_available():
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama locally and pull the configured models "
                f"({self.ollama.fast_model}, {self.ollama.main_model})."
            )

        report("validating_ollama", "Checking Ollama and installed models…", 8)
        resolved_models = self.ollama.validate_models()

        report("indexing_cv", "Indexing your CV with RAG…", 18)
        if reindex:
            chunk_count = self.prepare_cv_index(cv_template_path)
        elif not self.rag.documents:
            chunk_count = self.prepare_cv_index(cv_template_path)
        else:
            chunk_count = len(self.rag.documents)

        job_context = self._build_job_context(job)
        report("retrieving_context", "Retrieving relevant CV sections for this job…", 28)
        retrieved_chunks = self.rag.search(job_context, k=10)
        if not retrieved_chunks:
            raise ValueError("RAG retrieval returned no CV content for this job.")

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
        tailored_content = self._generate_tailored_cv(job, job_context, retrieved_chunks, analysis)

        report("building_document", "Building the tailored Word document…", 88)
        builder = CVDocumentBuilder(cv_template_path)
        builder.build(output_path, tailored_content)

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
Analyze the job and the candidate CV excerpts below.

JOB INFORMATION:
{job_context}

CANDIDATE CV EXCERPTS:
{cv_excerpt}

Return JSON with this exact shape:
{{
  "role_focus": "one sentence describing the target role",
  "priority_requirements": ["requirement 1", "requirement 2"],
  "matched_strengths": ["strength from CV that fits the role"],
  "gaps_or_risks": ["gap or risk if any"],
  "keywords_to_emphasize": ["keyword 1", "keyword 2"]
}}
"""
        return self.ollama.generate_json(
            prompt,
            model=self.ollama.fast_model,
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
    ) -> dict[str, Any]:
        cv_excerpt = self._format_retrieved_chunks(retrieved_chunks)
        prompt = f"""
Create a tailored CV content package for this application.

TARGET JOB:
Title: {job.get('title', '')}
Company: {job.get('company', '')}

JOB CONTEXT:
{job_context}

JOB ANALYSIS:
{analysis}

RELEVANT CV EXCERPTS (only use facts from here):
{cv_excerpt}

Instructions:
1. Rewrite the professional summary for this specific role and company tone.
2. Select and reorder the most relevant skills from the CV excerpts.
3. Reframe experience bullets to emphasize measurable impact aligned with the job.
4. Keep all claims truthful to the supplied CV excerpts.
5. Use strong action verbs and professional UK/US business English.
6. Limit the summary to 3-4 sentences.
7. Provide 8-14 key skills maximum.
8. Include 2-4 experience highlight groups with 2-4 bullets each.

Return JSON with this exact shape:
{{
  "professional_summary": "string",
  "key_skills": ["skill"],
  "experience_highlights": [
    {{
      "role": "string",
      "company": "string",
      "period": "string",
      "bullets": ["bullet"]
    }}
  ],
  "education": ["entry"],
  "additional_sections": {{
    "Certifications": ["optional entry"]
  }}
}}
"""
        content = self.ollama.generate_json(
            prompt,
            model=self.ollama.main_model,
            system=GENERATION_SYSTEM_PROMPT,
            temperature=0.25,
            max_attempts=3,
        )
        return self._normalize_generated_content(content)

    @staticmethod
    def _format_retrieved_chunks(chunks: list[dict[str, Any]]) -> str:
        formatted = []
        for index, chunk in enumerate(chunks, start=1):
            score = chunk.get("score", 0)
            text = chunk.get("original_text") or chunk.get("text", "")
            formatted.append(f"[Excerpt {index} | relevance={score:.3f}]\n{text}")
        return "\n\n".join(formatted)

    @staticmethod
    def _normalize_generated_content(content: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "professional_summary": str(content.get("professional_summary", "")).strip(),
            "key_skills": [str(skill).strip() for skill in content.get("key_skills", []) if str(skill).strip()],
            "experience_highlights": content.get("experience_highlights") or [],
            "education": content.get("education") or [],
            "additional_sections": content.get("additional_sections") or {},
        }
        if isinstance(normalized["education"], str):
            normalized["education"] = [normalized["education"]] if normalized["education"] else []
        return normalized


def batch_generate_cvs(
    cv_template_path: str,
    jobs: list[dict[str, Any]],
    output_dir: str,
) -> list[str]:
    """Generate tailored CVs for multiple jobs using one RAG index."""
    generator = RAGCVGenerator()
    generator.prepare_cv_index(cv_template_path)
    generated_paths: list[str] = []

    for job in jobs:
        today_date = datetime.today().strftime("%Y-%m-%d")
        company = str(job.get("company", "Company")).replace(" ", "_")
        title = str(job.get("title", "Role")).replace(" ", "_")
        output_path = os.path.join(output_dir, f"CV_{today_date}_{company}_{title}.docx")
        generator.generate_cv(cv_template_path, job, output_path, reindex=False)
        generated_paths.append(output_path)

    return generated_paths
