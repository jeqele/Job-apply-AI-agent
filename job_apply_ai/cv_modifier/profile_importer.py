"""Extract structured profile data from uploaded CV documents."""

from __future__ import annotations

import logging
import re
from typing import Any

from job_apply_ai.cv_modifier.llm_client import LLMClient, get_llm_client
from job_apply_ai.storage.user_profile import normalize_profile
from job_apply_ai.utils.helpers import extract_text_from_docx

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = (
    "You are a CV parser. Extract only factual information present in the CV text. "
    "Do not invent employers, dates, skills, or contact details. Return valid JSON only."
)

SECTION_ALIASES = {
    "summary": {"personal summary", "professional summary", "profile", "about me", "summary"},
    "technical_skills": {"technical skills", "skills", "core skills", "key skills"},
    "tools": {"tool & platforms", "tools & platforms", "tools and platforms", "tools", "platforms"},
    "experience": {"work experience", "experience", "employment history", "professional experience"},
    "projects": {"personal projects", "projects", "portfolio"},
    "soft_skills": {"soft skills", "interpersonal skills"},
    "languages": {"languages", "language skills"},
}


class ProfileImporter:
    """Parse CV documents into profile-shaped data using Ollama with heuristic fallback."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm_client()

    def extract_from_docx(self, file_path: str) -> dict[str, Any]:
        cv_text = extract_text_from_docx(file_path)
        if not cv_text or not cv_text.strip():
            raise ValueError("Could not extract text from the uploaded CV.")
        return self.extract_from_text(cv_text)

    def extract_from_text(self, cv_text: str) -> dict[str, Any]:
        if self.llm.is_available():
            try:
                self.llm.validate_models()
                extracted = self._extract_with_llm(cv_text)
                if extracted.get("full_name") or extracted.get("work_experience") or extracted.get("technical_skills"):
                    return extracted
                logger.warning("LLM extraction returned little data; falling back to heuristic parser")
            except Exception as exc:
                logger.warning("LLM CV extraction failed, using heuristic parser: %s", exc)

        return self._extract_with_heuristics(cv_text)

    def _extract_with_llm(self, cv_text: str) -> dict[str, Any]:
        prompt = f"""
Extract structured profile data from this CV.

CV TEXT:
{cv_text}

Return JSON with this exact shape:
{{
  "full_name": "string",
  "professional_title": "string",
  "email": "string",
  "github": "string",
  "phone": "string",
  "linkedin": "string",
  "personal_summary": "string",
  "technical_skills": ["skill"],
  "tools_platforms": ["tool or platform"],
  "work_experience": [
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

Rules:
1. Use only facts explicitly present in the CV.
2. Split technical skills and tools/platforms into separate lists when possible.
3. Preserve bullet achievements under the correct role or project.
4. Use empty strings or empty lists when a field is missing.
"""
        content = self.llm.generate_json(
            prompt,
            model=self.llm.fast_model,
            system=EXTRACT_SYSTEM_PROMPT,
            temperature=0.1,
            max_attempts=2,
        )
        return normalize_profile(content)

    def _extract_with_heuristics(self, cv_text: str) -> dict[str, Any]:
        lines = [line.strip() for line in cv_text.splitlines() if line.strip()]
        profile: dict[str, Any] = normalize_profile(None)

        if not lines:
            return profile

        header = lines[0]
        if "\t" in header:
            name, title = [part.strip() for part in header.split("\t", 1)]
            profile["full_name"] = name
            profile["professional_title"] = title
        else:
            profile["full_name"] = header

        contact_blob = " ".join(lines[1:3])
        profile["email"] = _find_email(contact_blob)
        profile["phone"] = _find_phone(contact_blob)
        profile["github"] = _find_github(contact_blob)
        profile["linkedin"] = _find_linkedin(contact_blob)

        sections = _split_text_sections(cv_text)
        profile["personal_summary"] = " ".join(sections.get("summary", []))
        profile["technical_skills"] = _flatten_bullet_lines(sections.get("technical_skills", []))
        profile["tools_platforms"] = _flatten_bullet_lines(sections.get("tools", []))
        profile["soft_skills"] = _flatten_bullet_lines(sections.get("soft_skills", []))
        profile["languages"] = _flatten_bullet_lines(sections.get("languages", []))
        profile["work_experience"] = _parse_experience_blocks(sections.get("experience", []))
        profile["personal_projects"] = _parse_project_blocks(sections.get("projects", []))

        if not profile["tools_platforms"] and profile["technical_skills"]:
            profile["tools_platforms"], profile["technical_skills"] = _split_skills_and_tools(
                profile["technical_skills"]
            )

        return normalize_profile(profile)


def _find_email(text: str) -> str:
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return match.group(0) if match else ""


def _find_phone(text: str) -> str:
    match = re.search(r"(?:\+\d[\d\s().-]{7,}\d|\(\d{2,4}\)\s*\d[\d\s-]{6,}\d)", text)
    return match.group(0).strip() if match else ""


def _find_github(text: str) -> str:
    match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[\w.-]+", text, re.IGNORECASE)
    if match:
        return match.group(0)
    match = re.search(r"github\.com/[\w.-]+", text, re.IGNORECASE)
    return match.group(0) if match else ""


def _find_linkedin(text: str) -> str:
    match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/[\w./-]+", text, re.IGNORECASE)
    if match:
        return match.group(0)
    match = re.search(r"linkedin\.com/[\w./-]+", text, re.IGNORECASE)
    return match.group(0) if match else ""


def _split_text_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = "header"
    sections[current] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or set(line) == {"-"}:
            continue

        section_key = _match_section(line)
        if section_key:
            current = section_key
            sections.setdefault(current, [])
            continue

        sections.setdefault(current, []).append(line)

    return sections


def _match_section(line: str) -> str | None:
    normalized = re.sub(r"\s+", " ", line.strip().lower())
    for key, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return key
    return None


def _flatten_bullet_lines(lines: list[str]) -> list[str]:
    items: list[str] = []
    for line in lines:
        for part in re.split(r"(?<=[.!?])\s+(?=•)|(?<=\))\s+(?=•)", line):
            chunk = part.strip().lstrip("•").strip()
            if not chunk:
                continue
            if ":" in chunk and len(chunk.split(":", 1)[0]) < 30:
                label, values = chunk.split(":", 1)
                prefix = label.strip()
                for value in re.split(r",(?![^(]*\))", values):
                    value = value.strip()
                    if value:
                        items.append(f"{prefix}: {value}" if prefix else value)
            else:
                items.append(chunk)
    return items


def _split_skills_and_tools(skills: list[str]) -> tuple[list[str], list[str]]:
    tool_keywords = (
        "docker",
        "kubernetes",
        "git",
        "github",
        "kafka",
        "redis",
        "firebase",
        "aws",
        "azure",
        "jenkins",
        "postgresql",
        "mysql",
        "mongodb",
    )
    tools: list[str] = []
    technical: list[str] = []
    for skill in skills:
        lowered = skill.lower()
        if any(keyword in lowered for keyword in tool_keywords):
            tools.append(skill)
        else:
            technical.append(skill)
    return tools, technical


def _parse_experience_blocks(lines: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if line.startswith("•") or _looks_like_period(line):
            index += 1
            continue

        role, company = _split_role_company(line)
        entry = {"role": role, "company": company, "period": "", "bullets": []}
        index += 1

        while index < len(lines):
            next_line = lines[index]
            if _match_section(next_line):
                break
            if next_line.startswith("•"):
                entry["bullets"].append(next_line.lstrip("•").strip())
                index += 1
                continue
            if _looks_like_period(next_line) and not entry["period"]:
                entry["period"] = next_line
                index += 1
                continue
            if "—" in next_line or " - " in next_line or " – " in next_line:
                break
            index += 1

        if entry["role"] or entry["company"] or entry["bullets"]:
            entries.append(entry)

    return normalize_profile({"work_experience": entries})["work_experience"]


def _parse_project_blocks(lines: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("•"):
            index += 1
            continue
        if _match_section(line):
            break

        name, description = _split_role_company(line)
        entry = {"name": name, "description": description, "bullets": []}
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if _match_section(next_line):
                break
            if not next_line.startswith("•") and ("—" in next_line or " - " in next_line):
                break
            if next_line.startswith("•"):
                entry["bullets"].append(next_line.lstrip("•").strip())
                index += 1
                continue
            index += 1
        entries.append(entry)

    return normalize_profile({"personal_projects": entries})["personal_projects"]


def _split_role_company(line: str) -> tuple[str, str]:
    main = line.split("→", 1)[0].strip()
    for separator in (" — ", " - ", " – "):
        if separator in main:
            role, company = main.split(separator, 1)
            return role.strip(), company.strip()
    return main, ""


def _looks_like_period(line: str) -> bool:
    lowered = line.lower()
    return bool(
        re.search(r"\b(19|20)\d{2}\b", line)
        or "present" in lowered
        or "remote" in lowered
        or "on-site" in lowered
        or "|" in line
    )
