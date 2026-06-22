"""Shared job and profile context for document chat editors."""

from __future__ import annotations

from typing import Any

from job_apply_ai.storage.user_profile import profile_to_text

MAX_JOB_DESCRIPTION_CHARS = 8000

PreviewLine = dict[str, str]


def build_job_context(job: dict[str, Any]) -> str:
    """Serialize job fields including the full description for LLM prompts."""
    description = str(job.get("description", "") or "").strip()
    if len(description) > MAX_JOB_DESCRIPTION_CHARS:
        description = description[:MAX_JOB_DESCRIPTION_CHARS] + "\n… (description truncated)"

    parts = [
        f"Job title: {job.get('title', '')}",
        f"Company: {job.get('company', '')}",
        f"Location: {job.get('location', '')}",
        f"Work type: {job.get('work_type', '')}",
        f"Employment type: {job.get('employment_type', '')}",
        f"Seniority: {job.get('seniority_level', '')}",
        f"Industry: {job.get('industry', '')}",
        f"Visa sponsorship: {job.get('visa_sponsorship', '')}",
        f"Source: {job.get('source', '')}",
        f"Description:\n{description}",
    ]
    return "\n".join(part for part in parts if part and not part.endswith(": "))


def build_profile_context(profile: dict[str, Any]) -> str:
    """Serialize the full stored profile for LLM prompts."""
    return profile_to_text(profile).strip() or "No profile data available."


def _inline_bullets(items: list[Any] | None, *, empty_label: str = "None") -> str:
    cleaned = [str(item).strip() for item in (items or []) if str(item).strip()]
    if not cleaned:
        return empty_label
    return " • ".join(f"• {item}" for item in cleaned)


def cv_content_to_preview_lines(
    content: dict[str, Any],
    profile_name: str = "",
) -> list[PreviewLine]:
    """Build ordered preview lines shown in the CV panel (one line number each)."""
    lines: list[PreviewLine] = []

    def add(text: str, kind: str, **extra: str) -> None:
        lines.append({"text": text, "kind": kind, **extra})

    add(profile_name or "Your Name", "name")
    title = str(content.get("professional_title", "") or "").strip()
    if title:
        add(title, "title")

    def add_section(label: str) -> None:
        add(label, "section")

    def add_skill_section(label: str, items: list[Any] | None, variant: str = "") -> None:
        add_section(label)
        extra = {"variant": variant} if variant else {}
        add(_inline_bullets(items), "skills", **extra)

    add_section("Professional Summary")
    add(
        str(content.get("professional_summary", "") or "").strip() or "No summary yet.",
        "text",
    )
    add_skill_section("Skills Matching Job", content.get("job_matched_skills"), "matched")
    add_skill_section("Job Skills Not In CV", content.get("job_skills_not_in_cv"), "missing")
    technical = content.get("technical_skills") or content.get("key_skills")
    add_skill_section("Technical Skills", technical)
    add_skill_section("Tools & Platforms", content.get("tools_platforms"))
    add_section("Experience Highlights")
    experience = content.get("experience_highlights") or []
    if not experience:
        add("No experience highlights.", "muted")
    else:
        for entry in experience:
            if isinstance(entry, str):
                add(entry, "text")
                continue
            role = str(entry.get("role", "") or "").strip()
            company = str(entry.get("company", "") or "").strip()
            period = str(entry.get("period", "") or "").strip()
            if role:
                add(role, "role")
            meta_parts = [part for part in [company, period] if part]
            if meta_parts:
                add(" · ".join(meta_parts), "meta")
            bullets = entry.get("bullets") or []
            if not bullets:
                add("No bullets listed.", "muted")
            for bullet in bullets:
                bullet_text = str(bullet).strip().lstrip("•").strip()
                if bullet_text:
                    add(f"• {bullet_text}", "bullet")

    add_section("Personal Projects")
    projects = content.get("personal_projects") or []
    if not projects:
        add("No projects listed.", "muted")
    else:
        for entry in projects:
            if isinstance(entry, str):
                add(entry, "text")
                continue
            name = str(entry.get("name", "") or "").strip()
            description = str(entry.get("description", "") or "").strip()
            if name:
                add(name, "role")
            if description:
                add(description, "meta")
            bullets = entry.get("bullets") or []
            for bullet in bullets:
                bullet_text = str(bullet).strip().lstrip("•").strip()
                if bullet_text:
                    add(f"• {bullet_text}", "bullet")

    add_skill_section("Soft Skills", content.get("soft_skills"))
    add_skill_section("Languages", content.get("languages"))
    return lines


def format_numbered_cv_preview(lines: list[PreviewLine]) -> str:
    """Format preview lines with stable 1-based line numbers for LLM prompts."""
    if not lines:
        return ""
    width = max(len(str(len(lines))), 2)
    formatted: list[str] = []
    for index, line in enumerate(lines, start=1):
        text = str(line.get("text", "")).strip()
        formatted.append(f"{index:>{width}} | {text}")
    return "\n".join(formatted)
