"""Shared job and profile context for document chat editors."""

from __future__ import annotations

from copy import deepcopy
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


ALLOWED_PREVIEW_LINE_KINDS = frozenset({
    "name",
    "title",
    "section",
    "text",
    "role",
    "meta",
    "muted",
    "skills",
    "bullet",
})


def normalize_preview_line(line: Any) -> PreviewLine | None:
    """Return a sanitized preview line dict or None when invalid."""
    if not isinstance(line, dict):
        return None
    text = str(line.get("text", "") or "")
    kind = str(line.get("kind", "text") or "text").strip() or "text"
    if kind not in ALLOWED_PREVIEW_LINE_KINDS:
        kind = "text"
    normalized: PreviewLine = {"text": text, "kind": kind}
    variant = str(line.get("variant", "") or "").strip()
    if variant in {"matched", "missing"}:
        normalized["variant"] = variant
    return normalized


def normalize_preview_lines(lines: Any) -> list[PreviewLine]:
    """Normalize a list of preview lines from client or storage."""
    if not isinstance(lines, list):
        return []
    normalized: list[PreviewLine] = []
    for line in lines:
        item = normalize_preview_line(line)
        if item is not None:
            normalized.append(item)
    return normalized


def _preview_line_key(line: PreviewLine) -> tuple[str, str, str]:
    return (
        str(line.get("kind", "") or ""),
        str(line.get("text", "") or ""),
        str(line.get("variant", "") or ""),
    )


def resolve_cv_preview_lines(
    content: dict[str, Any],
    profile_name: str = "",
    stored_lines: list[PreviewLine] | None = None,
    *,
    customized: bool = False,
) -> list[PreviewLine]:
    """Return preview lines from storage or generated content."""
    generated = cv_content_to_preview_lines(content, profile_name)
    if customized and stored_lines is not None:
        return normalize_preview_lines(stored_lines)
    if not stored_lines:
        return generated
    normalized = normalize_preview_lines(stored_lines)
    if len(normalized) != len(generated):
        return generated
    if sorted(_preview_line_key(line) for line in normalized) != sorted(
        _preview_line_key(line) for line in generated
    ):
        return generated
    return normalized


def resolve_effective_tailored_content(
    content: dict[str, Any],
    profile_name: str = "",
    *,
    stored_lines: list[PreviewLine] | None = None,
    customized: bool = False,
) -> dict[str, Any]:
    """Return tailored content, folding in user-customized preview lines when active."""
    if not content:
        return {}
    preview_lines = resolve_cv_preview_lines(
        content,
        profile_name,
        stored_lines=stored_lines,
        customized=customized,
    )
    if customized and preview_lines:
        return preview_lines_to_content(content, preview_lines, profile_name)
    return deepcopy(content)


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


PREVIEW_ONLY_SECTION_LABELS = frozenset({
    "skills matching job",
    "skills matching job description",
    "job skills not in cv",
    "job-matched skills",
    "matched job skills",
    "skills in job and cv",
    "skills not in cv",
    "job requirements not in cv",
    "missing job skills",
})


def _parse_skills_line(text: str) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned or cleaned.lower() == "none":
        return []
    return [
        part.replace("•", "").strip()
        for part in cleaned.split(" • ")
        if part.replace("•", "").strip()
    ]


def _split_section_lines(preview_lines: list[PreviewLine], section_label: str) -> list[PreviewLine]:
    target = section_label.strip().lower()
    collecting = False
    collected: list[PreviewLine] = []
    for line in preview_lines:
        kind = str(line.get("kind", "") or "")
        text = str(line.get("text", "") or "").strip()
        if kind == "section":
            if collecting:
                break
            collecting = text.lower() == target
            continue
        if collecting:
            collected.append(line)
    return collected


def _parse_experience_or_project_lines(lines: list[PreviewLine]) -> list[Any]:
    entries: list[Any] = []
    current: dict[str, Any] | None = None
    for line in lines:
        kind = str(line.get("kind", "") or "")
        text = str(line.get("text", "") or "").strip()
        if not text or kind == "muted":
            continue
        if kind == "role":
            if current:
                entries.append(current)
            current = {"role": text, "company": "", "period": "", "bullets": []}
            continue
        if kind == "meta" and current is not None:
            parts = [part.strip() for part in text.split("·")]
            if len(parts) >= 2:
                current["company"] = parts[0]
                current["period"] = parts[1]
            elif parts:
                current["company"] = parts[0]
            continue
        if kind == "bullet" and current is not None:
            current["bullets"].append(text.lstrip("•").strip())
            continue
        if kind == "text":
            entries.append(text)
    if current:
        entries.append(current)
    return entries


def _parse_project_lines(lines: list[PreviewLine]) -> list[Any]:
    entries: list[Any] = []
    current: dict[str, Any] | None = None
    for line in lines:
        kind = str(line.get("kind", "") or "")
        text = str(line.get("text", "") or "").strip()
        if not text or kind == "muted":
            continue
        if kind == "role":
            if current:
                entries.append(current)
            current = {"name": text, "description": "", "bullets": []}
            continue
        if kind == "meta" and current is not None:
            current["description"] = text
            continue
        if kind == "bullet" and current is not None:
            current["bullets"].append(text.lstrip("•").strip())
            continue
        if kind == "text":
            entries.append(text)
    if current:
        entries.append(current)
    return entries


def preview_lines_to_content(
    content: dict[str, Any],
    preview_lines: list[PreviewLine],
    profile_name: str = "",
) -> dict[str, Any]:
    """Map numbered preview lines back into tailored CV content JSON."""
    updated = deepcopy(content)
    normalized = normalize_preview_lines(preview_lines)

    for line in normalized:
        if line.get("kind") == "title":
            title = str(line.get("text", "") or "").strip()
            if title:
                updated["professional_title"] = title
            break

    summary_lines = _split_section_lines(normalized, "Professional Summary")
    for line in summary_lines:
        if line.get("kind") in {"text", "muted"}:
            text = str(line.get("text", "") or "").strip()
            if text and text.lower() != "no summary yet.":
                updated["professional_summary"] = text
            break

    matched_lines = _split_section_lines(normalized, "Skills Matching Job")
    for line in matched_lines:
        if line.get("kind") == "skills":
            updated["job_matched_skills"] = _parse_skills_line(str(line.get("text", "")))
            break

    missing_lines = _split_section_lines(normalized, "Job Skills Not In CV")
    for line in missing_lines:
        if line.get("kind") == "skills":
            updated["job_skills_not_in_cv"] = _parse_skills_line(str(line.get("text", "")))
            break

    technical_lines = _split_section_lines(normalized, "Technical Skills")
    for line in technical_lines:
        if line.get("kind") == "skills":
            updated["technical_skills"] = _parse_skills_line(str(line.get("text", "")))
            break

    tools_lines = _split_section_lines(normalized, "Tools & Platforms")
    for line in tools_lines:
        if line.get("kind") == "skills":
            updated["tools_platforms"] = _parse_skills_line(str(line.get("text", "")))
            break

    experience_lines = _split_section_lines(normalized, "Experience Highlights")
    if experience_lines:
        updated["experience_highlights"] = _parse_experience_or_project_lines(experience_lines)

    project_lines = _split_section_lines(normalized, "Personal Projects")
    if project_lines:
        updated["personal_projects"] = _parse_project_lines(project_lines)

    soft_lines = _split_section_lines(normalized, "Soft Skills")
    for line in soft_lines:
        if line.get("kind") == "skills":
            updated["soft_skills"] = _parse_skills_line(str(line.get("text", "")))
            break

    language_lines = _split_section_lines(normalized, "Languages")
    for line in language_lines:
        if line.get("kind") == "skills":
            parsed = _parse_skills_line(str(line.get("text", "")))
            updated["languages"] = parsed or [str(line.get("text", "")).strip()]
            break

    return updated
