"""User profile storage for CV generation."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any

from job_apply_ai.storage.database import get_connection

DEFAULT_PROFILE: dict[str, Any] = {
    "full_name": "",
    "professional_title": "",
    "email": "",
    "github": "",
    "phone": "",
    "linkedin": "",
    "personal_summary": "",
    "technical_skills": [],
    "tools_platforms": [],
    "work_experience": [],
    "personal_projects": [],
    "soft_skills": [],
    "languages": [],
}


def normalize_profile(data: dict[str, Any] | None) -> dict[str, Any]:
    """Merge stored data with defaults and normalize list fields."""
    profile = deepcopy(DEFAULT_PROFILE)
    if not data:
        return profile

    for key in DEFAULT_PROFILE:
        if key not in data:
            continue
        value = data[key]
        if isinstance(DEFAULT_PROFILE[key], list):
            profile[key] = _normalize_string_list(value)
        else:
            profile[key] = str(value or "").strip()

    profile["work_experience"] = _normalize_experience_entries(data.get("work_experience", []))
    profile["personal_projects"] = _normalize_project_entries(data.get("personal_projects", []))
    return profile


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [part.strip() for part in value.replace("\n", ",").split(",")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        items = []
    return [item for item in items if item]


def parse_professional_titles(value: str) -> list[str]:
    """Split a comma-separated professional title field into distinct titles."""
    if not value or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def pick_professional_title(titles: list[str], job: dict[str, Any] | None = None) -> str:
    """Pick the best professional title when multiple comma-separated options exist."""
    if not titles:
        return ""
    if len(titles) == 1:
        return titles[0]
    if not job:
        return titles[0]

    job_title = str(job.get("title") or "").lower()
    job_desc = str(job.get("description") or "").lower()[:500]
    haystack = f"{job_title} {job_desc}"

    best = titles[0]
    best_score = -1
    for title in titles:
        title_lower = title.lower()
        score = 0
        if title_lower in haystack:
            score += 3
        for word in title_lower.split():
            if len(word) > 2 and word in haystack:
                score += 1
        if score > best_score:
            best_score = score
            best = title
    return best


def _normalize_experience_entries(entries: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(entries, list):
        return normalized

    for entry in entries:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                normalized.append({"role": text, "company": "", "period": "", "bullets": []})
            continue
        if not isinstance(entry, dict):
            continue
        bullets = _normalize_string_list(entry.get("bullets") or entry.get("highlights") or [])
        normalized.append(
            {
                "role": str(entry.get("role") or entry.get("title") or "").strip(),
                "company": str(entry.get("company") or "").strip(),
                "period": str(entry.get("period") or entry.get("dates") or "").strip(),
                "bullets": bullets,
            }
        )
    return normalized


def _normalize_project_entries(entries: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(entries, list):
        return normalized

    for entry in entries:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                normalized.append({"name": text, "description": "", "bullets": []})
            continue
        if not isinstance(entry, dict):
            continue
        bullets = _normalize_string_list(entry.get("bullets") or entry.get("highlights") or [])
        normalized.append(
            {
                "name": str(entry.get("name") or entry.get("title") or "").strip(),
                "description": str(entry.get("description") or "").strip(),
                "bullets": bullets,
            }
        )
    return normalized


def profile_to_text(profile: dict[str, Any]) -> str:
    """Serialize profile data into plain text for RAG indexing."""
    profile = normalize_profile(profile)
    sections: list[str] = []

    titles = parse_professional_titles(profile["professional_title"])
    if len(titles) > 1:
        title_line = f"Professional Titles (choose best fit for the job): {', '.join(titles)}"
    elif titles:
        title_line = f"Professional Title: {titles[0]}"
    else:
        title_line = ""

    header = [
        f"Name: {profile['full_name']}",
        title_line,
        f"Email: {profile['email']}",
        f"GitHub: {profile['github']}",
        f"Phone: {profile['phone']}",
        f"LinkedIn: {profile['linkedin']}",
    ]
    sections.append("\n".join(line for line in header if not line.endswith(": ")))

    if profile["personal_summary"]:
        sections.append(f"Personal Summary:\n{profile['personal_summary']}")

    if profile["technical_skills"]:
        sections.append("Technical Skills:\n" + ", ".join(profile["technical_skills"]))

    if profile["tools_platforms"]:
        sections.append("Tools & Platforms:\n" + ", ".join(profile["tools_platforms"]))

    if profile["work_experience"]:
        lines = ["Work Experience:"]
        for entry in profile["work_experience"]:
            header_parts = [part for part in [entry["role"], entry["company"], entry["period"]] if part]
            if header_parts:
                lines.append(" | ".join(header_parts))
            lines.extend(f"- {bullet}" for bullet in entry.get("bullets", []))
        sections.append("\n".join(lines))

    if profile["personal_projects"]:
        lines = ["Personal Projects:"]
        for entry in profile["personal_projects"]:
            title_parts = [part for part in [entry["name"], entry["description"]] if part]
            if title_parts:
                lines.append(" | ".join(title_parts))
            lines.extend(f"- {bullet}" for bullet in entry.get("bullets", []))
        sections.append("\n".join(lines))

    if profile["soft_skills"]:
        sections.append("Soft Skills:\n" + ", ".join(profile["soft_skills"]))

    if profile["languages"]:
        sections.append("Languages:\n" + ", ".join(profile["languages"]))

    return "\n\n".join(section for section in sections if section.strip())


def profile_is_ready(profile: dict[str, Any] | None) -> bool:
    """Return True when enough profile data exists to generate a CV."""
    profile = normalize_profile(profile)
    if not profile["full_name"]:
        return False

    has_content = bool(
        profile["personal_summary"]
        or profile["technical_skills"]
        or profile["tools_platforms"]
        or profile["work_experience"]
        or profile["personal_projects"]
        or profile["soft_skills"]
        or profile["languages"]
    )
    return has_content


def parse_multiline_list(text: str) -> list[str]:
    """Parse comma-separated or newline-separated list values."""
    if not text:
        return []
    items: list[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            value = part.strip()
            if value:
                items.append(value)
    return items


def parse_work_experience_text(text: str) -> list[dict[str, Any]]:
    """Parse work experience blocks from a simple text format."""
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current and (current.get("role") or current.get("bullets")):
                entries.append(current)
                current = None
            continue

        if line.startswith("-") or line.startswith("•"):
            if current is None:
                current = {"role": "", "company": "", "period": "", "bullets": []}
            current.setdefault("bullets", []).append(line.lstrip("-•").strip())
            continue

        if current and (current.get("role") or current.get("bullets")):
            entries.append(current)
        parts = [part.strip() for part in line.split("|")]
        current = {
            "role": parts[0] if len(parts) > 0 else "",
            "company": parts[1] if len(parts) > 1 else "",
            "period": parts[2] if len(parts) > 2 else "",
            "bullets": [],
        }

    if current and (current.get("role") or current.get("bullets")):
        entries.append(current)

    return _normalize_experience_entries(entries)


def parse_projects_text(text: str) -> list[dict[str, Any]]:
    """Parse personal project blocks from a simple text format."""
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current and (current.get("name") or current.get("bullets")):
                entries.append(current)
                current = None
            continue

        if line.startswith("-") or line.startswith("•"):
            if current is None:
                current = {"name": "", "description": "", "bullets": []}
            current.setdefault("bullets", []).append(line.lstrip("-•").strip())
            continue

        if current and (current.get("name") or current.get("bullets")):
            entries.append(current)
        parts = [part.strip() for part in line.split("|")]
        current = {
            "name": parts[0] if len(parts) > 0 else "",
            "description": parts[1] if len(parts) > 1 else "",
            "bullets": [],
        }

    if current and (current.get("name") or current.get("bullets")):
        entries.append(current)

    return _normalize_project_entries(entries)


def profile_to_form_fields(profile: dict[str, Any]) -> dict[str, str]:
    """Convert stored profile data into form-friendly strings."""
    profile = normalize_profile(profile)

    work_lines: list[str] = []
    for entry in profile["work_experience"]:
        header = " | ".join(part for part in [entry["role"], entry["company"], entry["period"]] if part)
        if header:
            work_lines.append(header)
        work_lines.extend(f"- {bullet}" for bullet in entry.get("bullets", []))
        work_lines.append("")

    project_lines: list[str] = []
    for entry in profile["personal_projects"]:
        header = " | ".join(part for part in [entry["name"], entry["description"]] if part)
        if header:
            project_lines.append(header)
        project_lines.extend(f"- {bullet}" for bullet in entry.get("bullets", []))
        project_lines.append("")

    return {
        "full_name": profile["full_name"],
        "professional_title": profile["professional_title"],
        "email": profile["email"],
        "github": profile["github"],
        "phone": profile["phone"],
        "linkedin": profile["linkedin"],
        "personal_summary": profile["personal_summary"],
        "technical_skills": "\n".join(profile["technical_skills"]),
        "tools_platforms": "\n".join(profile["tools_platforms"]),
        "soft_skills": "\n".join(profile["soft_skills"]),
        "languages": "\n".join(profile["languages"]),
        "work_experience_text": "\n".join(work_lines).strip(),
        "personal_projects_text": "\n".join(project_lines).strip(),
    }


def profile_from_form(form_data: dict[str, str]) -> dict[str, Any]:
    """Build a normalized profile dict from submitted form values."""
    return normalize_profile(
        {
            "full_name": form_data.get("full_name", ""),
            "professional_title": form_data.get("professional_title", ""),
            "email": form_data.get("email", ""),
            "github": form_data.get("github", ""),
            "phone": form_data.get("phone", ""),
            "linkedin": form_data.get("linkedin", ""),
            "personal_summary": form_data.get("personal_summary", ""),
            "technical_skills": parse_multiline_list(form_data.get("technical_skills", "")),
            "tools_platforms": parse_multiline_list(form_data.get("tools_platforms", "")),
            "soft_skills": parse_multiline_list(form_data.get("soft_skills", "")),
            "languages": parse_multiline_list(form_data.get("languages", "")),
            "work_experience": parse_work_experience_text(form_data.get("work_experience_text", "")),
            "personal_projects": parse_projects_text(form_data.get("personal_projects_text", "")),
        }
    )


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def merge_string_lists(existing: list[str], incoming: list[str]) -> tuple[list[str], list[str]]:
    """Append only new list items, ignoring case-insensitive duplicates."""
    seen = {_normalize_key(item) for item in existing if _normalize_key(item)}
    merged = list(existing)
    added: list[str] = []

    for item in incoming:
        key = _normalize_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
        added.append(item)

    return merged, added


def _experience_key(entry: dict[str, Any]) -> str:
    return "|".join(
        [
            _normalize_key(entry.get("role", "")),
            _normalize_key(entry.get("company", "")),
            _normalize_key(entry.get("period", "")),
        ]
    )


def _project_key(entry: dict[str, Any]) -> str:
    return "|".join(
        [
            _normalize_key(entry.get("name", "")),
            _normalize_key(entry.get("description", "")),
        ]
    )


def merge_experience_lists(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Merge work experience entries and deduplicate bullets."""
    merged = deepcopy(existing)
    added_entries: list[dict[str, Any]] = []
    added_bullets: list[str] = []
    index_by_key: dict[str, int] = {}

    for index, entry in enumerate(merged):
        loose_key = "|".join(
            [_normalize_key(entry.get("role", "")), _normalize_key(entry.get("company", ""))]
        )
        index_by_key[_experience_key(entry)] = index
        if loose_key:
            index_by_key.setdefault(loose_key, index)

    for entry in incoming:
        loose_key = "|".join(
            [_normalize_key(entry.get("role", "")), _normalize_key(entry.get("company", ""))]
        )
        match_index = index_by_key.get(_experience_key(entry))
        if match_index is None and loose_key:
            match_index = index_by_key.get(loose_key)

        if match_index is None:
            merged.append(deepcopy(entry))
            index_by_key[_experience_key(entry)] = len(merged) - 1
            if loose_key:
                index_by_key[loose_key] = len(merged) - 1
            added_entries.append(entry)
            continue

        target = merged[match_index]
        if not target.get("period") and entry.get("period"):
            target["period"] = entry["period"]
        bullets, new_bullets = merge_string_lists(
            target.get("bullets", []),
            entry.get("bullets", []),
        )
        target["bullets"] = bullets
        added_bullets.extend(new_bullets)

    return merged, added_entries, added_bullets


def merge_project_lists(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Merge personal projects and deduplicate bullets."""
    merged = deepcopy(existing)
    added_entries: list[dict[str, Any]] = []
    added_bullets: list[str] = []
    index_by_key: dict[str, int] = {}

    for index, entry in enumerate(merged):
        index_by_key[_project_key(entry)] = index
        name_key = _normalize_key(entry.get("name", ""))
        if name_key:
            index_by_key.setdefault(name_key, index)

    for entry in incoming:
        match_index = index_by_key.get(_project_key(entry))
        name_key = _normalize_key(entry.get("name", ""))
        if match_index is None and name_key:
            match_index = index_by_key.get(name_key)

        if match_index is None:
            merged.append(deepcopy(entry))
            index_by_key[_project_key(entry)] = len(merged) - 1
            if name_key:
                index_by_key[name_key] = len(merged) - 1
            added_entries.append(entry)
            continue

        target = merged[match_index]
        if not target.get("description") and entry.get("description"):
            target["description"] = entry["description"]
        bullets, new_bullets = merge_string_lists(
            target.get("bullets", []),
            entry.get("bullets", []),
        )
        target["bullets"] = bullets
        added_bullets.extend(new_bullets)

    return merged, added_entries, added_bullets


def merge_profiles(
    base: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge imported CV data into the current profile without duplicating existing data."""
    merged = normalize_profile(base)
    imported = normalize_profile(incoming)
    changes: dict[str, Any] = {
        "filled_fields": [],
        "added_technical_skills": [],
        "added_tools_platforms": [],
        "added_soft_skills": [],
        "added_languages": [],
        "added_work_experience": [],
        "added_personal_projects": [],
        "added_bullets": [],
    }

    scalar_fields = [
        "full_name",
        "professional_title",
        "email",
        "github",
        "phone",
        "linkedin",
        "personal_summary",
    ]
    for field in scalar_fields:
        if not merged[field] and imported[field]:
            merged[field] = imported[field]
            changes["filled_fields"].append(field)

    list_fields = [
        ("technical_skills", "added_technical_skills"),
        ("tools_platforms", "added_tools_platforms"),
        ("soft_skills", "added_soft_skills"),
        ("languages", "added_languages"),
    ]
    for field_name, change_key in list_fields:
        merged[field_name], added = merge_string_lists(merged[field_name], imported[field_name])
        if added:
            changes[change_key] = added

    merged["work_experience"], added_experience, added_exp_bullets = merge_experience_lists(
        merged["work_experience"],
        imported["work_experience"],
    )
    if added_experience:
        changes["added_work_experience"] = added_experience
    if added_exp_bullets:
        changes["added_bullets"].extend(added_exp_bullets)

    merged["personal_projects"], added_projects, added_proj_bullets = merge_project_lists(
        merged["personal_projects"],
        imported["personal_projects"],
    )
    if added_projects:
        changes["added_personal_projects"] = added_projects
    if added_proj_bullets:
        changes["added_bullets"].extend(added_proj_bullets)

    return merged, changes


def summarize_import_changes(changes: dict[str, Any]) -> list[str]:
    """Convert a merge diff into user-facing summary lines."""
    lines: list[str] = []
    field_labels = {
        "full_name": "Full name",
        "professional_title": "Professional title",
        "email": "Email",
        "github": "GitHub",
        "phone": "Phone",
        "linkedin": "LinkedIn",
        "personal_summary": "Personal summary",
    }

    for field in changes.get("filled_fields", []):
        lines.append(f"Filled {field_labels.get(field, field.replace('_', ' '))}")

    for skill in changes.get("added_technical_skills", []):
        lines.append(f"Added technical skill: {skill}")
    for tool in changes.get("added_tools_platforms", []):
        lines.append(f"Added tool/platform: {tool}")
    for skill in changes.get("added_soft_skills", []):
        lines.append(f"Added soft skill: {skill}")
    for language in changes.get("added_languages", []):
        lines.append(f"Added language: {language}")

    for entry in changes.get("added_work_experience", []):
        label = " | ".join(
            part for part in [entry.get("role"), entry.get("company"), entry.get("period")] if part
        )
        lines.append(f"Added work experience: {label or 'New role'}")

    for entry in changes.get("added_personal_projects", []):
        label = " | ".join(part for part in [entry.get("name"), entry.get("description")] if part)
        lines.append(f"Added personal project: {label or 'New project'}")

    bullet_count = len(changes.get("added_bullets", []))
    if bullet_count:
        lines.append(f"Added {bullet_count} new experience/project bullet(s)")

    return lines


def import_has_changes(changes: dict[str, Any]) -> bool:
    return bool(summarize_import_changes(changes))


def get_default_cv_template_path() -> str:
    """Return the bundled empty CV template path."""
    assets_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "assets",
        "cv_template.docx",
    )
    if os.path.exists(assets_path):
        return assets_path

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    root_template = os.path.join(project_root, "template.docx")
    if os.path.exists(root_template):
        return root_template

    return assets_path


class UserProfileRepository:
    """Persist a single user profile used for CV generation."""

    def get_profile(self) -> dict[str, Any]:
        with get_connection() as conn:
            row = conn.execute("SELECT data FROM user_profile WHERE id = 1").fetchone()
        if not row:
            return normalize_profile(None)
        try:
            data = json.loads(row["data"])
        except json.JSONDecodeError:
            data = {}
        return normalize_profile(data)

    def save_profile(self, data: dict[str, Any]) -> dict[str, Any]:
        profile = normalize_profile(data)
        payload = json.dumps(profile, ensure_ascii=False)
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_profile (id, data, updated_at)
                VALUES (1, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                    data = excluded.data,
                    updated_at = datetime('now')
                """,
                (payload,),
            )
        return profile

    def profile_exists(self) -> bool:
        return profile_is_ready(self.get_profile())
