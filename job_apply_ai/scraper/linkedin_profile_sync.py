"""Compare HermesHire profile with LinkedIn and apply sync actions."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from job_apply_ai.storage.user_profile import (
    merge_experience_lists,
    merge_project_lists,
    merge_skill_items,
    merge_string_lists,
    normalize_profile,
    skill_item_name,
    skill_names,
)


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


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

SCALAR_FIELDS: tuple[tuple[str, str], ...] = (
    ("full_name", "Full name"),
    ("professional_title", "Headline / title"),
    ("email", "Email"),
    ("phone", "Phone"),
    ("linkedin", "LinkedIn URL"),
    ("personal_summary", "About / summary"),
)

SKILL_FIELDS: tuple[tuple[str, str], ...] = (
    ("technical_skills", "Technical skill"),
    ("soft_skills", "Soft skill"),
    ("languages", "Language"),
)

LINKEDIN_EDIT_URLS = {
    "full_name": "https://www.linkedin.com/in/me/edit/forms/intro/new/",
    "professional_title": "https://www.linkedin.com/in/me/edit/forms/intro/new/",
    "personal_summary": "https://www.linkedin.com/in/me/edit/forms/intro/new/",
    "email": "https://www.linkedin.com/in/me/edit/forms/contact-info/new/",
    "phone": "https://www.linkedin.com/in/me/edit/forms/contact-info/new/",
    "linkedin": "https://www.linkedin.com/in/me/edit/forms/contact-info/new/",
    "technical_skills": "https://www.linkedin.com/in/me/details/skills/edit/forms/new/",
    "soft_skills": "https://www.linkedin.com/in/me/details/skills/edit/forms/new/",
    "languages": "https://www.linkedin.com/in/me/details/languages/",
    "work_experience": "https://www.linkedin.com/in/me/details/experience/edit/forms/new/",
    "personal_projects": "https://www.linkedin.com/in/me/details/projects/edit/forms/new/",
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _normalize_key(value))
    return slug.strip("-") or "item"


def _scalar_diff_id(field: str) -> str:
    return f"scalar:{field}"


def _skill_diff_id(field: str, name: str) -> str:
    return f"skill:{field}:{_slug(name)}"


def _experience_diff_id(entry: dict[str, Any]) -> str:
    return f"experience:{_slug(entry.get('role', ''))}:{_slug(entry.get('company', ''))}"


def _project_diff_id(entry: dict[str, Any]) -> str:
    return f"project:{_slug(entry.get('name', ''))}"


def _format_experience(entry: dict[str, Any]) -> str:
    parts = [part for part in [entry.get("role"), entry.get("company"), entry.get("period")] if part]
    header = " | ".join(parts)
    bullets = entry.get("bullets") or []
    if bullets:
        return header + "\n" + "\n".join(f"- {bullet}" for bullet in bullets)
    return header


def _format_project(entry: dict[str, Any]) -> str:
    parts = [part for part in [entry.get("name"), entry.get("description")] if part]
    header = " | ".join(parts)
    bullets = entry.get("bullets") or []
    if bullets:
        return header + "\n" + "\n".join(f"- {bullet}" for bullet in bullets)
    return header


def _summary_similar(a: str, b: str) -> bool:
    left = _normalize_key(a)
    right = _normalize_key(b)
    if not left or not right:
        return left == right
    if left == right:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return shorter in longer and len(shorter) >= max(40, int(len(longer) * 0.7))


def _entries_by_key(entries: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        indexed[key_fn(entry)] = entry
        if key_fn is _experience_key:
            loose = "|".join(
                [_normalize_key(entry.get("role", "")), _normalize_key(entry.get("company", ""))]
            )
            indexed.setdefault(loose, entry)
    return indexed


def compare_profiles(
    local_profile: dict[str, Any] | None,
    linkedin_profile: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return diff rows between the stored profile and parsed LinkedIn profile."""
    local = normalize_profile(local_profile)
    linkedin = normalize_profile(linkedin_profile)
    diffs: list[dict[str, Any]] = []

    for field, label in SCALAR_FIELDS:
        local_value = str(local.get(field) or "").strip()
        linkedin_value = str(linkedin.get(field) or "").strip()
        if not local_value and not linkedin_value:
            continue
        if field == "personal_summary" and local_value and linkedin_value and _summary_similar(local_value, linkedin_value):
            continue
        if local_value == linkedin_value:
            continue
        status = "different"
        if local_value and not linkedin_value:
            status = "local_only"
        elif linkedin_value and not local_value:
            status = "linkedin_only"
        diffs.append(
            {
                "id": _scalar_diff_id(field),
                "category": "scalar",
                "field": field,
                "label": label,
                "status": status,
                "local": local_value,
                "linkedin": linkedin_value,
                "local_display": local_value,
                "linkedin_display": linkedin_value,
            }
        )

    for field, label in SKILL_FIELDS:
        local_names = {_normalize_key(name): name for name in skill_names(local.get(field, []))}
        linkedin_names = {_normalize_key(name): name for name in skill_names(linkedin.get(field, []))}
        all_keys = sorted(set(local_names) | set(linkedin_names))
        for key in all_keys:
            local_name = local_names.get(key, "")
            linkedin_name = linkedin_names.get(key, "")
            if local_name and linkedin_name:
                continue
            status = "different"
            if local_name and not linkedin_name:
                status = "local_only"
            elif linkedin_name and not local_name:
                status = "linkedin_only"
            diffs.append(
                {
                    "id": _skill_diff_id(field, linkedin_name or local_name),
                    "category": "skill",
                    "field": field,
                    "label": label,
                    "status": status,
                    "local": local_name,
                    "linkedin": linkedin_name,
                    "local_display": local_name,
                    "linkedin_display": linkedin_name,
                }
            )

    local_experience = _entries_by_key(local.get("work_experience", []), _experience_key)
    linkedin_experience = _entries_by_key(linkedin.get("work_experience", []), _experience_key)
    for key in sorted(set(local_experience) | set(linkedin_experience)):
        local_entry = local_experience.get(key)
        linkedin_entry = linkedin_experience.get(key)
        if local_entry and linkedin_entry:
            local_bullets = local_entry.get("bullets") or []
            linkedin_bullets = linkedin_entry.get("bullets") or []
            if _normalize_key(local_entry.get("period", "")) != _normalize_key(linkedin_entry.get("period", "")):
                diffs.append(
                    {
                        "id": f"{_experience_diff_id(local_entry)}:period",
                        "category": "experience_detail",
                        "field": "work_experience",
                        "label": "Experience period",
                        "status": "different",
                        "local": local_entry.get("period", ""),
                        "linkedin": linkedin_entry.get("period", ""),
                        "local_display": _format_experience(local_entry),
                        "linkedin_display": _format_experience(linkedin_entry),
                        "entry_key": key,
                    }
                )
            _, local_unique = merge_string_lists(linkedin_bullets, local_bullets)
            _, linkedin_unique = merge_string_lists(local_bullets, linkedin_bullets)
            if local_unique or linkedin_unique:
                diffs.append(
                    {
                        "id": f"{_experience_diff_id(local_entry or linkedin_entry)}:bullets",
                        "category": "experience_detail",
                        "field": "work_experience",
                        "label": "Experience description",
                        "status": "different" if local_unique and linkedin_unique else ("local_only" if local_unique else "linkedin_only"),
                        "local": "\n".join(local_unique) if local_unique else _format_experience(local_entry),
                        "linkedin": "\n".join(linkedin_unique) if linkedin_unique else _format_experience(linkedin_entry),
                        "local_display": _format_experience(local_entry),
                        "linkedin_display": _format_experience(linkedin_entry),
                        "entry_key": key,
                    }
                )
            continue

        status = "local_only" if local_entry else "linkedin_only"
        entry = local_entry or linkedin_entry or {}
        diffs.append(
            {
                "id": _experience_diff_id(entry),
                "category": "experience",
                "field": "work_experience",
                "label": "Work experience",
                "status": status,
                "local": _format_experience(local_entry) if local_entry else "",
                "linkedin": _format_experience(linkedin_entry) if linkedin_entry else "",
                "local_display": _format_experience(local_entry) if local_entry else "",
                "linkedin_display": _format_experience(linkedin_entry) if linkedin_entry else "",
                "entry_key": key,
                "entry": deepcopy(entry),
            }
        )

    local_projects = _entries_by_key(local.get("personal_projects", []), _project_key)
    linkedin_projects = _entries_by_key(linkedin.get("personal_projects", []), _project_key)
    for key in sorted(set(local_projects) | set(linkedin_projects)):
        local_entry = local_projects.get(key)
        linkedin_entry = linkedin_projects.get(key)
        if local_entry and linkedin_entry:
            continue
        status = "local_only" if local_entry else "linkedin_only"
        entry = local_entry or linkedin_entry or {}
        diffs.append(
            {
                "id": _project_diff_id(entry),
                "category": "project",
                "field": "personal_projects",
                "label": "Personal project",
                "status": status,
                "local": _format_project(local_entry) if local_entry else "",
                "linkedin": _format_project(linkedin_entry) if linkedin_entry else "",
                "local_display": _format_project(local_entry) if local_entry else "",
                "linkedin_display": _format_project(linkedin_entry) if linkedin_entry else "",
                "entry_key": key,
                "entry": deepcopy(entry),
            }
        )

    return diffs


def _linkedin_manual_action(diff: dict[str, Any], action: str, linkedin_url: str) -> dict[str, Any]:
    field = diff.get("field", "")
    edit_url = LINKEDIN_EDIT_URLS.get(field, linkedin_url or "https://www.linkedin.com/in/me/")
    if action == "add_to_linkedin":
        clipboard = diff.get("local") or diff.get("local_display") or ""
        message = "LinkedIn MCP is read-only. Copy this value and add it manually on LinkedIn."
    else:
        clipboard = diff.get("linkedin") or diff.get("linkedin_display") or ""
        message = "LinkedIn MCP is read-only. Remove this value manually on LinkedIn."
    return {
        "applied": False,
        "manual": True,
        "message": message,
        "edit_url": edit_url,
        "clipboard_text": clipboard,
    }


def apply_sync_action(
    local_profile: dict[str, Any] | None,
    linkedin_profile: dict[str, Any] | None,
    diff_id: str,
    action: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one sync action. Returns (updated_profile, result)."""
    if action not in {"add_to_profile", "remove_from_profile", "add_to_linkedin", "remove_from_linkedin"}:
        raise ValueError(f"Unsupported sync action: {action}")

    profile = normalize_profile(local_profile)
    linkedin = normalize_profile(linkedin_profile)
    diffs = compare_profiles(profile, linkedin)
    diff = next((item for item in diffs if item["id"] == diff_id), None)
    if not diff:
        raise ValueError("Diff item not found or already resolved.")

    linkedin_url = str(linkedin.get("_linkedin_url") or linkedin.get("linkedin") or "").strip()

    if action in {"add_to_linkedin", "remove_from_linkedin"}:
        return profile, _linkedin_manual_action(diff, action, linkedin_url)

    category = diff["category"]
    field = diff["field"]

    if action == "add_to_profile":
        if category == "scalar":
            profile[field] = diff["linkedin"]
        elif category == "skill":
            incoming = [{"name": diff["linkedin"], "familiarity": 70}]
            profile[field], _ = merge_skill_items(profile.get(field, []), incoming)
        elif category in {"experience", "experience_detail"}:
            entry = diff.get("entry") or _find_experience(linkedin, diff.get("entry_key", ""))
            if entry:
                profile["work_experience"], _, _ = merge_experience_lists(profile.get("work_experience", []), [entry])
            elif category == "experience_detail" and diff_id.endswith(":period"):
                _update_experience_period(profile, diff.get("entry_key", ""), diff["linkedin"])
            elif category == "experience_detail" and diff_id.endswith(":bullets"):
                _merge_experience_bullets(profile, diff.get("entry_key", ""), diff["linkedin"])
        elif category == "project":
            entry = diff.get("entry") or _find_project(linkedin, diff.get("entry_key", ""))
            if entry:
                profile["personal_projects"], _, _ = merge_project_lists(profile.get("personal_projects", []), [entry])
        return profile, {"applied": True, "manual": False, "message": "Added to your HermesHire profile."}

    if action == "remove_from_profile":
        if category == "scalar":
            profile[field] = ""
        elif category == "skill":
            key = _normalize_key(diff.get("local") or diff.get("local_display"))
            profile[field] = [
                item for item in profile.get(field, []) if _normalize_key(skill_item_name(item)) != key
            ]
        elif category in {"experience", "experience_detail"}:
            if category == "experience":
                profile["work_experience"] = [
                    entry
                    for entry in profile.get("work_experience", [])
                    if _experience_key(entry) != diff.get("entry_key")
                    and "|".join(
                        [_normalize_key(entry.get("role", "")), _normalize_key(entry.get("company", ""))]
                    )
                    != diff.get("entry_key")
                ]
            elif diff_id.endswith(":period"):
                _update_experience_period(profile, diff.get("entry_key", ""), "")
            elif diff_id.endswith(":bullets"):
                _remove_experience_bullets(profile, diff.get("entry_key", ""), diff.get("local", ""))
        elif category == "project":
            profile["personal_projects"] = [
                entry
                for entry in profile.get("personal_projects", [])
                if _project_key(entry) != diff.get("entry_key")
                and _normalize_key(entry.get("name", "")) != diff.get("entry_key")
            ]
        return profile, {"applied": True, "manual": False, "message": "Removed from your HermesHire profile."}

    raise ValueError(f"Unhandled action: {action}")


def _find_experience(profile: dict[str, Any], key: str) -> dict[str, Any] | None:
    for entry in profile.get("work_experience", []):
        if _experience_key(entry) == key:
            return entry
        loose = "|".join([_normalize_key(entry.get("role", "")), _normalize_key(entry.get("company", ""))])
        if loose == key:
            return entry
    return None


def _find_project(profile: dict[str, Any], key: str) -> dict[str, Any] | None:
    for entry in profile.get("personal_projects", []):
        if _project_key(entry) == key or _normalize_key(entry.get("name", "")) == key:
            return entry
    return None


def _update_experience_period(profile: dict[str, Any], key: str, period: str) -> None:
    for entry in profile.get("work_experience", []):
        loose = "|".join([_normalize_key(entry.get("role", "")), _normalize_key(entry.get("company", ""))])
        if _experience_key(entry) == key or loose == key:
            entry["period"] = period
            return


def _merge_experience_bullets(profile: dict[str, Any], key: str, linkedin_text: str) -> None:
    bullets = [line.strip("- ").strip() for line in str(linkedin_text).splitlines() if line.strip()]
    for entry in profile.get("work_experience", []):
        loose = "|".join([_normalize_key(entry.get("role", "")), _normalize_key(entry.get("company", ""))])
        if _experience_key(entry) == key or loose == key:
            merged, _ = merge_string_lists(entry.get("bullets", []), bullets)
            entry["bullets"] = merged
            return


def _remove_experience_bullets(profile: dict[str, Any], key: str, local_text: str) -> None:
    remove_keys = {_normalize_key(line.strip("- ").strip()) for line in str(local_text).splitlines() if line.strip()}
    for entry in profile.get("work_experience", []):
        loose = "|".join([_normalize_key(entry.get("role", "")), _normalize_key(entry.get("company", ""))])
        if _experience_key(entry) == key or loose == key:
            entry["bullets"] = [
                bullet for bullet in entry.get("bullets", []) if _normalize_key(bullet) not in remove_keys
            ]
            return


def diff_summary(diffs: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(diffs),
        "local_only": sum(1 for item in diffs if item["status"] == "local_only"),
        "linkedin_only": sum(1 for item in diffs if item["status"] == "linkedin_only"),
        "different": sum(1 for item in diffs if item["status"] == "different"),
    }
