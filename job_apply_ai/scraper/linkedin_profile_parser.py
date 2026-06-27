"""Parse linkedin-mcp-server get_my_profile payloads into HermesHire profile shape."""

from __future__ import annotations

import re
from typing import Any

from job_apply_ai.storage.user_profile import normalize_profile

LINKEDIN_PROFILE_SECTIONS = "experience,education,skills,languages,contact_info,projects"

_FOOTER_MARKERS = (
    "Profile language",
    "Who your viewers also viewed",
    "People you may know",
    "You might like",
    "Suggested for you",
    "Private to you",
    "Analytics",
    "Stand out to employers",
    "Try Premium",
)

_PERIOD_RE = re.compile(
    r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b|\b\d{4}\b|Present)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_LINKEDIN_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?", re.IGNORECASE)


def _strip_footer(text: str) -> str:
    if not text:
        return ""
    for marker in _FOOTER_MARKERS:
        index = text.find(marker)
        if index > 0:
            text = text[:index]
    return text.strip()


def _section_text(payload: dict[str, Any], section: str) -> str:
    sections = payload.get("sections") or {}
    if not isinstance(sections, dict):
        return ""
    return _strip_footer(str(sections.get(section) or ""))


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _looks_like_period(line: str) -> bool:
    lower = line.lower()
    if "·" in line and ("yr" in lower or "mo" in lower or "present" in lower):
        return True
    if _PERIOD_RE.search(line) and (" - " in line or " – " in line or " to " in lower):
        return True
    return False


def _parse_main_profile(text: str) -> dict[str, str]:
    cleaned = _strip_footer(text)
    lines = _lines(cleaned)
    result = {
        "full_name": "",
        "professional_title": "",
        "personal_summary": "",
        "location": "",
    }
    if not lines:
        return result

    index = 0
    while index < len(lines) and lines[index].lower().startswith("verify"):
        index += 1
    if index < len(lines):
        result["full_name"] = lines[index]
        index += 1

    while index < len(lines):
        line = lines[index]
        lower = line.lower()
        if lower in {"contact info", "open to", "add section", "enhance profile", "resources"}:
            index += 1
            continue
        if lower.startswith("open to work"):
            index += 1
            continue
        if not result["professional_title"] and len(line) > 8 and "|" in line:
            result["professional_title"] = line
            index += 1
            continue
        if not result["professional_title"] and len(line) > 12 and "developer" in lower:
            result["professional_title"] = line
            index += 1
            continue
        if not result["location"] and "·" in line and len(line) < 80:
            parts = [part.strip() for part in line.split("·") if part.strip()]
            if parts and not parts[0].lower().startswith("contact"):
                result["location"] = parts[0]
            index += 1
            continue
        break

    about_match = re.search(r"About\s*\n+(.*?)(?:\n\nTop skills|\n\nFeatured|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if about_match:
        summary = about_match.group(1).strip()
        summary = re.sub(r"\nTop skills.*", "", summary, flags=re.DOTALL).strip()
        result["personal_summary"] = summary
    return result


def _parse_contact_info(text: str) -> dict[str, str]:
    result = {"email": "", "phone": "", "linkedin": ""}
    for line in _lines(text):
        lower = line.lower()
        if lower.startswith("email"):
            continue
        if lower.startswith("phone"):
            continue
        if lower.startswith("address"):
            continue
        if lower.startswith("your profile"):
            continue
        email = _EMAIL_RE.search(line)
        if email and not result["email"]:
            result["email"] = email.group(0)
            continue
        phone = _PHONE_RE.search(line)
        if phone and not result["phone"]:
            result["phone"] = phone.group(0).strip()
            continue
        linkedin = _LINKEDIN_URL_RE.search(line)
        if linkedin and not result["linkedin"]:
            url = linkedin.group(0)
            if not url.startswith("http"):
                url = f"https://{url.lstrip('/')}"
            result["linkedin"] = url
    return result


def _parse_skills(text: str) -> list[str]:
    cleaned = _strip_footer(text)
    lines = _lines(cleaned)
    skills: list[str] = []
    skip_prefixes = (
        "skills",
        "all",
        "industry knowledge",
        "tools & technologies",
        "interpersonal skills",
        "languages",
        "other skills",
        "software engineers",
    )
    for line in lines:
        lower = line.lower()
        if lower in skip_prefixes or lower.startswith("add a skill"):
            continue
        if len(line) < 2 or len(line) > 80:
            continue
        skills.append(line)
    return skills


def _parse_languages(text: str) -> list[str]:
    cleaned = _strip_footer(text)
    lines = _lines(cleaned)
    languages: list[str] = []
    for line in lines:
        if line.lower() == "languages":
            continue
        if len(line) < 2 or len(line) > 40:
            continue
        languages.append(line)
    return languages


def _parse_experience(text: str) -> list[dict[str, Any]]:
    cleaned = _strip_footer(text)
    if not cleaned:
        return []

    lines = _lines(cleaned)
    if lines and lines[0].lower() == "experience":
        lines = lines[1:]

    entries: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        if index + 2 >= len(lines):
            break
        role = lines[index]
        company = lines[index + 1]
        period = lines[index + 2]
        if _looks_like_period(period) and not _looks_like_period(role):
            description: list[str] = []
            index += 3
            while index < len(lines):
                if (
                    index + 2 < len(lines)
                    and not _looks_like_period(lines[index])
                    and not _looks_like_period(lines[index + 1])
                    and _looks_like_period(lines[index + 2])
                ):
                    break
                description.append(lines[index])
                index += 1
            bullets = [line[2:].strip() for line in description if line.startswith("- ")]
            if not bullets and description:
                bullets = [" ".join(description).strip()]
            entries.append(
                {
                    "role": role,
                    "company": company,
                    "period": period,
                    "bullets": [bullet for bullet in bullets if bullet],
                }
            )
            continue
        index += 1
    return entries


def _parse_projects(text: str) -> list[dict[str, Any]]:
    cleaned = _strip_footer(text).lower()
    if "nothing to see" in cleaned or "add projects" in cleaned and "projects" in cleaned[:40]:
        return []
    entries: list[dict[str, Any]] = []
    body = _strip_footer(text)
    if body.lower().startswith("projects"):
        body = body.split("\n", 1)[-1].strip()
    for chunk in [part.strip() for part in re.split(r"\n{2,}", body) if part.strip()]:
        lines = _lines(chunk)
        if not lines:
            continue
        entries.append({"name": lines[0], "description": lines[1] if len(lines) > 1 else "", "bullets": lines[2:]})
    return entries


def profile_from_linkedin_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a get_my_profile MCP payload into a normalized HermesHire profile dict."""
    main = _parse_main_profile(_section_text(payload, "main_profile"))
    contact = _parse_contact_info(_section_text(payload, "contact_info"))
    skills = _parse_skills(_section_text(payload, "skills"))
    languages = _parse_languages(_section_text(payload, "languages"))
    experience = _parse_experience(_section_text(payload, "experience"))
    projects = _parse_projects(_section_text(payload, "projects"))

    technical_skills = skills
    soft_skills: list[str] = []
    interpersonal = {
        "time management",
        "teamwork",
        "communication",
        "leadership",
        "adaptive management",
        "team management",
    }
    for skill in skills:
        if skill.lower() in interpersonal:
            soft_skills.append(skill)

    profile = normalize_profile(
        {
            "full_name": main["full_name"],
            "professional_title": main["professional_title"],
            "personal_summary": main["personal_summary"],
            "email": contact["email"],
            "phone": contact["phone"],
            "linkedin": contact["linkedin"] or str(payload.get("url") or "").strip(),
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "languages": languages,
            "work_experience": experience,
            "personal_projects": projects,
        }
    )
    profile["_linkedin_url"] = str(payload.get("url") or profile.get("linkedin") or "").strip()
    profile["_raw_sections"] = payload.get("sections") or {}
    return profile


def fetch_linkedin_profile() -> dict[str, Any]:
    """Fetch and parse the authenticated user's LinkedIn profile via MCP."""
    from job_apply_ai.scraper.linkedin_mcp_client import call_linkedin_mcp_tool

    payload = call_linkedin_mcp_tool(
        "get_my_profile",
        {
            "sections": LINKEDIN_PROFILE_SECTIONS,
            "max_scrolls": 10,
        },
    )
    return profile_from_linkedin_payload(payload)
