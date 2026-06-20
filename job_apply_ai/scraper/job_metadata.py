"""Parse structured job metadata from LinkedIn pages and descriptions."""

import re
from typing import Optional

VISA_YES_PATTERNS = [
    r"\bvisa sponsorship\b",
    r"\bsponsor(?:ing)?(?:\s+\w+){0,4}\s+visa\b",
    r"\btier\s*2\b",
    r"\bskilled worker visa\b",
    r"\bright to work (?:is )?not required\b",
    r"\bwe (?:can|will) sponsor\b",
]

VISA_NO_PATTERNS = [
    r"\bno visa sponsorship\b",
    r"\bnot (?:offering|providing|able to) sponsor\b",
    r"\bunable to (?:offer|provide) visa\b",
    r"\bmust (?:already )?have (?:the )?right to work\b",
    r"\bright to work (?:in|for)\b",
    r"\bno sponsorship\b",
]

RELOCATION_YES_PATTERNS = [
    r"\brelocation (?:package|assistance|support|allowance|bonus)\b",
    r"\brelocation (?:is )?(?:available|offered|provided)\b",
    r"\bwe (?:offer|provide) relocation\b",
    r"\brelocation benefits\b",
]

RELOCATION_NO_PATTERNS = [
    r"\bno relocation\b",
    r"\brelocation (?:is )?not (?:available|offered|provided)\b",
]

SALARY_PATTERNS = [
    r"£\s?\d[\d,]*(?:\.\d{2})?(?:\s*(?:/|per)\s*(?:yr|year|annum|hour|hr|month))?",
    r"£\s?\d[\d,]*(?:\.\d{2})?\s*[-–to]+\s*£?\s?\d[\d,]*(?:\.\d{2})?(?:\s*(?:/|per)\s*(?:yr|year|annum|hour|hr|month))?",
    r"\$\s?\d[\d,]*(?:\.\d{2})?(?:\s*(?:/|per)\s*(?:yr|year|annum|hour|hr|month))?",
    r"\$\s?\d[\d,]*(?:\.\d{2})?\s*[-–to]+\s*\$?\s?\d[\d,]*(?:\.\d{2})?(?:\s*(?:/|per)\s*(?:yr|year|annum|hour|hr|month))?",
    r"\b\d[\d,]*k\s*[-–to]+\s*\d[\d,]*k\b",
]

WORK_TYPE_PATTERNS = [
    (r"\bremote or hybrid\b|\bhybrid or remote\b", "Hybrid / Remote"),
    (r"\bfully remote\b|\b100%\s*remote\b|\bremote[- ]first\b", "Remote"),
    (r"\bhybrid\b", "Hybrid"),
    (r"\bremote\b", "Remote"),
    (r"\bon[- ]site\b|\bin[- ]office\b|\boffice[- ]based\b", "On-site"),
]


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _first_match(text: str, patterns: list[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0).strip()
    return None


def infer_work_type(title: str = "", location: str = "", description: str = "") -> str:
    """Infer workplace type from title, location, and description."""
    for source in (title, location):
        for pattern, label in WORK_TYPE_PATTERNS:
            if re.search(pattern, source, re.I):
                return label

    description_lower = description.lower()
    for pattern, label in WORK_TYPE_PATTERNS:
        if re.search(pattern, description_lower, re.I):
            return label

    if location and re.search(r"\bremote\b", location, re.I):
        return "Remote"

    return "Not specified"


def parse_visa_sponsorship(description: str) -> str:
    """Return Yes, No, or Not mentioned for visa sponsorship."""
    text = description.lower()
    if not text.strip():
        return "Not mentioned"
    if _match_any(text, VISA_NO_PATTERNS):
        return "No"
    if _match_any(text, VISA_YES_PATTERNS):
        return "Yes"
    return "Not mentioned"


def parse_relocation_support(description: str) -> str:
    """Return Yes, No, or Not mentioned for relocation support."""
    text = description.lower()
    if not text.strip():
        return "Not mentioned"
    if _match_any(text, RELOCATION_NO_PATTERNS):
        return "No"
    if _match_any(text, RELOCATION_YES_PATTERNS):
        return "Yes"
    if "relocation" in text:
        return "Mentioned"
    return "Not mentioned"


def extract_relocation_info(description: str) -> str:
    """Extract a short relocation-related snippet from the description."""
    if not description:
        return ""

    for sentence in re.split(r"(?<=[.!?])\s+", description):
        if re.search(r"relocation", sentence, re.I):
            return " ".join(sentence.split())[:240]

    return ""


def extract_salary(*texts: str) -> str:
    """Extract salary range from page text or description."""
    for text in texts:
        if not text:
            continue
        match = _first_match(text, SALARY_PATTERNS)
        if match:
            return match
    return ""


def empty_job_details() -> dict:
    """Default detail fields for a scraped job."""
    return {
        "work_type": "Not specified",
        "salary": "",
        "employment_type": "",
        "seniority_level": "",
        "job_function": "",
        "industry": "",
        "visa_sponsorship": "Not mentioned",
        "relocation_support": "Not mentioned",
        "relocation_info": "",
        "applicant_count": "",
        "listing_benefit": "",
        "company_url": "",
        "posted_date": "",
        "description": "",
        "emails": "",
        "application_method": "unknown",
    }
