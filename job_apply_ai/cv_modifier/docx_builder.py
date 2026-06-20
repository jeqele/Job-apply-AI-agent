"""Build a tailored CV document from AI-generated structured content."""

from __future__ import annotations

import logging
import shutil
from typing import Any

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)

SECTION_KEYWORDS = {
    "summary": [
        "summary",
        "profile",
        "professional summary",
        "about me",
        "personal statement",
        "career summary",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core skills",
        "key skills",
        "competencies",
        "expertise",
        "qualifications",
    ],
    "experience": [
        "experience",
        "work experience",
        "employment history",
        "professional experience",
        "career history",
    ],
    "education": [
        "education",
        "academic background",
        "certifications",
    ],
}


class CVDocumentBuilder:
    """Apply structured CV content onto a Word template."""

    def __init__(self, template_path: str):
        self.template_path = template_path

    def build(self, output_path: str, content: dict[str, Any]) -> bool:
        shutil.copy2(self.template_path, output_path)
        doc = Document(output_path)

        updated = False
        if content.get("professional_summary"):
            updated |= self._replace_section(
                doc,
                SECTION_KEYWORDS["summary"],
                [content["professional_summary"]],
            )

        skills = content.get("key_skills") or []
        if skills:
            updated |= self._replace_section(
                doc,
                SECTION_KEYWORDS["skills"],
                [f"• {skill}" if not str(skill).startswith("•") else str(skill) for skill in skills],
            )

        experience_entries = content.get("experience_highlights") or []
        if experience_entries:
            updated |= self._replace_section(
                doc,
                SECTION_KEYWORDS["experience"],
                self._format_experience_entries(experience_entries),
            )

        education_lines = content.get("education") or []
        if isinstance(education_lines, str):
            education_lines = [education_lines]
        if education_lines:
            updated |= self._replace_section(doc, SECTION_KEYWORDS["education"], education_lines)

        additional = content.get("additional_sections") or {}
        for title, lines in additional.items():
            if not lines:
                continue
            if isinstance(lines, str):
                lines = [lines]
            updated |= self._replace_section(doc, [title.lower()], lines)

        if not updated:
            self._append_generated_sections(doc, content)

        doc.save(output_path)
        return True

    def _format_experience_entries(self, entries: list[Any]) -> list[str]:
        lines: list[str] = []
        for entry in entries:
            if isinstance(entry, str):
                lines.append(entry)
                continue
            role = entry.get("role") or entry.get("title") or "Role"
            company = entry.get("company") or ""
            period = entry.get("period") or entry.get("dates") or ""
            header_parts = [part for part in [role, company, period] if part]
            lines.append(" | ".join(header_parts))
            for bullet in entry.get("bullets") or entry.get("highlights") or []:
                bullet_text = str(bullet).strip()
                if bullet_text:
                    lines.append(f"• {bullet_text.lstrip('•').strip()}")
            lines.append("")
        return lines

    def _replace_section(self, doc: Document, keywords: list[str], lines: list[str]) -> bool:
        start_idx = self._find_section_start(doc, keywords)
        if start_idx is None:
            return False

        end_idx = self._find_section_end(doc, start_idx)
        self._remove_paragraph_range(doc, start_idx + 1, end_idx)

        anchor = doc.paragraphs[start_idx]
        for line in lines:
            if not str(line).strip():
                anchor = self._insert_paragraph_after(anchor, "")
                continue
            anchor = self._insert_paragraph_after(anchor, str(line).lstrip("•").strip())
            if str(line).startswith("•"):
                anchor.style = "List Bullet"
            elif "|" in str(line):
                for run in anchor.runs:
                    run.bold = True
        return True

    def _find_section_start(self, doc: Document, keywords: list[str]) -> int | None:
        for index, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.lower().strip()
            if not text:
                continue
            if any(keyword == text or text.startswith(keyword) for keyword in keywords):
                return index
            if paragraph.style.name.startswith("Heading") and any(keyword in text for keyword in keywords):
                return index
        return None

    def _find_section_end(self, doc: Document, start_idx: int) -> int:
        for index in range(start_idx + 1, len(doc.paragraphs)):
            paragraph = doc.paragraphs[index]
            if paragraph.style.name.startswith("Heading") and paragraph.text.strip():
                return index
        return len(doc.paragraphs)

    def _remove_paragraph_range(self, doc: Document, start_idx: int, end_idx: int) -> None:
        for index in reversed(range(start_idx, end_idx)):
            if index < len(doc.paragraphs):
                element = doc.paragraphs[index]._element
                element.getparent().remove(element)

    @staticmethod
    def _insert_paragraph_after(paragraph: Paragraph, text: str = "") -> Paragraph:
        new_p = OxmlElement("w:p")
        paragraph._p.addnext(new_p)
        new_para = Paragraph(new_p, paragraph._parent)
        if text:
            new_para.add_run(text)
        return new_para

    def _append_generated_sections(self, doc: Document, content: dict[str, Any]) -> None:
        doc.add_page_break()
        heading = doc.add_paragraph("Tailored Application Content")
        heading.style = "Heading 1"

        if content.get("professional_summary"):
            title = doc.add_paragraph("Professional Summary")
            title.style = "Heading 2"
            doc.add_paragraph(content["professional_summary"])

        skills = content.get("key_skills") or []
        if skills:
            title = doc.add_paragraph("Key Skills")
            title.style = "Heading 2"
            for skill in skills:
                paragraph = doc.add_paragraph(str(skill))
                paragraph.style = "List Bullet"

        experience_entries = content.get("experience_highlights") or []
        if experience_entries:
            title = doc.add_paragraph("Relevant Experience")
            title.style = "Heading 2"
            for line in self._format_experience_entries(experience_entries):
                if line.startswith("•"):
                    paragraph = doc.add_paragraph(line.lstrip("•").strip())
                    paragraph.style = "List Bullet"
                elif line.strip():
                    paragraph = doc.add_paragraph(line)
                    if "|" in line:
                        for run in paragraph.runs:
                            run.bold = True

        education_lines = content.get("education") or []
        if isinstance(education_lines, str):
            education_lines = [education_lines]
        if education_lines:
            title = doc.add_paragraph("Education")
            title.style = "Heading 2"
            for line in education_lines:
                doc.add_paragraph(str(line))
