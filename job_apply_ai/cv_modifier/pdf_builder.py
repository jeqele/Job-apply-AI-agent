"""Build PDF exports for tailored CVs and cover letters."""

from __future__ import annotations

import logging
import os
from typing import Any

from job_apply_ai.cv_modifier.chat_context import PREVIEW_ONLY_SECTION_LABELS, PreviewLine
from job_apply_ai.storage.user_profile import parse_professional_titles

logger = logging.getLogger(__name__)


def pdf_path_for_docx(docx_path: str) -> str:
    """Return the PDF path that pairs with a generated .docx file."""
    base, _ = os.path.splitext(docx_path)
    return f"{base}.pdf"


class CVPdfBuilder:
    """Render a CV PDF from numbered preview lines."""

    def build_from_preview_lines(
        self,
        output_path: str,
        preview_lines: list[PreviewLine],
        profile: dict[str, Any] | None = None,
        content: dict[str, Any] | None = None,
    ) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        name_style = ParagraphStyle(
            "CvName",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceAfter=4,
        )
        title_style = ParagraphStyle(
            "CvTitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            spaceAfter=6,
        )
        contact_style = ParagraphStyle(
            "CvContact",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor="#444444",
            spaceAfter=12,
        )
        section_style = ParagraphStyle(
            "CvSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=4,
            textTransform="uppercase",
        )
        body_style = ParagraphStyle(
            "CvBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=4,
        )
        bullet_style = ParagraphStyle(
            "CvBullet",
            parent=body_style,
            leftIndent=14,
            bulletIndent=0,
            spaceAfter=3,
        )
        role_style = ParagraphStyle(
            "CvRole",
            parent=body_style,
            fontName="Helvetica-Bold",
            spaceBefore=4,
        )
        meta_style = ParagraphStyle(
            "CvMeta",
            parent=body_style,
            fontSize=9,
            textColor="#555555",
            spaceAfter=2,
        )
        skills_style = ParagraphStyle(
            "CvSkills",
            parent=body_style,
            spaceAfter=6,
        )

        elements: list[Any] = []
        self._append_header(elements, profile, content, name_style, title_style, contact_style)

        skip_section = False
        title_used_in_header = bool(
            str((content or {}).get("professional_title", "") or "").strip()
        )
        for line in preview_lines:
            kind = str(line.get("kind", "text") or "text")
            text = str(line.get("text", "") or "").strip()
            if not text:
                continue

            if kind == "name":
                continue

            if kind == "section":
                skip_section = text.lower() in PREVIEW_ONLY_SECTION_LABELS
                if skip_section:
                    continue
                elements.append(Paragraph(self._escape(text), section_style))
                continue

            if skip_section:
                continue

            if kind == "title":
                if title_used_in_header:
                    title_used_in_header = False
                    continue

            escaped = self._escape(text)
            if kind == "bullet" or text.startswith("•"):
                bullet_text = text.lstrip("•").strip()
                elements.append(Paragraph(f"• {self._escape(bullet_text)}", bullet_style))
            elif kind == "role":
                elements.append(Paragraph(escaped, role_style))
            elif kind == "meta":
                elements.append(Paragraph(escaped, meta_style))
            elif kind == "skills":
                elements.append(Paragraph(escaped, skills_style))
            elif kind == "muted":
                elements.append(Paragraph(f"<i>{escaped}</i>", body_style))
            else:
                elements.append(Paragraph(escaped, body_style))

        doc.build(elements)
        logger.debug("Wrote CV PDF to %s", output_path)

    @staticmethod
    def _append_header(
        elements: list[Any],
        profile: dict[str, Any] | None,
        content: dict[str, Any] | None,
        name_style: Any,
        title_style: Any,
        contact_style: Any,
    ) -> None:
        from reportlab.platypus import Paragraph, Spacer

        profile = profile or {}
        content = content or {}
        name = str(profile.get("full_name", "") or "").strip()
        title = str(content.get("professional_title", "") or "").strip()
        if not title:
            titles = parse_professional_titles(profile.get("professional_title", ""))
            title = titles[0] if len(titles) == 1 else ""

        if name:
            elements.append(Paragraph(CVPdfBuilder._escape(name), name_style))
        if title:
            elements.append(Paragraph(CVPdfBuilder._escape(title), title_style))

        contact_parts = []
        if profile.get("email"):
            contact_parts.append(f"Email: {profile['email']}")
        if profile.get("github"):
            contact_parts.append(f"GitHub: {profile['github']}")
        if profile.get("phone"):
            contact_parts.append(f"Phone: {profile['phone']}")
        if profile.get("linkedin"):
            contact_parts.append(f"LinkedIn: {profile['linkedin']}")
        if contact_parts:
            elements.append(Paragraph(CVPdfBuilder._escape(" | ".join(contact_parts)), contact_style))
        else:
            elements.append(Spacer(1, 6))

    @staticmethod
    def _escape(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


class CoverLetterPdfBuilder:
    """Render a cover letter PDF from structured JSON content."""

    def build(self, output_path: str, content: dict[str, Any]) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
        )

        styles = getSampleStyleSheet()
        body_style = ParagraphStyle(
            "CoverBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            spaceAfter=10,
        )
        closing_style = ParagraphStyle(
            "CoverClosing",
            parent=body_style,
            spaceBefore=8,
            spaceAfter=4,
        )
        contact_style = ParagraphStyle(
            "CoverContact",
            parent=body_style,
            fontSize=9,
            textColor="#444444",
            spaceBefore=8,
        )

        elements: list[Any] = []
        if content.get("date"):
            elements.append(Paragraph(CVPdfBuilder._escape(str(content["date"])), body_style))
            elements.append(Spacer(1, 8))

        for key in ("recipient_name", "recipient_company"):
            value = str(content.get(key, "") or "").strip()
            if value:
                elements.append(Paragraph(CVPdfBuilder._escape(value), body_style))

        elements.append(Spacer(1, 12))
        if content.get("greeting"):
            elements.append(Paragraph(CVPdfBuilder._escape(str(content["greeting"])), body_style))

        for paragraph in content.get("body_paragraphs") or []:
            text = str(paragraph).strip()
            if text:
                elements.append(Paragraph(CVPdfBuilder._escape(text), body_style))

        elements.append(Spacer(1, 8))
        if content.get("closing"):
            elements.append(Paragraph(CVPdfBuilder._escape(str(content["closing"])), closing_style))
        if content.get("signature_name"):
            elements.append(Paragraph(CVPdfBuilder._escape(str(content["signature_name"])), body_style))

        contact_bits = []
        if content.get("candidate_email"):
            contact_bits.append(str(content["candidate_email"]))
        if content.get("candidate_phone"):
            contact_bits.append(str(content["candidate_phone"]))
        if contact_bits:
            elements.append(Paragraph(CVPdfBuilder._escape(" | ".join(contact_bits)), contact_style))

        doc.build(elements)
        logger.debug("Wrote cover letter PDF to %s", output_path)


def build_cv_pdf(
    docx_path: str,
    preview_lines: list[PreviewLine],
    profile: dict[str, Any] | None = None,
    content: dict[str, Any] | None = None,
) -> str:
    """Build a CV PDF alongside its .docx file and return the PDF path."""
    pdf_path = pdf_path_for_docx(docx_path)
    CVPdfBuilder().build_from_preview_lines(pdf_path, preview_lines, profile, content)
    return pdf_path


def build_cover_letter_pdf(docx_path: str, content: dict[str, Any]) -> str:
    """Build a cover letter PDF alongside its .docx file and return the PDF path."""
    pdf_path = pdf_path_for_docx(docx_path)
    CoverLetterPdfBuilder().build(pdf_path, content)
    return pdf_path
