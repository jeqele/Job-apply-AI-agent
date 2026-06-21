"""Send job application emails with CV and cover letter attachments."""

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    username: str
    password: str
    use_tls: bool
    from_email: str


def load_smtp_config(profile: dict[str, Any] | None = None) -> SmtpConfig | None:
    """Load SMTP settings from environment variables."""
    load_dotenv()

    host = os.environ.get("SMTP_HOST", "").strip()
    username = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    if not host or not username or not password:
        return None

    port = int(os.environ.get("SMTP_PORT", "587"))
    use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")
    profile_email = str((profile or {}).get("email", "")).strip()
    from_email = os.environ.get("SMTP_FROM", "").strip() or profile_email or username
    return SmtpConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        use_tls=use_tls,
        from_email=from_email,
    )


def smtp_is_configured(profile: dict[str, Any] | None = None) -> bool:
    return load_smtp_config(profile) is not None


def parse_recipient_emails(job: dict[str, Any]) -> list[str]:
    """Return recipient emails from a job record."""
    raw = str(job.get("emails", "")).strip()
    if not raw:
        return []

    recipients: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        email = part.strip().lower().rstrip(".,;)")
        if "@" not in email or email in seen:
            continue
        seen.add(email)
        recipients.append(email)
    return recipients


def cover_letter_to_plain_text(cover_letter: dict[str, Any] | None) -> str:
    """Render structured cover letter content as plain text."""
    if not cover_letter:
        return ""

    lines: list[str] = []
    if cover_letter.get("date"):
        lines.append(str(cover_letter["date"]))
        lines.append("")
    if cover_letter.get("recipient_name"):
        lines.append(str(cover_letter["recipient_name"]))
    if cover_letter.get("recipient_company"):
        lines.append(str(cover_letter["recipient_company"]))
    if lines and lines[-1] != "":
        lines.append("")

    greeting = str(cover_letter.get("greeting") or "Dear Hiring Manager,")
    lines.append(greeting)
    lines.append("")

    for paragraph in cover_letter.get("body_paragraphs") or []:
        text = str(paragraph).strip()
        if text:
            lines.append(text)
            lines.append("")

    closing = str(cover_letter.get("closing") or "Yours sincerely,")
    lines.append(closing)
    signature = str(cover_letter.get("signature_name") or "").strip()
    if signature:
        lines.append(signature)
    return "\n".join(lines).strip()


def build_application_subject(job: dict[str, Any], profile: dict[str, Any]) -> str:
    title = str(job.get("title") or "Role").strip()
    name = str(profile.get("full_name") or "Candidate").strip()
    return f"Application for {title} — {name}"


def build_application_body(
    job: dict[str, Any],
    profile: dict[str, Any],
    cover_letter: dict[str, Any] | None,
) -> str:
    """Build the email body, preferring cover letter text when available."""
    letter_text = cover_letter_to_plain_text(cover_letter)
    if letter_text:
        return letter_text

    name = str(profile.get("full_name") or "Candidate").strip()
    title = str(job.get("title") or "the role").strip()
    company = str(job.get("company") or "your company").strip()
    return (
        f"Dear Hiring Manager,\n\n"
        f"Please find attached my CV for the {title} position at {company}.\n\n"
        f"I would welcome the opportunity to discuss how my experience aligns with your needs.\n\n"
        f"Kind regards,\n{name}"
    )


class ApplicationMailer:
    """Send application emails with document attachments."""

    def __init__(self, config: SmtpConfig):
        self.config = config

    def send(
        self,
        *,
        to_emails: list[str],
        subject: str,
        body: str,
        attachments: list[tuple[str, str]],
    ) -> None:
        if not to_emails:
            raise ValueError("At least one recipient email is required")

        message = MIMEMultipart()
        message["From"] = self.config.from_email
        message["To"] = ", ".join(to_emails)
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain", "utf-8"))

        for filename, filepath in attachments:
            path = Path(filepath)
            if not path.is_file():
                raise FileNotFoundError(f"Attachment not found: {filepath}")

            part = MIMEBase("application", "octet-stream")
            with path.open("rb") as handle:
                part.set_payload(handle.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
            message.attach(part)

        logger.info("Sending application email to %s", ", ".join(to_emails))
        if self.config.use_tls:
            with smtplib.SMTP(self.config.host, self.config.port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.config.username, self.config.password)
                server.sendmail(self.config.from_email, to_emails, message.as_string())
            return

        with smtplib.SMTP_SSL(self.config.host, self.config.port, timeout=30) as server:
            server.login(self.config.username, self.config.password)
            server.sendmail(self.config.from_email, to_emails, message.as_string())
