"""Send job application emails with CV and cover letter attachments."""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SMTP_PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "gmail": {"host": "smtp.gmail.com", "port": 587, "use_tls": True},
    "hotmail": {"host": "smtp-mail.outlook.com", "port": 587, "use_tls": True},
    "outlook": {"host": "smtp-mail.outlook.com", "port": 587, "use_tls": True},
}

SMTP_PROVIDER_LABELS = {
    "gmail": "Gmail",
    "hotmail": "Hotmail",
    "outlook": "Outlook",
    "custom": "Custom SMTP",
}


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    username: str
    password: str
    use_tls: bool
    from_email: str
    account_id: str = ""
    label: str = ""


def account_is_sendable(account: dict[str, Any]) -> bool:
    email = str(account.get("email") or "").strip()
    if not email:
        return False
    if str(account.get("auth_type") or "password").lower() == "oauth":
        return bool(str(account.get("oauth_refresh_token") or "").strip())
    return bool(str(account.get("password") or "").strip())


def account_display_label(account: dict[str, Any]) -> str:
    label = str(account.get("label") or "").strip()
    if label:
        return label
    provider = str(account.get("provider") or "custom")
    suffix = " (OAuth)" if account.get("auth_type") == "oauth" else ""
    return f"{SMTP_PROVIDER_LABELS.get(provider, provider.title())}{suffix}"


def resolve_smtp_account(account: dict[str, Any]) -> SmtpConfig | None:
    """Convert a password-based profile account into SMTP settings."""
    if str(account.get("auth_type") or "password").lower() == "oauth":
        return None

    email = str(account.get("email") or "").strip()
    password = str(account.get("password") or "").strip()
    if not email or not password:
        return None

    provider = str(account.get("provider") or "gmail").strip().lower()
    preset = SMTP_PROVIDER_PRESETS.get(provider, {})
    host = str(account.get("host") or preset.get("host") or "").strip()
    if not host:
        return None

    port = int(account.get("port") or preset.get("port") or 587)
    use_tls = bool(account.get("use_tls", preset.get("use_tls", True)))
    return SmtpConfig(
        host=host,
        port=port,
        username=email,
        password=password,
        use_tls=use_tls,
        from_email=email,
        account_id=str(account.get("id") or ""),
        label=account_display_label(account),
    )


def list_smtp_accounts(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return profile sending accounts suitable for UI pickers."""
    accounts = (profile or {}).get("smtp_accounts") or []
    options: list[dict[str, Any]] = []
    for account in accounts:
        if not account_is_sendable(account):
            continue
        options.append(
            {
                "id": str(account.get("id") or ""),
                "email": str(account.get("email") or ""),
                "label": account_display_label(account),
                "provider": account.get("provider", "custom"),
                "auth_type": account.get("auth_type", "password"),
                "is_default": bool(account.get("is_default")),
            }
        )
    return options


def get_send_account(
    profile: dict[str, Any] | None,
    account_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the full stored account dict selected for sending."""
    accounts = (profile or {}).get("smtp_accounts") or []
    if not accounts:
        return None

    if account_id:
        for account in accounts:
            if str(account.get("id")) == str(account_id) and account_is_sendable(account):
                return account
        return None

    for account in accounts:
        if account.get("is_default") and account_is_sendable(account):
            return account
    for account in accounts:
        if account_is_sendable(account):
            return account
    return None


def get_smtp_config(
    profile: dict[str, Any] | None,
    account_id: str | None = None,
) -> SmtpConfig | None:
    """Load SMTP settings from a password-based profile account."""
    account = get_send_account(profile, account_id)
    if not account:
        return None
    return resolve_smtp_account(account)


def smtp_is_configured(profile: dict[str, Any] | None = None) -> bool:
    return bool(list_smtp_accounts(profile))


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
    """Send application emails with document attachments via SMTP."""

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

        logger.info(
            "Sending application email from %s to %s",
            self.config.from_email,
            ", ".join(to_emails),
        )
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


def send_application_email(
    account: dict[str, Any],
    *,
    to_emails: list[str],
    subject: str,
    body: str,
    attachments: list[tuple[str, str]],
    google_settings: dict[str, str] | None = None,
    microsoft_settings: dict[str, str] | None = None,
) -> dict[str, str]:
    """Send using OAuth or SMTP depending on the stored account type."""
    auth_type = str(account.get("auth_type") or "password").lower()
    if auth_type == "oauth":
        provider = str(account.get("provider") or "").lower()
        if provider == "gmail":
            if not google_settings:
                raise RuntimeError("Google OAuth is not configured on the server.")
            from job_apply_ai.email.oauth_google import send_gmail_message

            return send_gmail_message(
                account,
                google_settings,
                to_emails=to_emails,
                subject=subject,
                body=body,
                attachments=attachments,
            )
        if provider in {"hotmail", "outlook"}:
            if not microsoft_settings:
                raise RuntimeError("Microsoft OAuth is not configured on the server.")
            from job_apply_ai.email.oauth_microsoft import send_microsoft_message

            return send_microsoft_message(
                account,
                microsoft_settings,
                to_emails=to_emails,
                subject=subject,
                body=body,
                attachments=attachments,
            )
        raise RuntimeError(f"OAuth sending is not supported for provider '{provider}'.")

    smtp_config = resolve_smtp_account(account)
    if not smtp_config:
        raise RuntimeError("Sending account is missing SMTP credentials.")
    ApplicationMailer(smtp_config).send(
        to_emails=to_emails,
        subject=subject,
        body=body,
        attachments=attachments,
    )
    return {}
