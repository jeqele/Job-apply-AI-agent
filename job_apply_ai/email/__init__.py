"""Email delivery for job applications."""

from job_apply_ai.email.application_mailer import (
    ApplicationMailer,
    SmtpConfig,
    cover_letter_to_plain_text,
    load_smtp_config,
    parse_recipient_emails,
    smtp_is_configured,
)

__all__ = [
    "ApplicationMailer",
    "SmtpConfig",
    "cover_letter_to_plain_text",
    "load_smtp_config",
    "parse_recipient_emails",
    "smtp_is_configured",
]
