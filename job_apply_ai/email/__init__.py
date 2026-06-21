"""Email delivery for job applications."""

from job_apply_ai.email.application_mailer import (
    ApplicationMailer,
    SmtpConfig,
    account_display_label,
    account_is_sendable,
    build_application_body,
    build_application_subject,
    cover_letter_to_plain_text,
    get_send_account,
    get_smtp_config,
    list_smtp_accounts,
    parse_recipient_emails,
    send_application_email,
    smtp_is_configured,
)

__all__ = [
    "ApplicationMailer",
    "SmtpConfig",
    "account_display_label",
    "account_is_sendable",
    "build_application_body",
    "build_application_subject",
    "cover_letter_to_plain_text",
    "get_send_account",
    "get_smtp_config",
    "list_smtp_accounts",
    "parse_recipient_emails",
    "send_application_email",
    "smtp_is_configured",
]
