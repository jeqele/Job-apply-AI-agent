"""Tests for application email helpers."""

from job_apply_ai.email.application_mailer import (
    ApplicationMailer,
    SmtpConfig,
    build_application_body,
    build_application_subject,
    cover_letter_to_plain_text,
    parse_recipient_emails,
)


def test_parse_recipient_emails_splits_and_deduplicates():
    job = {"emails": "hr@acme.com, careers@acme.com, hr@acme.com"}
    assert parse_recipient_emails(job) == ["hr@acme.com", "careers@acme.com"]


def test_cover_letter_to_plain_text_renders_paragraphs():
    cover_letter = {
        "date": "22 June 2026",
        "recipient_name": "Hiring Manager",
        "recipient_company": "Acme Corp",
        "greeting": "Dear Hiring Manager,",
        "body_paragraphs": ["I am excited to apply.", "I bring Python experience."],
        "closing": "Yours sincerely,",
        "signature_name": "Jane Doe",
    }
    text = cover_letter_to_plain_text(cover_letter)
    assert "Dear Hiring Manager," in text
    assert "I am excited to apply." in text
    assert "Jane Doe" in text


def test_build_application_subject_uses_job_and_profile():
    subject = build_application_subject(
        {"title": "Backend Engineer"},
        {"full_name": "Jane Doe"},
    )
    assert subject == "Application for Backend Engineer — Jane Doe"


def test_build_application_body_falls_back_without_cover_letter():
    body = build_application_body(
        {"title": "Engineer", "company": "Acme"},
        {"full_name": "Jane Doe"},
        None,
    )
    assert "Jane Doe" in body
    assert "Acme" in body


def test_application_mailer_send_attaches_files(tmp_path, monkeypatch):
    cv_path = tmp_path / "cv.docx"
    cv_path.write_bytes(b"cv-content")
    sent: dict = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=30):
            sent["host"] = host
            sent["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def ehlo(self):
            return None

        def starttls(self):
            return None

        def login(self, username, password):
            sent["username"] = username
            sent["password"] = password

        def sendmail(self, from_addr, to_addrs, message):
            sent["from_addr"] = from_addr
            sent["to_addrs"] = to_addrs
            sent["message"] = message

    monkeypatch.setattr("job_apply_ai.email.application_mailer.smtplib.SMTP", FakeSMTP)

    config = SmtpConfig(
        host="smtp.example.com",
        port=587,
        username="user@example.com",
        password="secret",
        use_tls=True,
        from_email="user@example.com",
    )
    ApplicationMailer(config).send(
        to_emails=["hr@acme.com"],
        subject="Application",
        body="Hello",
        attachments=[("cv.docx", str(cv_path))],
    )

    assert sent["host"] == "smtp.example.com"
    assert sent["to_addrs"] == ["hr@acme.com"]
    assert "cv.docx" in sent["message"]
