"""Tests for OAuth account helpers."""

from job_apply_ai.storage.user_profile import (
    normalize_profile,
    remove_smtp_account,
    set_default_smtp_account,
    upsert_oauth_smtp_account,
)
from job_apply_ai.email.application_mailer import (
    account_is_sendable,
    get_send_account,
    list_smtp_accounts,
    send_application_email,
)


def test_upsert_oauth_account_and_list_for_ui():
    profile = upsert_oauth_smtp_account(
        {},
        provider="gmail",
        email="user@gmail.com",
        oauth_refresh_token="refresh-token",
        oauth_access_token="access-token",
    )
    assert len(profile["smtp_accounts"]) == 1
    account = profile["smtp_accounts"][0]
    assert account["auth_type"] == "oauth"
    assert account["is_default"] is True
    assert account_is_sendable(account)

    options = list_smtp_accounts(profile)
    assert options[0]["email"] == "user@gmail.com"
    assert options[0]["auth_type"] == "oauth"


def test_get_send_account_prefers_default():
    profile = normalize_profile(
        {
            "smtp_accounts": [
                {
                    "id": "a",
                    "provider": "gmail",
                    "auth_type": "oauth",
                    "email": "a@gmail.com",
                    "oauth_refresh_token": "r1",
                    "is_default": False,
                },
                {
                    "id": "b",
                    "provider": "outlook",
                    "auth_type": "oauth",
                    "email": "b@outlook.com",
                    "oauth_refresh_token": "r2",
                    "is_default": True,
                },
            ]
        }
    )
    selected = get_send_account(profile)
    assert selected is not None
    assert selected["email"] == "b@outlook.com"


def test_remove_and_set_default_smtp_account():
    profile = upsert_oauth_smtp_account(
        {},
        provider="gmail",
        email="one@gmail.com",
        oauth_refresh_token="r1",
    )
    profile = upsert_oauth_smtp_account(
        profile,
        provider="outlook",
        email="two@outlook.com",
        oauth_refresh_token="r2",
    )
    second_id = profile["smtp_accounts"][1]["id"]
    profile = set_default_smtp_account(profile, second_id)
    assert profile["smtp_accounts"][1]["is_default"] is True

    profile = remove_smtp_account(profile, second_id)
    assert len(profile["smtp_accounts"]) == 1
    assert profile["smtp_accounts"][0]["email"] == "one@gmail.com"


def test_send_application_email_routes_to_smtp(monkeypatch, tmp_path):
    cv_path = tmp_path / "cv.docx"
    cv_path.write_bytes(b"cv")
    sent = {"called": False}

    class FakeMailer:
        def __init__(self, config):
            sent["from_email"] = config.from_email

        def send(self, **kwargs):
            sent["called"] = True
            sent["to"] = kwargs["to_emails"]

    monkeypatch.setattr("job_apply_ai.email.application_mailer.ApplicationMailer", FakeMailer)

    account = {
        "id": "smtp1",
        "provider": "gmail",
        "auth_type": "password",
        "email": "user@gmail.com",
        "password": "app-password",
    }
    send_application_email(
        account,
        to_emails=["hr@acme.com"],
        subject="Hello",
        body="Body",
        attachments=[("cv.docx", str(cv_path))],
    )
    assert sent["called"] is True
    assert sent["from_email"] == "user@gmail.com"


def test_google_authorization_url_returns_code_verifier():
    from job_apply_ai.email.oauth_google import google_authorization_url

    settings = {
        "client_id": "test-client",
        "client_secret": "test-secret",
        "redirect_uri": "http://127.0.0.1:5000/profile/oauth/google/callback",
    }
    url, code_verifier = google_authorization_url(settings, "test-state")
    assert "accounts.google.com" in url
    assert code_verifier


def test_exchange_google_code_uses_stored_code_verifier(monkeypatch):
    from job_apply_ai.email.oauth_google import exchange_google_code

    captured = {}

    class FakeFlow:
        code_verifier = ""

        def fetch_token(self, code):
            captured["code"] = code
            captured["code_verifier"] = self.code_verifier

        @property
        def credentials(self):
            class Creds:
                token = "access"
                refresh_token = "refresh"
                expiry = None

            return Creds()

    def fake_build_flow(settings, state):
        return FakeFlow()

    monkeypatch.setattr("job_apply_ai.email.oauth_google.build_google_flow", fake_build_flow)
    monkeypatch.setattr(
        "job_apply_ai.email.oauth_google._fetch_google_email",
        lambda token: "user@gmail.com",
    )

    result = exchange_google_code(
        {"client_id": "x", "client_secret": "y", "redirect_uri": "http://localhost"},
        "auth-code",
        "state",
        code_verifier="pkce-verifier",
    )
    assert captured["code_verifier"] == "pkce-verifier"
    assert result["email"] == "user@gmail.com"
    assert result["oauth_refresh_token"] == "refresh"
