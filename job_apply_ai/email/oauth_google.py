"""Google OAuth and Gmail API sending."""

from __future__ import annotations

import base64
import logging
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from job_apply_ai.email.oauth_settings import GOOGLE_SCOPES

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URI = "https://www.googleapis.com/oauth2/v2/userinfo"


def build_google_flow(settings: dict[str, str], state: str) -> Flow:
    client_config = {
        "web": {
            "client_id": settings["client_id"],
            "client_secret": settings["client_secret"],
            "auth_uri": GOOGLE_AUTH_URI,
            "token_uri": GOOGLE_TOKEN_URI,
            "redirect_uris": [settings["redirect_uri"]],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=GOOGLE_SCOPES, state=state)
    flow.redirect_uri = settings["redirect_uri"]
    return flow


def google_authorization_url(settings: dict[str, str], state: str) -> tuple[str, str]:
    """Return the Google consent URL and PKCE code verifier for the callback."""
    flow = build_google_flow(settings, state)
    url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return url, str(flow.code_verifier or "")


def exchange_google_code(
    settings: dict[str, str],
    code: str,
    state: str,
    *,
    code_verifier: str = "",
) -> dict[str, Any]:
    flow = build_google_flow(settings, state)
    if code_verifier:
        flow.code_verifier = code_verifier
    flow.fetch_token(code=code)
    credentials = flow.credentials
    email = _fetch_google_email(credentials.token)
    return {
        "email": email,
        "oauth_refresh_token": credentials.refresh_token or "",
        "oauth_access_token": credentials.token or "",
        "oauth_expires_at": credentials.expiry.isoformat() if credentials.expiry else "",
    }


def _fetch_google_email(access_token: str) -> str:
    response = requests.get(
        GOOGLE_USERINFO_URI,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    response.raise_for_status()
    return str(response.json().get("email", "")).strip()


def google_credentials_from_account(account: dict[str, Any], settings: dict[str, str]) -> Credentials:
    credentials = Credentials(
        token=account.get("oauth_access_token") or None,
        refresh_token=account.get("oauth_refresh_token") or None,
        token_uri=GOOGLE_TOKEN_URI,
        client_id=settings["client_id"],
        client_secret=settings["client_secret"],
        scopes=GOOGLE_SCOPES,
    )
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(GoogleAuthRequest())
    elif not credentials.valid and credentials.refresh_token:
        credentials.refresh(GoogleAuthRequest())
    return credentials


def send_gmail_message(
    account: dict[str, Any],
    settings: dict[str, str],
    *,
    to_emails: list[str],
    subject: str,
    body: str,
    attachments: list[tuple[str, str]],
) -> dict[str, str]:
    credentials = google_credentials_from_account(account, settings)
    message = _build_mime_message(account["email"], to_emails, subject, body, attachments)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {
        "oauth_access_token": credentials.token or "",
        "oauth_expires_at": credentials.expiry.isoformat() if credentials.expiry else "",
    }


def _build_mime_message(
    from_email: str,
    to_emails: list[str],
    subject: str,
    body: str,
    attachments: list[tuple[str, str]],
) -> MIMEMultipart:
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = ", ".join(to_emails)
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain", "utf-8"))

    for filename, filepath in attachments:
        path = Path(filepath)
        part = MIMEBase("application", "octet-stream")
        with path.open("rb") as handle:
            part.set_payload(handle.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
        message.attach(part)
    return message
