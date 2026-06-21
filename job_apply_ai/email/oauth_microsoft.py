"""Microsoft OAuth and Graph API sending."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import msal
import requests

from job_apply_ai.email.oauth_settings import MICROSOFT_SCOPES

logger = logging.getLogger(__name__)

GRAPH_SEND_MAIL_URL = "https://graph.microsoft.com/v1.0/me/sendMail"
GRAPH_ME_URL = "https://graph.microsoft.com/v1.0/me"


def build_microsoft_app(settings: dict[str, str]) -> msal.ConfidentialClientApplication:
    return msal.ConfidentialClientApplication(
        settings["client_id"],
        authority=settings["authority"],
        client_credential=settings["client_secret"],
    )


def microsoft_authorization_url(settings: dict[str, str], state: str) -> str:
    app = build_microsoft_app(settings)
    return app.get_authorization_request_url(
        scopes=MICROSOFT_SCOPES,
        state=state,
        redirect_uri=settings["redirect_uri"],
        prompt="consent",
    )


def exchange_microsoft_code(settings: dict[str, str], code: str) -> dict[str, Any]:
    app = build_microsoft_app(settings)
    result = app.acquire_token_by_authorization_code(
        code,
        scopes=MICROSOFT_SCOPES,
        redirect_uri=settings["redirect_uri"],
    )
    if "error" in result:
        raise RuntimeError(result.get("error_description") or result["error"])

    access_token = result.get("access_token", "")
    email = _fetch_microsoft_email(access_token)
    return {
        "email": email,
        "oauth_refresh_token": result.get("refresh_token", ""),
        "oauth_access_token": access_token,
        "oauth_expires_at": str(result.get("expires_in", "")),
    }


def _fetch_microsoft_email(access_token: str) -> str:
    response = requests.get(
        GRAPH_ME_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("mail") or payload.get("userPrincipalName") or "").strip()


def microsoft_access_token(account: dict[str, Any], settings: dict[str, str]) -> str:
    cached = str(account.get("oauth_access_token") or "").strip()
    if cached:
        probe = requests.get(
            GRAPH_ME_URL,
            headers={"Authorization": f"Bearer {cached}"},
            timeout=15,
        )
        if probe.ok:
            return cached

    refresh_token = str(account.get("oauth_refresh_token") or "").strip()
    if not refresh_token:
        raise RuntimeError("Microsoft account is missing a refresh token. Reconnect the account.")

    app = build_microsoft_app(settings)
    result = app.acquire_token_by_refresh_token(refresh_token, scopes=MICROSOFT_SCOPES)
    if "error" in result:
        raise RuntimeError(result.get("error_description") or result["error"])
    return str(result.get("access_token") or "")


def send_microsoft_message(
    account: dict[str, Any],
    settings: dict[str, str],
    *,
    to_emails: list[str],
    subject: str,
    body: str,
    attachments: list[tuple[str, str]],
) -> dict[str, str]:
    access_token = microsoft_access_token(account, settings)
    graph_attachments = []
    for filename, filepath in attachments:
        path = Path(filepath)
        graph_attachments.append(
            {
                "@odata.type": "#microsoft.graph.fileAttachment",
                "name": filename,
                "contentBytes": base64.b64encode(path.read_bytes()).decode(),
            }
        )

    payload = {
        "message": {
            "subject": subject,
            "body": {"contentType": "Text", "content": body},
            "toRecipients": [{"emailAddress": {"address": email}} for email in to_emails],
            "attachments": graph_attachments,
        },
        "saveToSentItems": True,
    }
    response = requests.post(
        GRAPH_SEND_MAIL_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return {"oauth_access_token": access_token}
