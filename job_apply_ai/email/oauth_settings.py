"""OAuth app credentials loaded from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]

MICROSOFT_SCOPES = [
    "https://graph.microsoft.com/Mail.Send",
    "https://graph.microsoft.com/User.Read",
    "offline_access",
    "openid",
    "email",
]


def _env(name: str) -> str:
    load_dotenv()
    return os.environ.get(name, "").strip()


def google_oauth_settings(redirect_uri: str) -> dict[str, str] | None:
    client_id = _env("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = _env("GOOGLE_OAUTH_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }


def microsoft_oauth_settings(redirect_uri: str) -> dict[str, str] | None:
    client_id = _env("MICROSOFT_OAUTH_CLIENT_ID")
    client_secret = _env("MICROSOFT_OAUTH_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "authority": _env("MICROSOFT_OAUTH_AUTHORITY") or "https://login.microsoftonline.com/common",
    }


def google_oauth_configured() -> bool:
    return bool(_env("GOOGLE_OAUTH_CLIENT_ID") and _env("GOOGLE_OAUTH_CLIENT_SECRET"))


def microsoft_oauth_configured() -> bool:
    return bool(_env("MICROSOFT_OAUTH_CLIENT_ID") and _env("MICROSOFT_OAUTH_CLIENT_SECRET"))
