"""Persist tailored CV content and chat history alongside generated documents."""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Literal

DocumentKind = Literal["cv", "cover_letter"]

DEFAULT_STORE: dict[str, Any] = {
    "tailored_content": {},
    "chat_history": [],
    "cover_letter": {},
    "cover_letter_chat_history": [],
    "cv_chat_sessions": [],
    "cv_chat_active_session_id": "",
    "cover_letter_chat_sessions": [],
    "cover_letter_chat_active_session_id": "",
    "ats_analysis": {},
    "cv_preview_lines": [],
    "cv_preview_customized": False,
    "updated_at": "",
}

_SESSION_FIELDS: dict[DocumentKind, tuple[str, str, str]] = {
    "cv": ("chat_history", "cv_chat_sessions", "cv_chat_active_session_id"),
    "cover_letter": (
        "cover_letter_chat_history",
        "cover_letter_chat_sessions",
        "cover_letter_chat_active_session_id",
    ),
}


def cv_content_path(output_dir: str, cv_filename: str) -> str:
    """Return the sidecar JSON path for a generated CV file."""
    base, _ = os.path.splitext(cv_filename)
    return os.path.join(output_dir, f"{base}.content.json")


def _remove_document_pair(path: str) -> None:
    """Remove a generated document and its paired PDF export, if present."""
    if os.path.isfile(path):
        os.remove(path)
    base, _ = os.path.splitext(path)
    pdf_path = f"{base}.pdf"
    if os.path.isfile(pdf_path):
        os.remove(pdf_path)


def delete_cv_artifacts(
    output_dir: str,
    cv_filename: str = "",
    *,
    cover_letter_filename: str = "",
) -> None:
    """Remove generated CV, cover letter, and sidecar content files from disk."""
    if cv_filename:
        _remove_document_pair(os.path.join(output_dir, cv_filename))
        content_path = cv_content_path(output_dir, cv_filename)
        if os.path.isfile(content_path):
            os.remove(content_path)
    if cover_letter_filename:
        _remove_document_pair(os.path.join(output_dir, cover_letter_filename))


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _session_title(messages: list[dict[str, str]], fallback: str) -> str:
    for message in messages:
        if message.get("role") == "user":
            text = str(message.get("content", "")).strip()
            if text:
                return text[:60] + ("…" if len(text) > 60 else "")
    return fallback


def _new_session(*, title: str = "New session") -> dict[str, Any]:
    now = _utc_now()
    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def _ensure_chat_sessions(store: dict[str, Any], document: DocumentKind) -> None:
    """Migrate legacy flat history into session storage when needed."""
    legacy_key, sessions_key, active_key = _SESSION_FIELDS[document]
    sessions = store.get(sessions_key)
    if not isinstance(sessions, list):
        sessions = []
        store[sessions_key] = sessions

    legacy_messages = store.get(legacy_key) or []
    if legacy_messages and not sessions:
        session = _new_session(title=_session_title(legacy_messages, "Imported session"))
        session["messages"] = deepcopy(legacy_messages)
        sessions.append(session)
        store[active_key] = session["id"]
    elif sessions and not store.get(active_key):
        store[active_key] = sessions[-1]["id"]

    if not sessions:
        session = _new_session()
        sessions.append(session)
        store[active_key] = session["id"]


def normalize_store(data: dict[str, Any] | None) -> dict[str, Any]:
    """Return a store dict with chat sessions initialized."""
    store = data if isinstance(data, dict) else {}
    for key, default in DEFAULT_STORE.items():
        if key not in store:
            store[key] = deepcopy(default) if isinstance(default, (list, dict)) else default
    _ensure_chat_sessions(store, "cv")
    _ensure_chat_sessions(store, "cover_letter")
    return store


def get_chat_sessions(store: dict[str, Any], document: DocumentKind) -> list[dict[str, Any]]:
    """Return all chat sessions for a document type."""
    normalize_store(store)
    _, sessions_key, _ = _SESSION_FIELDS[document]
    return deepcopy(store.get(sessions_key, []))


def get_active_chat_session_id(store: dict[str, Any], document: DocumentKind) -> str:
    normalize_store(store)
    _, _, active_key = _SESSION_FIELDS[document]
    return str(store.get(active_key, "") or "")


def get_active_chat_messages(store: dict[str, Any], document: DocumentKind) -> list[dict[str, str]]:
    """Return messages for the active chat session."""
    normalize_store(store)
    _, sessions_key, active_key = _SESSION_FIELDS[document]
    active_id = store.get(active_key, "")
    for session in store.get(sessions_key, []):
        if session.get("id") == active_id:
            return deepcopy(session.get("messages", []))
    if store.get(sessions_key):
        return deepcopy(store[sessions_key][-1].get("messages", []))
    return []


def start_chat_session(store: dict[str, Any], document: DocumentKind) -> dict[str, Any]:
    """Create a new empty chat session and make it active."""
    normalize_store(store)
    legacy_key, sessions_key, active_key = _SESSION_FIELDS[document]
    session_number = len(store.get(sessions_key, [])) + 1
    session = _new_session(title=f"Session {session_number}")
    store[sessions_key].append(session)
    store[active_key] = session["id"]
    store[legacy_key] = []
    return deepcopy(session)


def set_active_chat_session(
    store: dict[str, Any],
    document: DocumentKind,
    session_id: str,
) -> bool:
    """Switch the active chat session when the id exists."""
    normalize_store(store)
    legacy_key, sessions_key, active_key = _SESSION_FIELDS[document]
    active_session = None
    for session in store.get(sessions_key, []):
        if session.get("id") == session_id:
            active_session = session
            break
    if active_session is None:
        return False
    store[active_key] = session_id
    store[legacy_key] = deepcopy(active_session.get("messages", []))
    return True


def append_active_chat_messages(
    store: dict[str, Any],
    document: DocumentKind,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Append messages to the active session and sync legacy history fields."""
    normalize_store(store)
    legacy_key, sessions_key, active_key = _SESSION_FIELDS[document]
    active_id = store.get(active_key, "")
    active_session = None
    for session in store.get(sessions_key, []):
        if session.get("id") == active_id:
            active_session = session
            break
    if active_session is None and store.get(sessions_key):
        active_session = store[sessions_key][-1]
        store[active_key] = active_session["id"]

    if active_session is None:
        start_chat_session(store, document)
        active_session = store[sessions_key][-1]

    active_session.setdefault("messages", []).extend(deepcopy(messages))
    active_session["updated_at"] = _utc_now()
    if active_session.get("title") in ("New session", f"Session {len(store.get(sessions_key, []))}"):
        active_session["title"] = _session_title(active_session["messages"], active_session["title"])

    store[legacy_key] = deepcopy(active_session.get("messages", []))
    return deepcopy(active_session.get("messages", []))


def save_cv_content(
    output_dir: str,
    cv_filename: str,
    tailored_content: dict[str, Any],
    *,
    chat_history: list[dict[str, str]] | None = None,
    cover_letter: dict[str, Any] | None = None,
    cover_letter_chat_history: list[dict[str, str]] | None = None,
    store: dict[str, Any] | None = None,
) -> str:
    """Save tailored CV content, cover letter, and chat histories to disk."""
    existing = normalize_store(load_cv_content(output_dir, cv_filename) or {})
    if store is not None:
        existing.update({k: deepcopy(v) for k, v in store.items()})
        normalize_store(existing)

    if chat_history is not None:
        existing["chat_history"] = deepcopy(chat_history)
        _, sessions_key, active_key = _SESSION_FIELDS["cv"]
        active_id = existing.get(active_key)
        updated = False
        for session in existing.get(sessions_key, []):
            if session.get("id") == active_id:
                session["messages"] = deepcopy(chat_history)
                session["updated_at"] = _utc_now()
                updated = True
                break
        if not updated and chat_history:
            session = _new_session(title=_session_title(chat_history, "Imported session"))
            session["messages"] = deepcopy(chat_history)
            existing[sessions_key].append(session)
            existing[active_key] = session["id"]

    if cover_letter_chat_history is not None:
        existing["cover_letter_chat_history"] = deepcopy(cover_letter_chat_history)
        _, sessions_key, active_key = _SESSION_FIELDS["cover_letter"]
        active_id = existing.get(active_key)
        updated = False
        for session in existing.get(sessions_key, []):
            if session.get("id") == active_id:
                session["messages"] = deepcopy(cover_letter_chat_history)
                session["updated_at"] = _utc_now()
                updated = True
                break
        if not updated and cover_letter_chat_history:
            session = _new_session(title=_session_title(cover_letter_chat_history, "Imported session"))
            session["messages"] = deepcopy(cover_letter_chat_history)
            existing[sessions_key].append(session)
            existing[active_key] = session["id"]

    payload = {
        "tailored_content": deepcopy(tailored_content),
        "chat_history": deepcopy(existing.get("chat_history", [])),
        "cover_letter": deepcopy(
            cover_letter if cover_letter is not None else existing.get("cover_letter", {})
        ),
        "cover_letter_chat_history": deepcopy(existing.get("cover_letter_chat_history", [])),
        "cv_chat_sessions": deepcopy(existing.get("cv_chat_sessions", [])),
        "cv_chat_active_session_id": existing.get("cv_chat_active_session_id", ""),
        "cover_letter_chat_sessions": deepcopy(existing.get("cover_letter_chat_sessions", [])),
        "cover_letter_chat_active_session_id": existing.get("cover_letter_chat_active_session_id", ""),
        "ats_analysis": deepcopy(existing.get("ats_analysis", {})),
        "cv_preview_lines": deepcopy(existing.get("cv_preview_lines", [])),
        "cv_preview_customized": bool(existing.get("cv_preview_customized")),
        "updated_at": _utc_now(),
    }
    path = cv_content_path(output_dir, cv_filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def load_cv_content(output_dir: str, cv_filename: str) -> dict[str, Any] | None:
    """Load stored CV content if the sidecar file exists."""
    path = cv_content_path(output_dir, cv_filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return normalize_store(data)


def append_chat_message(
    output_dir: str,
    cv_filename: str,
    role: str,
    content: str,
) -> None:
    """Append one chat message to the stored history."""
    store = load_cv_content(output_dir, cv_filename) or normalize_store({})
    append_active_chat_messages(store, "cv", [{"role": role, "content": content}])
    save_cv_content(
        output_dir,
        cv_filename,
        store.get("tailored_content", {}),
        store=store,
    )
