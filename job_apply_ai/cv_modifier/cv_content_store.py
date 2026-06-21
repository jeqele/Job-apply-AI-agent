"""Persist tailored CV content and chat history alongside generated documents."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any

DEFAULT_STORE: dict[str, Any] = {
    "tailored_content": {},
    "chat_history": [],
    "cover_letter": {},
    "cover_letter_chat_history": [],
    "updated_at": "",
}


def cv_content_path(output_dir: str, cv_filename: str) -> str:
    """Return the sidecar JSON path for a generated CV file."""
    base, _ = os.path.splitext(cv_filename)
    return os.path.join(output_dir, f"{base}.content.json")


def save_cv_content(
    output_dir: str,
    cv_filename: str,
    tailored_content: dict[str, Any],
    *,
    chat_history: list[dict[str, str]] | None = None,
    cover_letter: dict[str, Any] | None = None,
    cover_letter_chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Save tailored CV content, cover letter, and chat histories to disk."""
    existing = load_cv_content(output_dir, cv_filename) or deepcopy(DEFAULT_STORE)
    payload = {
        "tailored_content": deepcopy(tailored_content),
        "chat_history": deepcopy(
            chat_history if chat_history is not None else existing.get("chat_history", [])
        ),
        "cover_letter": deepcopy(
            cover_letter if cover_letter is not None else existing.get("cover_letter", {})
        ),
        "cover_letter_chat_history": deepcopy(
            cover_letter_chat_history
            if cover_letter_chat_history is not None
            else existing.get("cover_letter_chat_history", [])
        ),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
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
    store = deepcopy(DEFAULT_STORE)
    store.update(data)
    return store


def append_chat_message(
    output_dir: str,
    cv_filename: str,
    role: str,
    content: str,
) -> None:
    """Append one chat message to the stored history."""
    store = load_cv_content(output_dir, cv_filename) or deepcopy(DEFAULT_STORE)
    store.setdefault("chat_history", []).append({"role": role, "content": content})
    save_cv_content(
        output_dir,
        cv_filename,
        store.get("tailored_content", {}),
        chat_history=store.get("chat_history", []),
        cover_letter=store.get("cover_letter", {}),
        cover_letter_chat_history=store.get("cover_letter_chat_history", []),
    )
