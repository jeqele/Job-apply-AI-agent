"""Session-backed undo/redo history for job workflow folder moves."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SESSION_KEY = "job_move_history"
MAX_HISTORY_ENTRIES = 50

JobMoveChange = dict[str, Any]
HistoryEntry = dict[str, Any]


def _empty_stacks() -> dict[str, list[HistoryEntry]]:
    return {"undo": [], "redo": []}


def _get_stacks(session) -> dict[str, list[HistoryEntry]]:
    raw = session.get(SESSION_KEY)
    if not isinstance(raw, dict):
        return _empty_stacks()
    undo = raw.get("undo")
    redo = raw.get("redo")
    return {
        "undo": list(undo) if isinstance(undo, list) else [],
        "redo": list(redo) if isinstance(redo, list) else [],
    }


def _save_stacks(session, stacks: dict[str, list[HistoryEntry]]) -> None:
    session[SESSION_KEY] = stacks
    if hasattr(session, "modified"):
        session.modified = True


def _normalize_changes(changes: list[JobMoveChange]) -> list[JobMoveChange]:
    normalized: list[JobMoveChange] = []
    for change in changes:
        job_id = change.get("job_id")
        from_status = change.get("from_status")
        to_status = change.get("to_status")
        if job_id is None or not from_status or not to_status:
            continue
        if from_status == to_status:
            continue
        normalized.append(
            {
                "job_id": int(job_id),
                "from_status": str(from_status),
                "to_status": str(to_status),
            }
        )
    return normalized


def record_job_moves(session, changes: list[JobMoveChange], label: str) -> None:
    """Push a move operation onto the undo stack and clear redo."""
    normalized = _normalize_changes(changes)
    if not normalized:
        return

    stacks = _get_stacks(session)
    stacks["undo"].append(
        {
            "label": label.strip() or f"Move {len(normalized)} job(s)",
            "changes": deepcopy(normalized),
        }
    )
    if len(stacks["undo"]) > MAX_HISTORY_ENTRIES:
        stacks["undo"] = stacks["undo"][-MAX_HISTORY_ENTRIES:]
    stacks["redo"] = []
    _save_stacks(session, stacks)


def can_undo_job_moves(session) -> bool:
    return bool(_get_stacks(session)["undo"])


def can_redo_job_moves(session) -> bool:
    return bool(_get_stacks(session)["redo"])


def undo_job_move_label(session) -> str | None:
    stacks = _get_stacks(session)
    if not stacks["undo"]:
        return None
    return stacks["undo"][-1].get("label")


def redo_job_move_label(session) -> str | None:
    stacks = _get_stacks(session)
    if not stacks["redo"]:
        return None
    return stacks["redo"][-1].get("label")


def pop_undo_job_moves(session) -> HistoryEntry | None:
    stacks = _get_stacks(session)
    if not stacks["undo"]:
        return None
    entry = stacks["undo"].pop()
    stacks["redo"].append(entry)
    _save_stacks(session, stacks)
    return entry


def pop_redo_job_moves(session) -> HistoryEntry | None:
    stacks = _get_stacks(session)
    if not stacks["redo"]:
        return None
    entry = stacks["redo"].pop()
    stacks["undo"].append(entry)
    _save_stacks(session, stacks)
    return entry
