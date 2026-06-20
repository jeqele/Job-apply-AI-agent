"""Background task tracking for AI CV generation."""

from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

_tasks: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def create_task(
    task_type: str,
    *,
    job_id: int | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    task_id = uuid.uuid4().hex
    record = {
        "task_id": task_id,
        "task_type": task_type,
        "status": "pending",
        "step": "pending",
        "message": "Waiting to start…",
        "percent": 0,
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "result": None,
        "error": None,
        "meta": meta or {},
    }
    with _lock:
        _tasks[task_id] = record
    return task_id


def update_task(
    task_id: str,
    *,
    status: str | None = None,
    step: str | None = None,
    message: str | None = None,
    percent: int | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return
        if status is not None:
            task["status"] = status
        if step is not None:
            task["step"] = step
        if message is not None:
            task["message"] = message
        if percent is not None:
            task["percent"] = max(0, min(100, percent))
        if meta:
            task["meta"].update(meta)
        task["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")


def complete_task(task_id: str, result: dict[str, Any]) -> None:
    update_task(
        task_id,
        status="complete",
        step="complete",
        message="CV generated successfully",
        percent=100,
    )
    with _lock:
        if task_id in _tasks:
            _tasks[task_id]["result"] = result


def fail_task(task_id: str, error: str) -> None:
    update_task(
        task_id,
        status="error",
        step="error",
        message=error,
    )
    with _lock:
        if task_id in _tasks:
            _tasks[task_id]["error"] = error


def get_task(task_id: str) -> dict[str, Any] | None:
    with _lock:
        task = _tasks.get(task_id)
        return deepcopy(task) if task else None


def start_background_task(task_id: str, worker: Callable[[], None]) -> None:
    update_task(task_id, status="running", step="starting", message="Starting…", percent=1)

    def _run() -> None:
        try:
            worker()
        except Exception as exc:
            fail_task(task_id, str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
