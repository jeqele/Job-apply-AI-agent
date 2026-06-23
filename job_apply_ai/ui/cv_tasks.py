"""Background task tracking for AI CV generation."""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

_tasks: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


class TaskStopped(Exception):
    """Raised when a background task should exit early."""


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
        "control": None,
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

    try:
        from job_apply_ai.dev_logging import dev_log, is_dev_mode

        if is_dev_mode():
            dev_log(
                "task",
                "task_update",
                message or "",
                data={"step": step, "status": status, "percent": percent},
                task_id=task_id,
            )
    except Exception:
        pass


def complete_task(
    task_id: str,
    result: dict[str, Any],
    *,
    message: str = "CV generated successfully",
) -> None:
    update_task(
        task_id,
        status="complete",
        step="complete",
        message=message,
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


def pause_task(task_id: str) -> bool:
    with _lock:
        task = _tasks.get(task_id)
        if not task or task.get("status") != "running":
            return False
        task["status"] = "paused"
        task["message"] = "Paused"
        task["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    return True


def resume_task(task_id: str) -> bool:
    with _lock:
        task = _tasks.get(task_id)
        if not task or task.get("status") != "paused":
            return False
        task["status"] = "running"
        task["message"] = "Resumed"
        task["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    return True


def request_task_stop(task_id: str) -> bool:
    with _lock:
        task = _tasks.get(task_id)
        if not task or task.get("status") not in ("pending", "running", "paused"):
            return False
        task["control"] = "stop"
        task["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    return True


def task_control_checkpoint(task_id: str) -> None:
    """Block while paused; raise TaskStopped if stop was requested."""
    while True:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                raise TaskStopped("Task not found")
            if task.get("control") == "stop":
                raise TaskStopped("Stopped by user")
            paused = task.get("status") == "paused"
        if not paused:
            return
        time.sleep(0.25)


def start_background_task(task_id: str, worker: Callable[[], None]) -> None:
    update_task(task_id, status="running", step="starting", message="Starting…", percent=1)

    def _run() -> None:
        try:
            worker()
        except Exception as exc:
            fail_task(task_id, str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()