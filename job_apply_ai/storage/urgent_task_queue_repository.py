"""CRUD and worker coordination for urgent UI-bound I/O tasks (search, scrape)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from job_apply_ai.storage.database import get_connection

VALID_STATUSES = frozenset({"pending", "running", "paused", "completed", "failed", "cancelled"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

URGENT_TASK_TYPES = frozenset({"single_search", "linkedin_job_import"})

CONTROLLABLE_URGENT_TASK_TYPES = frozenset({"single_search"})


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _deserialize_job(row: dict) -> dict:
    job = dict(row)
    job["payload"] = json.loads(job.pop("payload_json") or "{}")
    job["result"] = json.loads(job.pop("result_json") or "{}")
    return job


def to_urgent_task_snapshot(job: dict) -> dict:
    """Map a queue job to the cv_tasks-compatible polling payload."""
    status_map = {
        "pending": "pending",
        "running": "running",
        "paused": "paused",
        "completed": "complete",
        "failed": "error",
        "cancelled": "error",
    }
    mapped_status = status_map.get(job["status"], job["status"])
    payload = job.get("payload") or {}
    result = job.get("result") or {}
    meta = dict(payload.get("meta") or {})
    meta["queue_job_id"] = job["id"]

    for key in ("keyword", "location", "linkedin_url", "total_jobs", "current_index"):
        if key in payload and key not in meta:
            meta[key] = payload[key]

    snapshot = {
        "task_id": job["task_id"],
        "task_type": job["task_type"],
        "status": mapped_status,
        "step": job.get("progress_step") or job["status"],
        "message": job.get("progress_message") or "",
        "percent": job.get("progress_percent", 0),
        "job_id": job.get("job_id"),
        "meta": meta,
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "result": result if job["status"] == "completed" else None,
        "error": job.get("last_error") or None,
    }
    if mapped_status == "error" and not snapshot["error"]:
        snapshot["error"] = snapshot["message"] or "Task failed"
    return snapshot


class UrgentTaskQueueRepository:
    """Repository for urgent UI I/O tasks stored in SQLite."""

    def list_jobs(self, *, include_terminal: bool = True) -> list[dict]:
        query = "SELECT * FROM urgent_task_jobs"
        if not include_terminal:
            query += " WHERE status NOT IN ('completed', 'failed', 'cancelled')"
        query += " ORDER BY id DESC"
        with get_connection() as conn:
            rows = conn.execute(query).fetchall()
        return [_deserialize_job(dict(row)) for row in rows]

    def get_job(self, job_id: int) -> dict | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM urgent_task_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return _deserialize_job(dict(row)) if row else None

    def get_job_by_task_id(self, task_id: str) -> dict | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM urgent_task_jobs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return _deserialize_job(dict(row)) if row else None

    def count_running(self) -> int:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM urgent_task_jobs WHERE status = 'running'"
            ).fetchone()
        return int(row["c"]) if row else 0

    def create_job(
        self,
        *,
        task_type: str,
        payload: dict | None = None,
        job_id: int | None = None,
    ) -> dict:
        if task_type not in URGENT_TASK_TYPES:
            raise ValueError(f"Invalid urgent task type: {task_type}")

        now = _now_iso()
        task_id = uuid.uuid4().hex
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO urgent_task_jobs (
                    task_type, status, job_id, payload_json, task_id, created_at, updated_at
                ) VALUES (?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    task_type,
                    job_id,
                    json.dumps(payload or {}),
                    task_id,
                    now,
                    now,
                ),
            )
            inserted_id = int(cursor.lastrowid)
        return self.get_job(inserted_id)  # type: ignore[return-value]

    def pause_job(self, job_id: int) -> bool:
        with get_connection() as conn:
            updated = conn.execute(
                """
                UPDATE urgent_task_jobs
                SET status = 'paused',
                    progress_message = 'Paused',
                    updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (_now_iso(), job_id),
            )
        return updated.rowcount > 0

    def resume_job(self, job_id: int) -> bool:
        with get_connection() as conn:
            updated = conn.execute(
                """
                UPDATE urgent_task_jobs
                SET status = 'running',
                    progress_message = 'Resumed',
                    updated_at = ?
                WHERE id = ? AND status = 'paused'
                """,
                (_now_iso(), job_id),
            )
        return updated.rowcount > 0

    def request_stop(self, job_id: int) -> bool:
        with get_connection() as conn:
            updated = conn.execute(
                """
                UPDATE urgent_task_jobs
                SET control = 'stop', updated_at = ?
                WHERE id = ? AND status IN ('pending', 'running', 'paused')
                """,
                (_now_iso(), job_id),
            )
        return updated.rowcount > 0

    def claim_next_pending(self, *, max_concurrent: int = 1) -> dict | None:
        """Claim the oldest pending urgent job if running count is below max_concurrent."""
        max_concurrent = max(1, max_concurrent)
        with get_connection() as conn:
            running = conn.execute(
                "SELECT COUNT(*) AS c FROM urgent_task_jobs WHERE status = 'running'"
            ).fetchone()
            if running and int(running["c"]) >= max_concurrent:
                return None

            row = conn.execute(
                """
                SELECT id FROM urgent_task_jobs
                WHERE status = 'pending'
                ORDER BY id
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None

            job_id = row["id"]
            updated = conn.execute(
                """
                UPDATE urgent_task_jobs
                SET status = 'running',
                    control = NULL,
                    progress_message = 'Starting…',
                    progress_step = 'starting',
                    progress_percent = 1,
                    last_error = '',
                    updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (_now_iso(), job_id),
            )
            if updated.rowcount == 0:
                return None
        return self.get_job(job_id)

    def update_progress(
        self,
        job_id: int,
        *,
        step: str | None = None,
        message: str | None = None,
        percent: int | None = None,
        payload_patch: dict | None = None,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        fields: dict[str, Any] = {"updated_at": _now_iso()}
        if step is not None:
            fields["progress_step"] = step
        if message is not None:
            fields["progress_message"] = message
        if percent is not None:
            fields["progress_percent"] = max(0, min(100, percent))
        if payload_patch:
            merged_payload = {**(job.get("payload") or {}), **payload_patch}
            fields["payload_json"] = json.dumps(merged_payload)

        assignments = ", ".join(f"{column} = ?" for column in fields)
        values = [*fields.values(), job_id]
        with get_connection() as conn:
            conn.execute(
                f"UPDATE urgent_task_jobs SET {assignments} WHERE id = ?",
                values,
            )

    def complete_job(self, job_id: int, *, result: dict, message: str) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE urgent_task_jobs
                SET status = 'completed',
                    result_json = ?,
                    progress_message = ?,
                    progress_percent = 100,
                    progress_step = 'complete',
                    control = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(result), message, _now_iso(), job_id),
            )

    def fail_job(self, job_id: int, error: str, *, result: dict | None = None) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE urgent_task_jobs
                SET status = 'failed',
                    last_error = ?,
                    result_json = ?,
                    progress_message = ?,
                    progress_step = 'error',
                    control = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (error, json.dumps(result or {}), error, _now_iso(), job_id),
            )

    def mark_cancelled(self, job_id: int, *, message: str, result: dict) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE urgent_task_jobs
                SET status = 'cancelled',
                    result_json = ?,
                    progress_message = ?,
                    progress_step = 'cancelled',
                    progress_percent = 100,
                    control = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(result), message, _now_iso(), job_id),
            )

    def get_control_state(self, job_id: int) -> tuple[str, str | None]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT status, control FROM urgent_task_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            return "", None
        return row["status"], row["control"]
