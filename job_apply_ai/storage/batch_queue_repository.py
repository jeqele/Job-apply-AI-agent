"""CRUD and worker coordination for the batch search job queue."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from job_apply_ai.job_sources import UI_DEFAULT_JOB_SOURCES
from job_apply_ai.batch_search import (
    build_search_queue,
    shuffle_search_queue,
    split_batch_inputs,
    validate_batch_queue,
)
from job_apply_ai.scraper.search_filters import SearchFilters
from job_apply_ai.storage.database import get_connection

logger = logging.getLogger(__name__)

VALID_STATUSES = frozenset(
    {"pending", "running", "paused", "completed", "failed", "cancelled"}
)
VALID_SCHEDULES = frozenset({"once", "daily", "weekly"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

SCHEDULE_LABELS = {
    "once": "One-time",
    "daily": "Daily",
    "weekly": "Weekly",
}

STATUS_LABELS = {
    "pending": "Pending",
    "running": "Running",
    "paused": "Paused",
    "completed": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _compute_next_run(schedule_type: str, *, from_time: datetime | None = None) -> str | None:
    if schedule_type == "once":
        return None
    base = from_time or datetime.utcnow()
    delta = timedelta(days=1 if schedule_type == "daily" else 7)
    return (base + delta).strftime("%Y-%m-%d %H:%M:%S")


def _deserialize_job(row: dict) -> dict:
    job = dict(row)
    job["titles"] = json.loads(job.pop("titles_json") or "[]")
    job["locations"] = json.loads(job.pop("locations_json") or "[]")
    job["search_filters"] = json.loads(job.pop("search_filters_json") or "{}")
    job["result"] = json.loads(job.pop("result_json") or "{}")
    job["shuffle_queue"] = bool(job.get("shuffle_queue"))
    return job


def _serialize_filters(search_filters: SearchFilters | dict | None) -> str:
    if isinstance(search_filters, SearchFilters):
        payload = {
            "remote": search_filters.remote,
            "relocation": search_filters.relocation,
            "visa_sponsorship": search_filters.visa_sponsorship,
        }
    elif isinstance(search_filters, dict):
        payload = search_filters
    else:
        payload = {}
    return json.dumps(payload)


def filters_from_dict(data: dict | None) -> SearchFilters:
    if not data:
        return SearchFilters()
    return SearchFilters(
        remote=bool(data.get("remote")),
        relocation=bool(data.get("relocation")),
        visa_sponsorship=bool(data.get("visa_sponsorship")),
    )


def to_task_snapshot(job: dict) -> dict:
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
    result = job.get("result") or {}
    meta = {
        "queue_job_id": job["id"],
        "total_searches": job.get("total_combinations", 0),
        "current_index": job.get("current_index", 0),
    }
    if job.get("progress_step") == "searching":
        keyword = result.get("current_keyword")
        location = result.get("current_location")
        if keyword and location:
            meta["keyword"] = keyword
            meta["location"] = location

    snapshot = {
        "task_id": job["task_id"],
        "task_type": "batch_search",
        "status": mapped_status,
        "step": job.get("progress_step") or job["status"],
        "message": job.get("progress_message") or "",
        "percent": job.get("progress_percent", 0),
        "meta": meta,
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "result": result if job["status"] == "completed" else None,
        "error": job.get("last_error") or None,
    }
    if mapped_status == "error" and not snapshot["error"]:
        snapshot["error"] = snapshot["message"] or "Batch search failed"
    return snapshot


class BatchQueueRepository:
    """Repository for batch search queue jobs stored in SQLite."""

    def list_jobs(self, *, include_terminal: bool = True) -> list[dict]:
        query = "SELECT * FROM batch_search_jobs"
        if not include_terminal:
            query += " WHERE status NOT IN ('completed', 'failed', 'cancelled')"
        query += " ORDER BY id DESC"
        with get_connection() as conn:
            rows = conn.execute(query).fetchall()
        return [_deserialize_job(dict(row)) for row in rows]

    def get_job(self, job_id: int) -> dict | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM batch_search_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return _deserialize_job(dict(row)) if row else None

    def get_job_by_task_id(self, task_id: str) -> dict | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM batch_search_jobs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return _deserialize_job(dict(row)) if row else None

    def create_jobs(
        self,
        *,
        name: str,
        titles: list[str],
        locations: list[str],
        schedule_type: str = "once",
        shuffle_queue: bool = False,
        max_jobs: int = 5,
        sources: str = UI_DEFAULT_JOB_SOURCES,
        mode: str = "both",
        search_filters: SearchFilters | dict | None = None,
        run_immediately: bool = True,
    ) -> list[dict]:
        schedule_type = schedule_type if schedule_type in VALID_SCHEDULES else "once"
        chunks = split_batch_inputs(titles, locations)
        if not chunks:
            raise ValueError("Provide at least one job title and one location.")

        total_parts = len(chunks)
        base_name = name.strip()
        jobs: list[dict] = []
        for part_index, (part_titles, part_locations) in enumerate(chunks, start=1):
            queue = build_search_queue(part_titles, part_locations)
            if shuffle_queue:
                queue = shuffle_search_queue(queue)
            queue_error = validate_batch_queue(queue)
            if queue_error:
                raise ValueError(queue_error)

            if total_parts > 1:
                part_suffix = f" (part {part_index}/{total_parts})"
                if base_name:
                    job_name = f"{base_name}{part_suffix}"
                else:
                    job_name = f"Batch search ({len(queue)} searches){part_suffix}"
            else:
                job_name = base_name or f"Batch search ({len(queue)} combinations)"

            jobs.append(
                self._insert_job(
                    name=job_name,
                    titles=part_titles,
                    locations=part_locations,
                    schedule_type=schedule_type,
                    shuffle_queue=shuffle_queue,
                    max_jobs=max_jobs,
                    sources=sources,
                    mode=mode,
                    search_filters=search_filters,
                    total_combinations=len(queue),
                    run_immediately=run_immediately,
                )
            )
        return jobs

    def create_job(
        self,
        *,
        name: str,
        titles: list[str],
        locations: list[str],
        schedule_type: str = "once",
        shuffle_queue: bool = False,
        max_jobs: int = 5,
        sources: str = UI_DEFAULT_JOB_SOURCES,
        mode: str = "both",
        search_filters: SearchFilters | dict | None = None,
        run_immediately: bool = True,
    ) -> dict:
        jobs = self.create_jobs(
            name=name,
            titles=titles,
            locations=locations,
            schedule_type=schedule_type,
            shuffle_queue=shuffle_queue,
            max_jobs=max_jobs,
            sources=sources,
            mode=mode,
            search_filters=search_filters,
            run_immediately=run_immediately,
        )
        return jobs[0]

    def _insert_job(
        self,
        *,
        name: str,
        titles: list[str],
        locations: list[str],
        schedule_type: str,
        shuffle_queue: bool,
        max_jobs: int,
        sources: str,
        mode: str,
        search_filters: SearchFilters | dict | None,
        total_combinations: int,
        run_immediately: bool,
    ) -> dict:
        now = _now_iso()
        next_run_at = now if run_immediately else None
        task_id = uuid.uuid4().hex

        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO batch_search_jobs (
                    name, status, schedule_type, titles_json, locations_json,
                    shuffle_queue, max_jobs, sources, mode, search_filters_json,
                    total_combinations, task_id, next_run_at, created_at, updated_at
                ) VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    schedule_type,
                    json.dumps(titles),
                    json.dumps(locations),
                    int(shuffle_queue),
                    max_jobs,
                    sources,
                    mode,
                    _serialize_filters(search_filters),
                    total_combinations,
                    task_id,
                    next_run_at,
                    now,
                    now,
                ),
            )
            job_id = int(cursor.lastrowid)
        return self.get_job(job_id)  # type: ignore[return-value]

    def update_job(
        self,
        job_id: int,
        *,
        name: str | None = None,
        titles: list[str] | None = None,
        locations: list[str] | None = None,
        schedule_type: str | None = None,
        shuffle_queue: bool | None = None,
        max_jobs: int | None = None,
        sources: str | None = None,
        mode: str | None = None,
        search_filters: SearchFilters | dict | None = None,
    ) -> dict | None:
        job = self.get_job(job_id)
        if not job:
            return None
        if job["status"] not in {"pending", "paused"}:
            raise ValueError("Only pending or paused jobs can be edited.")

        new_titles = titles if titles is not None else job["titles"]
        new_locations = locations if locations is not None else job["locations"]
        new_shuffle = shuffle_queue if shuffle_queue is not None else job["shuffle_queue"]
        queue = build_search_queue(new_titles, new_locations)
        if new_shuffle:
            queue = shuffle_search_queue(queue)
        queue_error = validate_batch_queue(queue)
        if queue_error:
            raise ValueError(queue_error)

        fields: dict[str, Any] = {
            "name": name.strip() if name is not None else job["name"],
            "titles_json": json.dumps(new_titles),
            "locations_json": json.dumps(new_locations),
            "shuffle_queue": int(new_shuffle),
            "total_combinations": len(queue),
            "updated_at": _now_iso(),
        }
        if schedule_type is not None:
            if schedule_type not in VALID_SCHEDULES:
                raise ValueError(f"Invalid schedule type: {schedule_type}")
            fields["schedule_type"] = schedule_type
        if max_jobs is not None:
            fields["max_jobs"] = max_jobs
        if sources is not None:
            fields["sources"] = sources
        if mode is not None:
            fields["mode"] = mode
        if search_filters is not None:
            fields["search_filters_json"] = _serialize_filters(search_filters)

        assignments = ", ".join(f"{column} = ?" for column in fields)
        values = [*fields.values(), job_id]
        with get_connection() as conn:
            conn.execute(
                f"UPDATE batch_search_jobs SET {assignments} WHERE id = ?",
                values,
            )
        return self.get_job(job_id)

    def delete_job(self, job_id: int) -> bool:
        job = self.get_job(job_id)
        if not job:
            return False
        if job["status"] == "running":
            raise ValueError("Cannot delete a running job. Stop it first.")
        with get_connection() as conn:
            conn.execute("DELETE FROM batch_search_jobs WHERE id = ?", (job_id,))
        return True

    def pause_job(self, job_id: int) -> bool:
        with get_connection() as conn:
            updated = conn.execute(
                """
                UPDATE batch_search_jobs
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
                UPDATE batch_search_jobs
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
                UPDATE batch_search_jobs
                SET control = 'stop', updated_at = ?
                WHERE id = ? AND status IN ('pending', 'running', 'paused')
                """,
                (_now_iso(), job_id),
            )
        return updated.rowcount > 0

    def cancel_job(self, job_id: int) -> bool:
        """Immediately mark a non-running job as cancelled."""
        with get_connection() as conn:
            updated = conn.execute(
                """
                UPDATE batch_search_jobs
                SET status = 'cancelled',
                    control = 'stop',
                    progress_message = 'Cancelled',
                    updated_at = ?
                WHERE id = ? AND status IN ('pending', 'paused')
                """,
                (_now_iso(), job_id),
            )
        return updated.rowcount > 0

    def claim_next_pending(self) -> dict | None:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT id FROM batch_search_jobs
                WHERE (
                    status = 'pending'
                    AND (next_run_at IS NULL OR next_run_at <= datetime('now'))
                ) OR (
                    status = 'completed'
                    AND schedule_type IN ('daily', 'weekly')
                    AND next_run_at IS NOT NULL
                    AND next_run_at <= datetime('now')
                )
                ORDER BY COALESCE(next_run_at, created_at), id
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            job_id = row["id"]
            updated = conn.execute(
                """
                UPDATE batch_search_jobs
                SET status = 'running',
                    control = NULL,
                    progress_message = 'Starting…',
                    progress_step = 'starting',
                    progress_percent = 1,
                    last_error = '',
                    current_index = 0,
                    last_run_at = ?,
                    updated_at = ?
                WHERE id = ? AND status IN ('pending', 'completed')
                """,
                (_now_iso(), _now_iso(), job_id),
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
        current_index: int | None = None,
        result_patch: dict | None = None,
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
        if current_index is not None:
            fields["current_index"] = current_index
        if result_patch:
            merged = {**(job.get("result") or {}), **result_patch}
            fields["result_json"] = json.dumps(merged)

        assignments = ", ".join(f"{column} = ?" for column in fields)
        values = [*fields.values(), job_id]
        with get_connection() as conn:
            conn.execute(
                f"UPDATE batch_search_jobs SET {assignments} WHERE id = ?",
                values,
            )

    def complete_job(
        self,
        job_id: int,
        *,
        search_run_id: int,
        result: dict,
        message: str,
        reschedule: bool,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        now = _now_iso()
        schedule_type = job["schedule_type"]
        if reschedule and schedule_type in {"daily", "weekly"}:
            next_run_at = _compute_next_run(schedule_type)
            progress_message = f"{message} — next run at {next_run_at}"
        else:
            next_run_at = None
            progress_message = message

        with get_connection() as conn:
            conn.execute(
                """
                UPDATE batch_search_jobs
                SET status = 'completed',
                    search_run_id = ?,
                    result_json = ?,
                    progress_message = ?,
                    progress_percent = 100,
                    progress_step = 'complete',
                    control = NULL,
                    next_run_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    search_run_id,
                    json.dumps(result),
                    progress_message,
                    next_run_at,
                    now,
                    job_id,
                ),
            )

    def fail_job(self, job_id: int, error: str, *, result: dict | None = None) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE batch_search_jobs
                SET status = 'failed',
                    last_error = ?,
                    result_json = ?,
                    progress_message = ?,
                    progress_step = 'error',
                    control = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    error,
                    json.dumps(result or {}),
                    error,
                    _now_iso(),
                    job_id,
                ),
            )

    def mark_cancelled(self, job_id: int, *, message: str, result: dict) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE batch_search_jobs
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
                "SELECT status, control FROM batch_search_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            return "", None
        return row["status"], row["control"]
