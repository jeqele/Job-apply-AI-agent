"""CRUD operations for job listings stored in SQLite."""

import json
import logging
from datetime import datetime
from typing import Any

from job_apply_ai.job_dedupe import compute_dedupe_key
from job_apply_ai.job_schema import JOB_COLUMNS
from job_apply_ai.job_status import DEFAULT_JOB_STATUS, is_valid_job_status
from job_apply_ai.storage.database import get_connection

logger = logging.getLogger(__name__)

JOB_DB_COLUMNS = JOB_COLUMNS + ["matched_skills", "matched_categories"]
UPSERT_UPDATE_COLUMNS = [
    *JOB_COLUMNS,
    "matched_skills",
    "matched_categories",
    "search_run_id",
    "updated_at",
]


def _serialize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def _deserialize_job(row: dict) -> dict:
    job = dict(row)
    for field in ("matched_skills", "matched_categories"):
        raw = job.get(field) or ("[]" if field == "matched_skills" else "{}")
        try:
            job[field] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            job[field] = [] if field == "matched_skills" else {}
    job.pop("dedupe_key", None)
    return job


def _job_row_values(job: dict, now: str) -> dict[str, str]:
    values = {
        column: _serialize_value(job.get(column, ""))
        for column in JOB_COLUMNS
    }
    values["matched_skills"] = _serialize_value(job.get("matched_skills", []))
    values["matched_categories"] = _serialize_value(
        job.get("matched_categories", {})
    )
    values["updated_at"] = now
    return values


class JobRepository:
    """Repository for job CRUD against SQLite."""

    def create_search_run(
        self,
        keyword: str,
        location: str,
        sources: str = "",
        mode: str = "both",
    ) -> int:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO search_runs (keyword, location, sources, mode)
                VALUES (?, ?, ?, ?)
                """,
                (keyword, location, sources, mode),
            )
            return int(cursor.lastrowid)

    def upsert_jobs(
        self,
        jobs: list[dict],
        search_run_id: int | None = None,
    ) -> list[int]:
        """Insert or update jobs by dedupe key and return their database IDs."""
        if not jobs:
            return []

        ids: list[int] = []
        now = datetime.utcnow().isoformat(timespec="seconds")
        insert_columns = [
            "search_run_id",
            "workflow_status",
            "dedupe_key",
            *JOB_COLUMNS,
            "matched_skills",
            "matched_categories",
            "created_at",
            "updated_at",
        ]
        update_assignments = ", ".join(
            f"{column} = excluded.{column}" for column in UPSERT_UPDATE_COLUMNS
        )

        with get_connection() as conn:
            for job in jobs:
                dedupe_key = compute_dedupe_key(job)
                if not dedupe_key:
                    logger.warning(
                        "Skipping job without dedupe key: %r",
                        job.get("title", ""),
                    )
                    continue

                values = _job_row_values(job, now)
                workflow_status = job.get("workflow_status", DEFAULT_JOB_STATUS)
                if not is_valid_job_status(workflow_status):
                    workflow_status = DEFAULT_JOB_STATUS

                job_search_run_id = job.get("search_run_id", search_run_id)
                if job_search_run_id is not None:
                    job_search_run_id = int(job_search_run_id)

                insert_values = (
                    job_search_run_id,
                    workflow_status,
                    dedupe_key,
                    *(values[column] for column in JOB_COLUMNS),
                    values["matched_skills"],
                    values["matched_categories"],
                    now,
                    values["updated_at"],
                )

                conn.execute(
                    f"""
                    INSERT INTO jobs ({", ".join(insert_columns)})
                    VALUES ({", ".join("?" for _ in insert_columns)})
                    ON CONFLICT(dedupe_key) DO UPDATE SET
                        {update_assignments}
                    """,
                    insert_values,
                )
                row = conn.execute(
                    "SELECT id FROM jobs WHERE dedupe_key = ?",
                    (dedupe_key,),
                ).fetchone()
                if row:
                    ids.append(int(row["id"]))

        logger.info("Upserted %s jobs into SQLite", len(ids))
        return ids

    def _build_job_filters(
        self,
        search_run_id: int | None = None,
        workflow_status: str | None = None,
        search: str | None = None,
        exclude_workflow_statuses: list[str] | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if search_run_id is not None:
            clauses.append("search_run_id = ?")
            params.append(search_run_id)

        if workflow_status is not None:
            clauses.append("workflow_status = ?")
            params.append(workflow_status)

        if exclude_workflow_statuses:
            placeholders = ", ".join("?" for _ in exclude_workflow_statuses)
            clauses.append(f"workflow_status NOT IN ({placeholders})")
            params.extend(exclude_workflow_statuses)

        if search:
            term = f"%{search.strip()}%"
            search_clause = " OR ".join(
                f"{column} LIKE ?"
                for column in (
                    "title",
                    "company",
                    "location",
                    "description",
                    "source",
                    "emails",
                )
            )
            clauses.append(f"({search_clause})")
            params.extend([term] * 6)

        if not clauses:
            return "", params
        return " WHERE " + " AND ".join(clauses), params

    def list_jobs(
        self,
        search_run_id: int | None = None,
        workflow_status: str | None = None,
        search: str | None = None,
        limit: int | None = None,
        exclude_workflow_statuses: list[str] | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM jobs"
        where_clause, params = self._build_job_filters(
            search_run_id=search_run_id,
            workflow_status=workflow_status,
            search=search,
            exclude_workflow_statuses=exclude_workflow_statuses,
        )
        query += where_clause
        query += " ORDER BY id DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [_deserialize_job(dict(row)) for row in rows]

    def get_job(self, job_id: int) -> dict | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return _deserialize_job(dict(row)) if row else None

    def create_job(self, job: dict, search_run_id: int | None = None) -> int:
        ids = self.upsert_jobs([job], search_run_id=search_run_id)
        if not ids:
            raise ValueError("Job must include at least a title to create a dedupe key")
        return ids[0]

    def update_job(self, job_id: int, job: dict) -> bool:
        existing = self.get_job(job_id)
        if not existing:
            return False

        now = datetime.utcnow().isoformat(timespec="seconds")
        merged = {**existing, **job}
        values = _job_row_values(merged, now)
        values["cv_filename"] = _serialize_value(
            job.get("cv_filename", existing.get("cv_filename", ""))
        )
        values["cover_letter_filename"] = _serialize_value(
            job.get("cover_letter_filename", existing.get("cover_letter_filename", ""))
        )
        dedupe_key = compute_dedupe_key(merged)
        if dedupe_key:
            values["dedupe_key"] = dedupe_key

        set_clause = ", ".join(f"{col} = ?" for col in values)
        with get_connection() as conn:
            conn.execute(
                f"UPDATE jobs SET {set_clause} WHERE id = ?",
                (*values.values(), job_id),
            )
        return True

    def update_job_status(self, job_id: int, workflow_status: str) -> bool:
        return self.update_jobs_status([job_id], workflow_status) == 1

    def move_jobs_status(
        self,
        job_ids: list[int],
        workflow_status: str,
    ) -> list[dict[str, int | str]]:
        """Move jobs to a folder and return change records for undo/redo."""
        if not job_ids or not is_valid_job_status(workflow_status):
            return []

        unique_ids = sorted({int(job_id) for job_id in job_ids})
        placeholders = ", ".join("?" for _ in unique_ids)
        changes: list[dict[str, int | str]] = []

        with get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, workflow_status
                FROM jobs
                WHERE id IN ({placeholders})
                """,
                unique_ids,
            ).fetchall()

            for row in rows:
                from_status = row["workflow_status"] or DEFAULT_JOB_STATUS
                if from_status == workflow_status:
                    continue
                changes.append(
                    {
                        "job_id": int(row["id"]),
                        "from_status": from_status,
                        "to_status": workflow_status,
                    }
                )

            if not changes:
                return []

            now = datetime.utcnow().isoformat(timespec="seconds")
            change_ids = [int(change["job_id"]) for change in changes]
            change_placeholders = ", ".join("?" for _ in change_ids)
            conn.execute(
                f"""
                UPDATE jobs
                SET workflow_status = ?, updated_at = ?
                WHERE id IN ({change_placeholders})
                """,
                (workflow_status, now, *change_ids),
            )

        return changes

    def apply_job_status_changes(
        self,
        changes: list[dict[str, int | str]],
        *,
        use_from_status: bool,
    ) -> int:
        """Restore or re-apply folder moves from stored change records."""
        if not changes:
            return 0

        status_key = "from_status" if use_from_status else "to_status"
        now = datetime.utcnow().isoformat(timespec="seconds")
        updated = 0

        with get_connection() as conn:
            for change in changes:
                job_id = int(change["job_id"])
                workflow_status = change.get(status_key)
                if not is_valid_job_status(workflow_status):
                    continue
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET workflow_status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (workflow_status, now, job_id),
                )
                updated += int(cursor.rowcount)

        return updated

    def update_jobs_status(self, job_ids: list[int], workflow_status: str) -> int:
        return len(self.move_jobs_status(job_ids, workflow_status))

    def count_jobs(
        self,
        search_run_id: int | None = None,
        workflow_status: str | None = None,
        search: str | None = None,
    ) -> int:
        query = "SELECT COUNT(*) AS total FROM jobs"
        where_clause, params = self._build_job_filters(
            search_run_id=search_run_id,
            workflow_status=workflow_status,
            search=search,
        )
        query += where_clause

        with get_connection() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row["total"]) if row else 0

    def count_jobs_by_status(self) -> dict[str, int]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT workflow_status, COUNT(*) AS total
                FROM jobs
                GROUP BY workflow_status
                """
            ).fetchall()

        counts: dict[str, int] = {}
        for row in rows:
            status = row["workflow_status"] or DEFAULT_JOB_STATUS
            counts[status] = counts.get(status, 0) + int(row["total"])
        return counts
