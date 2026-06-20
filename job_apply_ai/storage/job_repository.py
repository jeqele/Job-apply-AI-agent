"""CRUD operations for job listings stored in SQLite."""

import json
import logging
from datetime import datetime
from typing import Any

from job_apply_ai.job_schema import JOB_COLUMNS
from job_apply_ai.job_status import DEFAULT_JOB_STATUS, is_valid_job_status
from job_apply_ai.storage.database import get_connection

logger = logging.getLogger(__name__)

JOB_DB_COLUMNS = JOB_COLUMNS + ["matched_skills", "matched_categories"]


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
    return job


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
        """Insert jobs and return their database IDs."""
        if not jobs:
            return []

        ids: list[int] = []
        now = datetime.utcnow().isoformat(timespec="seconds")

        with get_connection() as conn:
            for job in jobs:
                link = (job.get("link") or "").strip()
                existing_id = None
                if link:
                    row = conn.execute(
                        "SELECT id FROM jobs WHERE link = ?",
                        (link,),
                    ).fetchone()
                    if row:
                        existing_id = row["id"]

                values = {
                    column: _serialize_value(job.get(column, ""))
                    for column in JOB_COLUMNS
                }
                values["matched_skills"] = _serialize_value(
                    job.get("matched_skills", [])
                )
                values["matched_categories"] = _serialize_value(
                    job.get("matched_categories", {})
                )
                values["updated_at"] = now

                if existing_id:
                    set_clause = ", ".join(f"{col} = ?" for col in values)
                    conn.execute(
                        f"UPDATE jobs SET {set_clause} WHERE id = ?",
                        (*values.values(), existing_id),
                    )
                    ids.append(existing_id)
                else:
                    columns = ["search_run_id", "workflow_status", *values.keys()]
                    placeholders = ", ".join("?" for _ in columns)
                    cursor = conn.execute(
                        f"""
                        INSERT INTO jobs ({", ".join(columns)})
                        VALUES ({placeholders})
                        """,
                        (search_run_id, DEFAULT_JOB_STATUS, *values.values()),
                    )
                    ids.append(int(cursor.lastrowid))

        logger.info("Upserted %s jobs into SQLite", len(ids))
        return ids

    def _build_job_filters(
        self,
        search_run_id: int | None = None,
        workflow_status: str | None = None,
        search: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if search_run_id is not None:
            clauses.append("search_run_id = ?")
            params.append(search_run_id)

        if workflow_status is not None:
            clauses.append("workflow_status = ?")
            params.append(workflow_status)

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
    ) -> list[dict]:
        query = "SELECT * FROM jobs"
        where_clause, params = self._build_job_filters(
            search_run_id=search_run_id,
            workflow_status=workflow_status,
            search=search,
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
        now = datetime.utcnow().isoformat(timespec="seconds")
        values = {
            column: _serialize_value(job.get(column, ""))
            for column in JOB_COLUMNS
        }
        values["matched_skills"] = _serialize_value(job.get("matched_skills", []))
        values["matched_categories"] = _serialize_value(
            job.get("matched_categories", {})
        )
        values["created_at"] = now
        values["updated_at"] = now

        workflow_status = job.get("workflow_status", DEFAULT_JOB_STATUS)
        if not is_valid_job_status(workflow_status):
            workflow_status = DEFAULT_JOB_STATUS

        columns = ["search_run_id", "workflow_status", *values.keys()]
        placeholders = ", ".join("?" for _ in columns)

        with get_connection() as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO jobs ({", ".join(columns)})
                VALUES ({placeholders})
                """,
                (search_run_id, workflow_status, *values.values()),
            )
            return int(cursor.lastrowid)

    def update_job(self, job_id: int, job: dict) -> bool:
        existing = self.get_job(job_id)
        if not existing:
            return False

        now = datetime.utcnow().isoformat(timespec="seconds")
        values = {
            column: _serialize_value(job.get(column, existing.get(column, "")))
            for column in JOB_COLUMNS
        }
        values["matched_skills"] = _serialize_value(
            job.get("matched_skills", existing.get("matched_skills", []))
        )
        values["matched_categories"] = _serialize_value(
            job.get("matched_categories", existing.get("matched_categories", {}))
        )
        values["cv_filename"] = _serialize_value(
            job.get("cv_filename", existing.get("cv_filename", ""))
        )
        values["updated_at"] = now

        set_clause = ", ".join(f"{col} = ?" for col in values)
        with get_connection() as conn:
            conn.execute(
                f"UPDATE jobs SET {set_clause} WHERE id = ?",
                (*values.values(), job_id),
            )
        return True

    def update_job_status(self, job_id: int, workflow_status: str) -> bool:
        if not is_valid_job_status(workflow_status):
            return False

        existing = self.get_job(job_id)
        if not existing:
            return False

        now = datetime.utcnow().isoformat(timespec="seconds")
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET workflow_status = ?, updated_at = ?
                WHERE id = ?
                """,
                (workflow_status, now, job_id),
            )
        return True

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
