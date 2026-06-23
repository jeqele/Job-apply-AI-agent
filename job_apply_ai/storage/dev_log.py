"""Persistent developer logs for agent and LLM debugging."""

from __future__ import annotations

import json
from typing import Any

from job_apply_ai.storage.database import get_connection

DEV_LOG_CATEGORIES = ("agent", "llm", "task", "system")
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000


class DevLogRepository:
    """Store and query developer logs in SQLite."""

    def add_log(
        self,
        *,
        category: str,
        event: str,
        message: str = "",
        agent: str = "",
        data: dict[str, Any] | None = None,
        task_id: str = "",
        job_id: int | None = None,
    ) -> int:
        category = category if category in DEV_LOG_CATEGORIES else "system"
        payload = json.dumps(data or {}, ensure_ascii=False)
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO dev_logs (category, agent, event, message, data, task_id, job_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (category, agent or "", event or "", message or "", payload, task_id or "", job_id),
            )
            return int(cursor.lastrowid)

    def list_logs(
        self,
        *,
        category: str = "",
        agent: str = "",
        search: str = "",
        task_id: str = "",
        since_id: int = 0,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if category and category in DEV_LOG_CATEGORIES:
            clauses.append("category = ?")
            params.append(category)
        if agent:
            clauses.append("agent = ?")
            params.append(agent)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if since_id > 0:
            clauses.append("id > ?")
            params.append(since_id)
        if search:
            needle = f"%{search.strip()}%"
            clauses.append(
                "(message LIKE ? OR event LIKE ? OR agent LIKE ? OR data LIKE ?)"
            )
            params.extend([needle, needle, needle, needle])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        safe_limit = max(1, min(int(limit), MAX_LIMIT))
        safe_offset = max(0, int(offset))

        query = f"""
            SELECT id, category, agent, event, message, data, task_id, job_id, created_at
            FROM dev_logs
            {where}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """
        params.extend([safe_limit, safe_offset])

        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [_row_to_dict(row) for row in rows]

    def count_logs(
        self,
        *,
        category: str = "",
        agent: str = "",
        search: str = "",
        task_id: str = "",
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []

        if category and category in DEV_LOG_CATEGORIES:
            clauses.append("category = ?")
            params.append(category)
        if agent:
            clauses.append("agent = ?")
            params.append(agent)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if search:
            needle = f"%{search.strip()}%"
            clauses.append(
                "(message LIKE ? OR event LIKE ? OR agent LIKE ? OR data LIKE ?)"
            )
            params.extend([needle, needle, needle, needle])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with get_connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS total FROM dev_logs {where}", params).fetchone()
        return int(row["total"]) if row else 0

    def list_agents(self) -> list[str]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT agent
                FROM dev_logs
                WHERE agent != ''
                ORDER BY agent ASC
                """
            ).fetchall()
        return [row["agent"] for row in rows]

    def clear_logs(
        self,
        *,
        category: str = "",
        task_id: str = "",
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []

        if category and category in DEV_LOG_CATEGORIES:
            clauses.append("category = ?")
            params.append(category)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)

        if not clauses:
            with get_connection() as conn:
                cursor = conn.execute("DELETE FROM dev_logs")
            return cursor.rowcount

        where = " AND ".join(clauses)
        with get_connection() as conn:
            cursor = conn.execute(f"DELETE FROM dev_logs WHERE {where}", params)
        return cursor.rowcount


def _row_to_dict(row: Any) -> dict[str, Any]:
    try:
        data = json.loads(row["data"] or "{}")
    except json.JSONDecodeError:
        data = {"raw": row["data"]}
    if not isinstance(data, dict):
        data = {"value": data}
    return {
        "id": row["id"],
        "category": row["category"],
        "agent": row["agent"],
        "event": row["event"],
        "message": row["message"],
        "data": data,
        "task_id": row["task_id"] or "",
        "job_id": row["job_id"],
        "created_at": row["created_at"],
    }
