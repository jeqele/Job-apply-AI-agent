"""SQLite database initialization and connection helpers."""

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs",
    "data",
)


def get_db_path() -> str:
    """Return the path to the SQLite database file."""
    custom = os.environ.get("JOB_APPLY_AI_DB")
    if custom:
        return custom
    os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
    return os.path.join(DEFAULT_DB_DIR, "jobs.db")


def init_db(db_path: str | None = None) -> None:
    """Create tables if they do not exist."""
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS search_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                location TEXT NOT NULL,
                sources TEXT,
                mode TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_run_id INTEGER,
                title TEXT NOT NULL DEFAULT '',
                company TEXT DEFAULT '',
                location TEXT DEFAULT '',
                work_type TEXT DEFAULT '',
                salary TEXT DEFAULT '',
                employment_type TEXT DEFAULT '',
                seniority_level TEXT DEFAULT '',
                visa_sponsorship TEXT DEFAULT '',
                relocation_support TEXT DEFAULT '',
                relocation_info TEXT DEFAULT '',
                job_function TEXT DEFAULT '',
                industry TEXT DEFAULT '',
                applicant_count TEXT DEFAULT '',
                listing_benefit TEXT DEFAULT '',
                emails TEXT DEFAULT '',
                application_method TEXT DEFAULT '',
                posted_days_ago TEXT DEFAULT '',
                posted_date TEXT DEFAULT '',
                link TEXT DEFAULT '',
                company_url TEXT DEFAULT '',
                description TEXT DEFAULT '',
                source TEXT DEFAULT '',
                fetch_method TEXT DEFAULT '',
                matched_skills TEXT DEFAULT '[]',
                matched_categories TEXT DEFAULT '{}',
                dedupe_key TEXT NOT NULL DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (search_run_id) REFERENCES search_runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_search_run ON jobs(search_run_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_link ON jobs(link);
            """
        )

        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "workflow_status" not in columns:
            conn.execute(
                """
                ALTER TABLE jobs
                ADD COLUMN workflow_status TEXT NOT NULL DEFAULT 'new'
                """
            )
        if "cv_filename" not in columns:
            conn.execute(
                """
                ALTER TABLE jobs
                ADD COLUMN cv_filename TEXT NOT NULL DEFAULT ''
                """
            )
        if "cover_letter_filename" not in columns:
            conn.execute(
                """
                ALTER TABLE jobs
                ADD COLUMN cover_letter_filename TEXT NOT NULL DEFAULT ''
                """
            )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_workflow_status
            ON jobs(workflow_status)
            """
        )
        conn.execute(
            """
            UPDATE jobs
            SET workflow_status = 'new'
            WHERE workflow_status IS NULL OR workflow_status = ''
            """
        )

        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        dedupe_column_added = False
        if "dedupe_key" not in columns:
            conn.execute(
                """
                ALTER TABLE jobs
                ADD COLUMN dedupe_key TEXT NOT NULL DEFAULT ''
                """
            )
            dedupe_column_added = True

        if dedupe_column_added:
            _backfill_dedupe_keys(conn)
        else:
            _ensure_dedupe_keys(conn)

        conn.execute("DROP INDEX IF EXISTS idx_jobs_dedupe_key")
        conn.execute(
            """
            CREATE UNIQUE INDEX idx_jobs_dedupe_key
            ON jobs(dedupe_key)
            """
        )

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS app_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS dev_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'system',
                agent TEXT NOT NULL DEFAULT '',
                event TEXT NOT NULL DEFAULT '',
                message TEXT NOT NULL DEFAULT '',
                data TEXT NOT NULL DEFAULT '{}',
                task_id TEXT NOT NULL DEFAULT '',
                job_id INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_dev_logs_category ON dev_logs(category);
            CREATE INDEX IF NOT EXISTS idx_dev_logs_task_id ON dev_logs(task_id);
            CREATE INDEX IF NOT EXISTS idx_dev_logs_created_at ON dev_logs(created_at);
            """
        )
        conn.commit()


def _ensure_dedupe_keys(conn: sqlite3.Connection) -> None:
    """Fill missing dedupe keys without re-processing fully keyed rows."""
    from job_apply_ai.job_dedupe import compute_dedupe_key

    rows = conn.execute(
        """
        SELECT id, title, company, location, link
        FROM jobs
        WHERE dedupe_key = '' OR dedupe_key IS NULL
        ORDER BY id ASC
        """
    ).fetchall()

    for row in rows:
        job = {
            "title": row["title"],
            "company": row["company"],
            "location": row["location"],
            "link": row["link"],
        }
        dedupe_key = compute_dedupe_key(job) or f"legacy:{row['id']}"
        existing = conn.execute(
            "SELECT id FROM jobs WHERE dedupe_key = ? AND id != ?",
            (dedupe_key, row["id"]),
        ).fetchone()
        if existing:
            conn.execute("DELETE FROM jobs WHERE id = ?", (row["id"],))
            continue

        conn.execute(
            "UPDATE jobs SET dedupe_key = ? WHERE id = ?",
            (dedupe_key, row["id"]),
        )


def _backfill_dedupe_keys(conn: sqlite3.Connection) -> None:
    """Populate dedupe keys and remove duplicate rows from existing databases."""
    from job_apply_ai.job_dedupe import compute_dedupe_key

    rows = conn.execute(
        """
        SELECT id, title, company, location, link
        FROM jobs
        ORDER BY id ASC
        """
    ).fetchall()

    kept_by_key: dict[str, int] = {}
    removed = 0

    for row in rows:
        job = {
            "title": row["title"],
            "company": row["company"],
            "location": row["location"],
            "link": row["link"],
        }
        dedupe_key = compute_dedupe_key(job) or f"legacy:{row['id']}"

        existing_id = kept_by_key.get(dedupe_key)
        if existing_id is not None:
            conn.execute("DELETE FROM jobs WHERE id = ?", (row["id"],))
            removed += 1
            continue

        kept_by_key[dedupe_key] = row["id"]
        conn.execute(
            "UPDATE jobs SET dedupe_key = ? WHERE id = ?",
            (dedupe_key, row["id"]),
        )

    if removed:
        logger.info("Removed %s duplicate jobs during dedupe_key migration", removed)


@contextmanager
def get_connection(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row factory enabled."""
    path = db_path or get_db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
