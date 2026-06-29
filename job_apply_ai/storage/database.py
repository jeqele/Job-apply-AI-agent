"""SQLite database initialization and connection helpers."""

import logging
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "data",
    )
)

_DB_CONNECT_TIMEOUT_SECONDS = 30.0
_init_lock = threading.Lock()
_initialized_paths: set[str] = set()


def _ensure_db_parent_dir(path: str) -> None:
    """Create the parent directory for a SQLite database file when needed."""
    parent = os.path.dirname(path)
    if not parent:
        return
    os.makedirs(parent, exist_ok=True)


def _diagnose_db_open_error(path: str, exc: Exception) -> str:
    """Build a human-readable hint for SQLite open/write failures."""
    parent = os.path.dirname(path) or "."
    details = [f"SQLite error for database {path!r}: {exc}"]

    if not os.path.exists(parent):
        details.append(f"Parent directory does not exist: {parent!r}")
    elif not os.access(parent, os.W_OK):
        details.append(f"Parent directory is not writable: {parent!r}")

    try:
        usage = shutil.disk_usage(parent)
        free_mb = usage.free // (1024 * 1024)
        details.append(f"Disk free in parent directory: {free_mb} MB")
        if free_mb < 64:
            details.append("Disk space is critically low; SQLite cannot create journal files")
    except OSError as disk_exc:
        details.append(f"Could not inspect disk usage for {parent!r}: {disk_exc}")

    details.append(
        "Ensure web UI and all workers share the same absolute JOB_APPLY_AI_DB path"
    )
    return ". ".join(details)


def _connect(path: str) -> sqlite3.Connection:
    """Open a SQLite connection with a generous lock timeout."""
    try:
        return sqlite3.connect(path, timeout=_DB_CONNECT_TIMEOUT_SECONDS)
    except sqlite3.OperationalError as exc:
        message = _diagnose_db_open_error(path, exc)
        logger.error(message)
        raise sqlite3.OperationalError(message) from exc


def get_db_path() -> str:
    """Return the absolute path to the SQLite database file."""
    custom = os.environ.get("JOB_APPLY_AI_DB", "").strip()
    if custom:
        path = os.path.abspath(os.path.expanduser(custom))
        _ensure_db_parent_dir(path)
        return path

    _ensure_db_parent_dir(os.path.join(DEFAULT_DB_DIR, "jobs.db"))
    return os.path.join(DEFAULT_DB_DIR, "jobs.db")


def reset_init_cache() -> None:
    """Clear the init_db cache (used by tests)."""
    with _init_lock:
        _initialized_paths.clear()


def init_db(db_path: str | None = None) -> None:
    """Create tables if they do not exist."""
    path = os.path.abspath(db_path) if db_path else get_db_path()
    with _init_lock:
        if path in _initialized_paths:
            return

    _ensure_db_parent_dir(path)

    try:
        with _connect(path) as conn:
            _initialize_schema(conn)
    except sqlite3.OperationalError as exc:
        message = _diagnose_db_open_error(path, exc)
        logger.error(message)
        raise sqlite3.OperationalError(message) from exc
    except Exception as exc:
        message = _diagnose_db_open_error(path, exc)
        logger.error(message)
        raise sqlite3.OperationalError(message) from exc

    with _init_lock:
        _initialized_paths.add(path)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    """Apply schema migrations to an open SQLite connection."""
    with conn:
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

            CREATE TABLE IF NOT EXISTS batch_search_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                control TEXT,
                schedule_type TEXT NOT NULL DEFAULT 'once',
                titles_json TEXT NOT NULL DEFAULT '[]',
                locations_json TEXT NOT NULL DEFAULT '[]',
                shuffle_queue INTEGER NOT NULL DEFAULT 0,
                max_jobs INTEGER NOT NULL DEFAULT 5,
                sources TEXT NOT NULL DEFAULT 'linkedin-mcp,linkedin,adzuna,reed,indeed,totaljobs,cv-library,remoteok,arbeitnow',
                mode TEXT NOT NULL DEFAULT 'both',
                search_filters_json TEXT NOT NULL DEFAULT '{}',
                total_combinations INTEGER NOT NULL DEFAULT 0,
                current_index INTEGER NOT NULL DEFAULT 0,
                progress_percent INTEGER NOT NULL DEFAULT 0,
                progress_message TEXT NOT NULL DEFAULT '',
                progress_step TEXT NOT NULL DEFAULT '',
                last_error TEXT NOT NULL DEFAULT '',
                search_run_id INTEGER,
                task_id TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL DEFAULT '{}',
                next_run_at TEXT,
                last_run_at TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_batch_search_jobs_status
            ON batch_search_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_batch_search_jobs_task_id
            ON batch_search_jobs(task_id);
            CREATE INDEX IF NOT EXISTS idx_batch_search_jobs_next_run
            ON batch_search_jobs(next_run_at);

            CREATE TABLE IF NOT EXISTS ai_task_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                control TEXT,
                job_id INTEGER,
                payload_json TEXT NOT NULL DEFAULT '{}',
                progress_percent INTEGER NOT NULL DEFAULT 0,
                progress_message TEXT NOT NULL DEFAULT '',
                progress_step TEXT NOT NULL DEFAULT '',
                last_error TEXT NOT NULL DEFAULT '',
                task_id TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_ai_task_jobs_status
            ON ai_task_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_ai_task_jobs_task_id
            ON ai_task_jobs(task_id);
            CREATE INDEX IF NOT EXISTS idx_ai_task_jobs_task_type
            ON ai_task_jobs(task_type);

            CREATE TABLE IF NOT EXISTS urgent_task_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                control TEXT,
                job_id INTEGER,
                payload_json TEXT NOT NULL DEFAULT '{}',
                progress_percent INTEGER NOT NULL DEFAULT 0,
                progress_message TEXT NOT NULL DEFAULT '',
                progress_step TEXT NOT NULL DEFAULT '',
                last_error TEXT NOT NULL DEFAULT '',
                task_id TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_urgent_task_jobs_status
            ON urgent_task_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_urgent_task_jobs_task_id
            ON urgent_task_jobs(task_id);
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
    path = os.path.abspath(db_path) if db_path else get_db_path()
    init_db(path)
    conn = _connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
