"""Export and restore profile, jobs, settings, queue, logs, and file artifacts."""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, BinaryIO

from job_apply_ai.cv_modifier.cv_content_store import cv_content_path, save_cv_content
from job_apply_ai.storage.app_settings import AppSettingsRepository
from job_apply_ai.storage.database import get_connection
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.storage.user_profile import (
    UserProfileRepository,
    merge_profiles,
    normalize_profile,
    profile_to_export_dict,
)

logger = logging.getLogger(__name__)

BACKUP_FORMAT = "job_apply_ai_backup"
BACKUP_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
MANIFEST_NAME = "manifest.json"
CV_CONTENT_PREFIX = "cv_content/"
CV_DOCS_PREFIX = "documents/cv/"
COVER_LETTER_PREFIX = "documents/cover_letters/"
FILES_CVS_PREFIX = "files/cvs/"
FILES_JOBS_PREFIX = "files/jobs/"
FILES_PROJECT_CVS_PREFIX = "files/project_outputs/cvs/"
FILES_PROJECT_JOBS_PREFIX = "files/project_outputs/jobs/"

_JOB_JSON_FIELDS = ("matched_skills", "matched_categories")
_API_KEY_PROVIDERS = ("alibaba", "freellmapi")
_DOC_EXTENSIONS = (".docx", ".pdf", ".doc")


@dataclass(frozen=True)
class RestoreScope:
    """Which backup sections to restore."""

    include_task_queue: bool = True
    include_settings: bool = True
    include_all_others: bool = True

    def validate(self) -> None:
        if not (self.include_task_queue or self.include_settings or self.include_all_others):
            raise ValueError("Select at least one section to restore.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_project_outputs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def _resolve_backup_dirs(
    *,
    data_dir: str | None = None,
    cv_output_dir: str | None = None,
    jobs_output_dir: str | None = None,
) -> tuple[str, str]:
    if data_dir:
        return (
            os.path.join(data_dir, "cvs"),
            os.path.join(data_dir, "jobs"),
        )
    if cv_output_dir:
        cv_dir = cv_output_dir
        jobs_dir = jobs_output_dir or os.path.join(os.path.dirname(cv_output_dir), "jobs")
        return cv_dir, jobs_dir
    raise ValueError("data_dir or cv_output_dir is required")


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    for field in _JOB_JSON_FIELDS:
        raw = data.get(field) or ("[]" if field == "matched_skills" else "{}")
        try:
            data[field] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data[field] = [] if field == "matched_skills" else {}
    return data


def _dev_log_row_to_dict(row: Any) -> dict[str, Any]:
    try:
        payload = json.loads(row["data"] or "{}")
    except json.JSONDecodeError:
        payload = {"raw": row["data"]}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return {
        "id": row["id"],
        "category": row["category"],
        "agent": row["agent"],
        "event": row["event"],
        "message": row["message"],
        "data": payload,
        "task_id": row["task_id"] or "",
        "job_id": row["job_id"],
        "created_at": row["created_at"],
    }


def _settings_for_backup(settings: dict[str, Any]) -> dict[str, Any]:
    """Strip LLM provider API keys from settings before export."""
    sanitized = deepcopy(settings)
    for provider in _API_KEY_PROVIDERS:
        bucket = sanitized.get(provider)
        if isinstance(bucket, dict):
            bucket = dict(bucket)
            bucket["api_key"] = ""
            sanitized[provider] = bucket
    return sanitized


def _merge_settings_preserving_api_keys(
    incoming: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Restore settings while keeping locally stored API keys when backup omitted them."""
    merged = deepcopy(incoming)
    for provider in _API_KEY_PROVIDERS:
        incoming_bucket = merged.get(provider)
        current_bucket = current.get(provider)
        if not isinstance(incoming_bucket, dict):
            continue
        incoming_key = str(incoming_bucket.get("api_key") or "").strip()
        current_key = str((current_bucket or {}).get("api_key") or "").strip()
        if not incoming_key and current_key:
            incoming_bucket["api_key"] = current_key
    return AppSettingsRepository().save_llm_settings(merged)


def _fetch_search_runs(db_path: str | None = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, keyword, location, sources, mode, created_at FROM search_runs ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def _fetch_all_jobs(db_path: str | None = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()
    return [_row_to_dict(row) for row in rows]


def _fetch_all_dev_logs(db_path: str | None = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM dev_logs ORDER BY id").fetchall()
    return [_dev_log_row_to_dict(row) for row in rows]


def _fetch_batch_search_jobs(db_path: str | None = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM batch_search_jobs ORDER BY id").fetchall()
    jobs: list[dict[str, Any]] = []
    for row in rows:
        job = dict(row)
        job["titles"] = json.loads(job.pop("titles_json") or "[]")
        job["locations"] = json.loads(job.pop("locations_json") or "[]")
        job["search_filters"] = json.loads(job.pop("search_filters_json") or "{}")
        job["result"] = json.loads(job.pop("result_json") or "{}")
        job["shuffle_queue"] = bool(job.get("shuffle_queue"))
        jobs.append(job)
    return jobs


def _cv_filename_for_sidecar(cv_output_dir: str, sidecar_name: str) -> str:
    base = sidecar_name[: -len(".content.json")]
    for ext in _DOC_EXTENSIONS:
        candidate = f"{base}{ext}"
        if os.path.isfile(os.path.join(cv_output_dir, candidate)):
            return candidate
    return f"{base}.docx"


def _load_cv_sidecar(cv_output_dir: str, sidecar_path: str, cv_filename: str) -> dict[str, Any] | None:
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable CV sidecar %s: %s", sidecar_path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _collect_cv_content(cv_output_dir: str, jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collect all CV sidecar JSON payloads from the CV output directory."""
    cv_content: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    for job in jobs:
        cv_filename = str(job.get("cv_filename") or "").strip()
        if not cv_filename or cv_filename in seen:
            continue
        seen.add(cv_filename)
        sidecar_path = cv_content_path(cv_output_dir, cv_filename)
        if os.path.isfile(sidecar_path):
            payload = _load_cv_sidecar(cv_output_dir, sidecar_path, cv_filename)
            if payload is not None:
                cv_content[cv_filename] = payload

    if not os.path.isdir(cv_output_dir):
        return cv_content

    for name in os.listdir(cv_output_dir):
        if not name.endswith(".content.json"):
            continue
        cv_filename = _cv_filename_for_sidecar(cv_output_dir, name)
        if cv_filename in cv_content:
            continue
        sidecar_path = os.path.join(cv_output_dir, name)
        payload = _load_cv_sidecar(cv_output_dir, sidecar_path, cv_filename)
        if payload is not None:
            cv_content[cv_filename] = payload

    return cv_content


def _collect_directory_files(base_dir: str, zip_prefix: str) -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []
    if not os.path.isdir(base_dir):
        return documents
    for root, _, files in os.walk(base_dir):
        for name in files:
            disk_path = os.path.join(root, name)
            rel = os.path.relpath(disk_path, base_dir).replace("\\", "/")
            documents.append((f"{zip_prefix}{rel}", disk_path))
    return documents


def _collect_legacy_documents(
    cv_output_dir: str,
    jobs: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Legacy v1 document paths for backward-compatible archives."""
    documents: list[tuple[str, str]] = []
    seen_cv: set[str] = set()
    seen_cl: set[str] = set()
    for job in jobs:
        cv_filename = str(job.get("cv_filename") or "").strip()
        if cv_filename and cv_filename not in seen_cv:
            seen_cv.add(cv_filename)
            cv_path = os.path.join(cv_output_dir, cv_filename)
            if os.path.isfile(cv_path):
                documents.append((f"{CV_DOCS_PREFIX}{cv_filename}", cv_path))

        cl_filename = str(job.get("cover_letter_filename") or "").strip()
        if cl_filename and cl_filename not in seen_cl:
            seen_cl.add(cl_filename)
            cl_path = os.path.join(cv_output_dir, cl_filename)
            if os.path.isfile(cl_path):
                documents.append((f"{COVER_LETTER_PREFIX}{cl_filename}", cl_path))
    return documents


def build_manifest(
    *,
    profile: dict[str, Any],
    search_runs: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    cv_content: dict[str, dict[str, Any]],
    app_settings: dict[str, Any],
    dev_logs: list[dict[str, Any]],
    batch_search_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    export_profile = profile_to_export_dict(profile)
    return {
        "format": BACKUP_FORMAT,
        "version": BACKUP_VERSION,
        "exported_at": _utc_now(),
        "profile": export_profile["profile"],
        "search_runs": search_runs,
        "jobs": jobs,
        "cv_content": cv_content,
        "app_settings": app_settings,
        "dev_logs": dev_logs,
        "batch_search_jobs": batch_search_jobs,
    }


def export_backup(
    cv_output_dir: str | None = None,
    db_path: str | None = None,
    *,
    data_dir: str | None = None,
    jobs_output_dir: str | None = None,
) -> io.BytesIO:
    """Export profile, database state, and on-disk artifacts to an in-memory zip."""
    cv_dir, jobs_dir = _resolve_backup_dirs(
        data_dir=data_dir,
        cv_output_dir=cv_output_dir,
        jobs_output_dir=jobs_output_dir,
    )

    profile = UserProfileRepository().get_profile()
    search_runs = _fetch_search_runs(db_path)
    jobs = _fetch_all_jobs(db_path)
    cv_content = _collect_cv_content(cv_dir, jobs)
    app_settings = _settings_for_backup(AppSettingsRepository().get_settings())
    dev_logs = _fetch_all_dev_logs(db_path)
    batch_search_jobs = _fetch_batch_search_jobs(db_path)

    manifest = build_manifest(
        profile=profile,
        search_runs=search_runs,
        jobs=jobs,
        cv_content=cv_content,
        app_settings=app_settings,
        dev_logs=dev_logs,
        batch_search_jobs=batch_search_jobs,
    )

    file_documents = _collect_directory_files(cv_dir, FILES_CVS_PREFIX)
    file_documents.extend(_collect_directory_files(jobs_dir, FILES_JOBS_PREFIX))

    project_outputs = _default_project_outputs_dir()
    file_documents.extend(
        _collect_directory_files(
            os.path.join(project_outputs, "cvs"),
            FILES_PROJECT_CVS_PREFIX,
        )
    )
    file_documents.extend(
        _collect_directory_files(
            os.path.join(project_outputs, "jobs"),
            FILES_PROJECT_JOBS_PREFIX,
        )
    )
    legacy_documents = _collect_legacy_documents(cv_dir, jobs)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            MANIFEST_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        seen_paths: set[str] = set()
        for zip_path, disk_path in (*file_documents, *legacy_documents):
            if zip_path in seen_paths:
                continue
            seen_paths.add(zip_path)
            archive.write(disk_path, zip_path)

    buffer.seek(0)
    logger.info(
        "Exported backup with %s jobs, %s search runs, %s CV sidecars, "
        "%s dev logs, %s batch queue jobs, %s files",
        len(jobs),
        len(search_runs),
        len(cv_content),
        len(dev_logs),
        len(batch_search_jobs),
        len(file_documents),
    )
    return buffer


def _parse_manifest(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Backup manifest must be a JSON object.")
    if raw.get("format") != BACKUP_FORMAT:
        raise ValueError("Unrecognized backup format.")
    version = raw.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported backup version: {version!r}")
    if not isinstance(raw.get("profile"), dict):
        raise ValueError("Backup manifest is missing profile data.")
    if not isinstance(raw.get("jobs"), list):
        raise ValueError("Backup manifest is missing jobs data.")
    if not isinstance(raw.get("search_runs"), list):
        raise ValueError("Backup manifest is missing search run history.")
    cv_content = raw.get("cv_content", {})
    if not isinstance(cv_content, dict):
        raise ValueError("Backup cv_content must be an object.")
    for key in ("app_settings", "dev_logs", "batch_search_jobs"):
        if key not in raw:
            raw[key] = {} if key == "app_settings" else []
        elif key == "app_settings" and not isinstance(raw[key], dict):
            raise ValueError("Backup app_settings must be an object.")
        elif key != "app_settings" and not isinstance(raw[key], list):
            raise ValueError(f"Backup {key} must be a list.")
    return raw


def _clear_jobs_and_search_runs(db_path: str | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM search_runs")


def _clear_dev_logs(db_path: str | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM dev_logs")


def _clear_batch_search_jobs(db_path: str | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM batch_search_jobs")


def _restore_search_runs(
    search_runs: list[dict[str, Any]],
    db_path: str | None = None,
) -> dict[int, int]:
    id_map: dict[int, int] = {}
    with get_connection(db_path) as conn:
        for run in search_runs:
            old_id = int(run["id"])
            cursor = conn.execute(
                """
                INSERT INTO search_runs (keyword, location, sources, mode, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run.get("keyword", ""),
                    run.get("location", ""),
                    run.get("sources", ""),
                    run.get("mode", ""),
                    run.get("created_at"),
                ),
            )
            id_map[old_id] = int(cursor.lastrowid)
    return id_map


def _restore_dev_logs(
    dev_logs: list[dict[str, Any]],
    db_path: str | None = None,
) -> int:
    if not dev_logs:
        return 0
    with get_connection(db_path) as conn:
        for entry in dev_logs:
            conn.execute(
                """
                INSERT INTO dev_logs (
                    category, agent, event, message, data, task_id, job_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.get("category", "system"),
                    entry.get("agent", ""),
                    entry.get("event", ""),
                    entry.get("message", ""),
                    json.dumps(entry.get("data") or {}, ensure_ascii=False),
                    entry.get("task_id", ""),
                    entry.get("job_id"),
                    entry.get("created_at"),
                ),
            )
    return len(dev_logs)


def _restore_batch_search_jobs(
    batch_jobs: list[dict[str, Any]],
    db_path: str | None = None,
) -> int:
    if not batch_jobs:
        return 0
    with get_connection(db_path) as conn:
        for job in batch_jobs:
            conn.execute(
                """
                INSERT INTO batch_search_jobs (
                    name, status, control, schedule_type, titles_json, locations_json,
                    shuffle_queue, max_jobs, sources, mode, search_filters_json,
                    total_combinations, current_index, progress_percent, progress_message,
                    progress_step, last_error, search_run_id, task_id, result_json,
                    next_run_at, last_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.get("name", ""),
                    job.get("status", "pending"),
                    job.get("control"),
                    job.get("schedule_type", "once"),
                    json.dumps(job.get("titles") or []),
                    json.dumps(job.get("locations") or []),
                    int(bool(job.get("shuffle_queue"))),
                    int(job.get("max_jobs") or 5),
                    job.get("sources", ""),
                    job.get("mode", "both"),
                    json.dumps(job.get("search_filters") or {}),
                    int(job.get("total_combinations") or 0),
                    int(job.get("current_index") or 0),
                    int(job.get("progress_percent") or 0),
                    job.get("progress_message", ""),
                    job.get("progress_step", ""),
                    job.get("last_error", ""),
                    job.get("search_run_id"),
                    job.get("task_id", ""),
                    json.dumps(job.get("result") or {}),
                    job.get("next_run_at"),
                    job.get("last_run_at"),
                    job.get("created_at"),
                    job.get("updated_at"),
                ),
            )
    return len(batch_jobs)


def _job_for_restore(
    job: dict[str, Any],
    search_run_id_map: dict[int, int],
    *,
    preserve_search_runs: bool,
) -> dict[str, Any]:
    restored = {key: job.get(key, "") for key in job if key not in ("id", "dedupe_key")}
    if preserve_search_runs:
        old_search_run_id = job.get("search_run_id")
        if old_search_run_id is not None:
            restored["search_run_id"] = search_run_id_map.get(int(old_search_run_id))
    else:
        restored.pop("search_run_id", None)
    for field in _JOB_JSON_FIELDS:
        if field in job:
            restored[field] = job[field]
    return restored


def _extract_archive_file(archive: zipfile.ZipFile, zip_path: str, target_path: str) -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with archive.open(zip_path) as source, open(target_path, "wb") as handle:
        handle.write(source.read())


def _restore_data_files(
    archive: zipfile.ZipFile,
    cv_output_dir: str,
    jobs_output_dir: str,
) -> int:
    restored = 0
    project_outputs = _default_project_outputs_dir()
    project_cv_dir = os.path.join(project_outputs, "cvs")
    project_jobs_dir = os.path.join(project_outputs, "jobs")
    prefix_targets = (
        (FILES_CVS_PREFIX, cv_output_dir),
        (FILES_JOBS_PREFIX, jobs_output_dir),
        (FILES_PROJECT_CVS_PREFIX, project_cv_dir),
        (FILES_PROJECT_JOBS_PREFIX, project_jobs_dir),
    )
    for name in archive.namelist():
        if name.endswith("/"):
            continue
        for prefix, target_root in prefix_targets:
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix) :]
            target = os.path.join(target_root, rel)
            _extract_archive_file(archive, name, target)
            restored += 1
            break
    return restored


def _restore_cv_artifacts(
    cv_output_dir: str,
    jobs: list[dict[str, Any]],
    cv_content: dict[str, dict[str, Any]],
    archive: zipfile.ZipFile | None = None,
) -> int:
    os.makedirs(cv_output_dir, exist_ok=True)
    restored = 0

    if archive is not None:
        for name in archive.namelist():
            if name.startswith(CV_DOCS_PREFIX) and not name.endswith("/"):
                filename = name[len(CV_DOCS_PREFIX) :]
                target = os.path.join(cv_output_dir, filename)
                if not os.path.isfile(target):
                    _extract_archive_file(archive, name, target)
            elif name.startswith(COVER_LETTER_PREFIX) and not name.endswith("/"):
                filename = name[len(COVER_LETTER_PREFIX) :]
                target = os.path.join(cv_output_dir, filename)
                if not os.path.isfile(target):
                    _extract_archive_file(archive, name, target)

    restored_keys: set[str] = set()
    for cv_filename, payload in cv_content.items():
        if not isinstance(payload, dict) or cv_filename in restored_keys:
            continue
        save_cv_content(
            cv_output_dir,
            cv_filename,
            payload.get("tailored_content", {}),
            chat_history=payload.get("chat_history"),
            cover_letter=payload.get("cover_letter"),
            cover_letter_chat_history=payload.get("cover_letter_chat_history"),
            store=payload,
        )
        restored_keys.add(cv_filename)
        restored += 1

    for job in jobs:
        cv_filename = str(job.get("cv_filename") or "").strip()
        if not cv_filename or cv_filename in restored_keys:
            continue
        payload = cv_content.get(cv_filename)
        if isinstance(payload, dict):
            continue
        if archive is not None:
            cv_zip_path = f"{CV_DOCS_PREFIX}{cv_filename}"
            if cv_zip_path in archive.namelist():
                target = os.path.join(cv_output_dir, cv_filename)
                if not os.path.isfile(target):
                    _extract_archive_file(archive, cv_zip_path, target)

            cl_filename = str(job.get("cover_letter_filename") or "").strip()
            if cl_filename:
                cl_zip_path = f"{COVER_LETTER_PREFIX}{cl_filename}"
                if cl_zip_path in archive.namelist():
                    target = os.path.join(cv_output_dir, cl_filename)
                    if not os.path.isfile(target):
                        _extract_archive_file(archive, cl_zip_path, target)

    return restored


def restore_backup(
    backup: str | bytes | BinaryIO,
    cv_output_dir: str | None = None,
    *,
    replace: bool = False,
    merge_profile: bool = True,
    include_task_queue: bool = True,
    include_settings: bool = True,
    include_all_others: bool = True,
    db_path: str | None = None,
    data_dir: str | None = None,
    jobs_output_dir: str | None = None,
) -> dict[str, int]:
    """Restore profile, database state, and on-disk artifacts from a backup zip."""
    scope = RestoreScope(
        include_task_queue=include_task_queue,
        include_settings=include_settings,
        include_all_others=include_all_others,
    )
    scope.validate()

    cv_dir, jobs_dir = _resolve_backup_dirs(
        data_dir=data_dir,
        cv_output_dir=cv_output_dir,
        jobs_output_dir=jobs_output_dir,
    )
    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)

    if isinstance(backup, str):
        with zipfile.ZipFile(backup, "r") as archive:
            return _restore_from_archive(
                archive,
                cv_dir,
                jobs_dir,
                replace=replace,
                merge_profile=merge_profile,
                scope=scope,
                db_path=db_path,
            )

    payload = backup.read() if hasattr(backup, "read") else backup
    try:
        archive_ctx = zipfile.ZipFile(io.BytesIO(payload), "r")
    except zipfile.BadZipFile as exc:
        raise ValueError("Backup file is not a valid zip archive.") from exc
    with archive_ctx as archive:
        return _restore_from_archive(
            archive,
            cv_dir,
            jobs_dir,
            replace=replace,
            merge_profile=merge_profile,
            scope=scope,
            db_path=db_path,
        )


def _restore_from_archive(
    archive: zipfile.ZipFile,
    cv_output_dir: str,
    jobs_output_dir: str,
    *,
    replace: bool,
    merge_profile: bool,
    scope: RestoreScope,
    db_path: str | None,
) -> dict[str, int]:
    if MANIFEST_NAME not in archive.namelist():
        raise ValueError(f"Backup zip is missing {MANIFEST_NAME}.")

    manifest = _parse_manifest(json.loads(archive.read(MANIFEST_NAME).decode("utf-8")))
    profile_repo = UserProfileRepository()
    job_repo = JobRepository()
    settings_repo = AppSettingsRepository()

    if replace and scope.include_all_others:
        _clear_jobs_and_search_runs(db_path)
        _clear_dev_logs(db_path)
    if replace and scope.include_task_queue:
        _clear_batch_search_jobs(db_path)

    search_run_id_map: dict[int, int] = {}
    dev_logs_restored = 0
    if scope.include_all_others:
        if replace:
            search_run_id_map = _restore_search_runs(manifest["search_runs"], db_path)
        dev_logs_restored = _restore_dev_logs(manifest.get("dev_logs", []), db_path)

    batch_jobs_restored = 0
    if scope.include_task_queue:
        batch_jobs_restored = _restore_batch_search_jobs(
            manifest.get("batch_search_jobs", []),
            db_path,
        )

    settings_restored = 0
    if scope.include_settings:
        incoming_settings = manifest.get("app_settings") or {}
        if incoming_settings:
            _merge_settings_preserving_api_keys(incoming_settings, settings_repo.get_settings())
            settings_restored = 1

    restored_job_ids: list[int] = []
    files_restored = 0
    restored_sidecars = 0
    if scope.include_all_others:
        jobs = manifest["jobs"]
        restore_jobs = [
            _job_for_restore(job, search_run_id_map, preserve_search_runs=replace)
            for job in jobs
        ]
        restored_job_ids = job_repo.upsert_jobs(restore_jobs)

        incoming_profile = normalize_profile(manifest["profile"])
        if replace or not merge_profile:
            profile_repo.save_profile(incoming_profile)
        else:
            current = profile_repo.get_profile()
            merged, _changes = merge_profiles(current, incoming_profile)
            profile_repo.save_profile(merged)

        files_restored = _restore_data_files(archive, cv_output_dir, jobs_output_dir)
        cv_content = manifest.get("cv_content", {})
        restored_sidecars = _restore_cv_artifacts(cv_output_dir, jobs, cv_content, archive)

    stats = {
        "jobs_restored": len(restored_job_ids),
        "search_runs_restored": len(search_run_id_map),
        "cv_sidecars_restored": restored_sidecars,
        "dev_logs_restored": dev_logs_restored,
        "batch_jobs_restored": batch_jobs_restored,
        "files_restored": files_restored,
        "settings_restored": settings_restored,
    }
    logger.info("Restored backup: %s", stats)
    return stats


def backup_filename(profile: dict[str, Any] | None = None) -> str:
    name = "profile"
    if profile and profile.get("full_name"):
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in profile["full_name"])
        safe = safe.strip("_") or "profile"
        name = safe
    today = datetime.today().strftime("%Y-%m-%d")
    return f"{name}_backup_{today}.zip"
