"""
Web Interface for Job Application AI Agent

This module provides a Flask web application for the job application AI agent.
"""

import os
import json
import logging
import secrets
import tempfile
from datetime import datetime
from typing import Any
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify, Response
import zipfile
import io

from job_apply_ai.job_sources import (
    UI_DEFAULT_JOB_SOURCES,
    UI_JOB_SOURCE_OPTIONS,
    format_sources_csv,
    job_source_options_for_ui,
    parse_sources_csv,
    selected_source_ids_from_csv,
)
from job_apply_ai.scraper.aggregator import search_jobs as aggregate_search_jobs
from job_apply_ai.scraper.linkedin_job_url import parse_linkedin_job_url
from job_apply_ai.scraper.linkedin_mcp_client import LinkedInMcpError, check_linkedin_mcp_health
from job_apply_ai.scraper.linkedin_profile_parser import fetch_linkedin_profile
from job_apply_ai.scraper.linkedin_profile_sync import (
    apply_sync_action,
    compare_profiles,
    diff_summary,
)
from job_apply_ai.scraper.search_filters import SearchFilters
from job_apply_ai.batch_search import (
    build_search_queue,
    decode_uploaded_text,
    parse_lines,
    shuffle_search_queue,
    validate_batch_queue,
)
from job_apply_ai.job_schema import JOB_COLUMNS
from job_apply_ai.job_sort import (
    DEFAULT_JOB_SORT,
    JOB_SORT_OPTIONS,
    get_profile_match_analysis,
    get_profile_match_score,
    sort_jobs,
    validate_job_sort,
)
from job_apply_ai.job_status import (
    DEFAULT_JOB_STATUS,
    JOB_STATUS_BADGE_CLASSES,
    JOB_STATUS_ICONS,
    JOB_STATUS_LABELS,
    JOB_WORKFLOW_STATUSES,
    is_valid_job_status,
    job_status_label,
)
from job_apply_ai.job_move_history import (
    can_redo_job_moves,
    can_undo_job_moves,
    pop_redo_job_moves,
    pop_undo_job_moves,
    record_job_moves,
    redo_job_move_label,
    undo_job_move_label,
)
from job_apply_ai.cv_modifier.cv_analyzer import CVAnalyzer
from job_apply_ai.cv_modifier.job_match_analyzer import (
    NOT_MATCH_STATUS,
    classify_jobs_by_profile_match,
    analyze_jobs_with_threshold,
    heuristic_job_match,
    job_meets_threshold,
    normalize_min_match_score,
    profile_has_matchable_skills,
)
from job_apply_ai.cv_modifier.job_title_suggester import suggest_job_titles
from job_apply_ai.cv_modifier.cover_letter_builder import CoverLetterBuilder
from job_apply_ai.cv_modifier.cover_letter_chat_editor import CoverLetterChatEditor
from job_apply_ai.cv_modifier.cover_letter_generator import CoverLetterGenerator
from job_apply_ai.cv_modifier.chat_context import (
    cv_content_to_preview_lines,
    normalize_preview_lines,
    preview_lines_to_content,
    resolve_cv_preview_lines,
    resolve_effective_tailored_content,
)
from job_apply_ai.cv_modifier.docx_builder import CVDocumentBuilder
from job_apply_ai.cv_modifier.pdf_builder import build_cover_letter_pdf, build_cv_pdf, pdf_path_for_docx
from job_apply_ai.cv_modifier.ats_friendly_analyzer import (
    ATSFriendlyAnalyzer,
    get_suggestion,
    normalize_ats_analysis,
    pending_suggestions,
    replace_suggestion,
    update_suggestion_status,
    update_suggestions_status,
)
from job_apply_ai.cv_modifier.cv_chat_editor import CVChatEditor
from job_apply_ai.cv_modifier.cv_ask_assistant import CVAskAssistant
from job_apply_ai.cv_modifier.cv_content_store import (
    append_active_chat_messages,
    delete_cv_artifacts,
    get_active_chat_messages,
    get_active_chat_session_id,
    get_chat_sessions,
    load_cv_content,
    normalize_store,
    save_cv_content,
    set_active_chat_session,
    start_chat_session,
)
from job_apply_ai.cv_modifier.cv_generator import RAGCVGenerator
from job_apply_ai.cv_workflows import BATCH_ATS_FRIENDLY_PASSES
from job_apply_ai.ui.cv_tasks import (
    complete_task,
    create_task,
    fail_task,
    get_task,
    pause_task,
    request_task_stop,
    resume_task,
    start_background_task,
    task_control_checkpoint,
    TaskStopped,
    update_task,
)
from job_apply_ai.paths import get_data_dir
from job_apply_ai.utils.helpers import ensure_directory_exists, sanitize_filename
from job_apply_ai.storage.database import init_db
from job_apply_ai.storage.batch_queue_repository import (
    BatchQueueRepository,
    SCHEDULE_LABELS,
    STATUS_LABELS,
    TERMINAL_STATUSES,
    to_task_snapshot,
)
from job_apply_ai.storage.ai_task_queue_repository import (
    AiTaskQueueRepository,
    AI_STATUS_LABELS,
    AI_TASK_TYPE_LABELS,
    CONTROLLABLE_AI_TASK_TYPES,
    to_ai_task_snapshot,
)
from job_apply_ai.storage.urgent_task_queue_repository import (
    UrgentTaskQueueRepository,
    CONTROLLABLE_URGENT_TASK_TYPES,
    to_urgent_task_snapshot,
)
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.storage.user_profile import (
    UserProfileRepository,
    get_default_cv_template_path,
    import_has_changes,
    merge_profiles,
    profile_from_export_dict,
    profile_from_form,
    profile_is_ready,
    profile_to_export_dict,
    profile_to_form_fields,
    remove_smtp_account,
    set_default_smtp_account,
    summarize_import_changes,
    update_smtp_account_tokens,
    upsert_oauth_smtp_account,
)
from job_apply_ai.storage.app_settings import (
    AppSettingsRepository,
    ensure_alibaba_rotation_pools,
    llm_settings_from_form,
    uses_alibaba_provider,
    uses_freellmapi_provider,
    worker_settings_from_form,
)
from job_apply_ai.storage.dev_log import DEV_LOG_CATEGORIES, DevLogRepository
from job_apply_ai.dev_logging import dev_agent, dev_task, dev_llm_context, invalidate_dev_mode_cache
from job_apply_ai.cv_modifier.alibaba_client import AlibabaClient, KNOWN_MODELS
from job_apply_ai.cv_modifier.freellmapi_client import AUTO_MODEL, FreeLLMAPIClient
from job_apply_ai.cv_modifier.ollama_client import OllamaClient
from job_apply_ai.storage.exports import export_jobs
from job_apply_ai.email.application_mailer import (
    build_application_body,
    build_application_subject,
    get_send_account,
    list_smtp_accounts,
    parse_recipient_emails,
    send_application_email,
    smtp_is_configured,
)
from job_apply_ai.email.oauth_google import exchange_google_code, google_authorization_url
from job_apply_ai.email.oauth_microsoft import exchange_microsoft_code, microsoft_authorization_url
from job_apply_ai.email.oauth_settings import (
    google_oauth_configured,
    google_oauth_settings,
    microsoft_oauth_configured,
    microsoft_oauth_settings,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['UPLOAD_FOLDER'] = get_data_dir()
ensure_directory_exists(app.config['UPLOAD_FOLDER'])

# Create output directories
app.config['CV_OUTPUT_DIR'] = os.path.join(app.config['UPLOAD_FOLDER'], 'cvs')
app.config['JOBS_OUTPUT_DIR'] = os.path.join(app.config['UPLOAD_FOLDER'], 'jobs')
ensure_directory_exists(app.config['CV_OUTPUT_DIR'])
ensure_directory_exists(app.config['JOBS_OUTPUT_DIR'])

# Initialize SQLite database
init_db()
job_repo = JobRepository()
batch_queue_repo = BatchQueueRepository()
ai_task_queue_repo = AiTaskQueueRepository()
urgent_task_queue_repo = UrgentTaskQueueRepository()
profile_repo = UserProfileRepository()
app_settings_repo = AppSettingsRepository()

# Clear abandoned background tasks after this many seconds without progress
BACKGROUND_TASK_STALE_SECONDS = 900
BACKGROUND_TASK_SESSION_KEYS = (
    'cv_generation_active',
    'ats_friendly_active',
    'batch_ats_friendly_active',
    'single_search_active',
    'batch_search_active',
    'job_match_analyze_active',
)

# Ensure session data is saved
app.config['SESSION_TYPE'] = 'filesystem'


def _resolve_task(task_id: str) -> dict | None:
    """Return an in-memory task or a queue-backed task snapshot."""
    task = get_task(task_id)
    if task:
        return task
    queue_job = batch_queue_repo.get_job_by_task_id(task_id)
    if queue_job:
        return to_task_snapshot(queue_job)
    ai_job = ai_task_queue_repo.get_job_by_task_id(task_id)
    if ai_job:
        return to_ai_task_snapshot(ai_job)
    urgent_job = urgent_task_queue_repo.get_job_by_task_id(task_id)
    if urgent_job:
        return to_urgent_task_snapshot(urgent_job)
    return None


def _enqueue_urgent_task(
    task_type: str,
    payload: dict,
    *,
    job_id: int | None = None,
) -> str:
    """Enqueue an urgent UI I/O task for the urgent-worker process."""
    job = urgent_task_queue_repo.create_job(
        task_type=task_type,
        payload=payload,
        job_id=job_id,
    )
    return job['task_id']


def _enqueue_ai_task(
    task_type: str,
    payload: dict,
    *,
    job_id: int | None = None,
) -> str:
    """Enqueue an AI task for the separate ai-worker process."""
    job = ai_task_queue_repo.create_job(
        task_type=task_type,
        payload=payload,
        job_id=job_id,
    )
    return job['task_id']


def _ai_cv_complete_url(task: dict) -> str:
    task_id = task['task_id']
    if task.get('task_type') == 'single_cv':
        return url_for('make_cv_complete', task_id=task_id)
    meta = task.get('meta') or {}
    if meta.get('selected'):
        return url_for('batch_make_cvs_complete', task_id=task_id)
    return url_for('make_all_cvs_complete', task_id=task_id)


def _ai_cv_back_context(task: dict) -> tuple[str, str]:
    meta = task.get('meta') or {}
    if task.get('task_type') == 'single_cv':
        payload = ai_task_queue_repo.get_job_by_task_id(task['task_id'])
        payload_data = (payload or {}).get('payload') or {}
        return (
            _cv_generation_back_url(
                bool(payload_data.get('return_from_manage')),
                payload_data.get('return_folder', 'all'),
                payload_data.get('return_search', ''),
                payload_data.get('return_sort', ''),
            ),
            'Back to Jobs',
        )
    back_url = session.get('batch_cv_back_url') or url_for('job_list')
    back_label = session.get('batch_cv_back_label') or 'Back to Job List'
    return back_url, back_label


def _ai_task_progress_url(task: dict) -> str | None:
    task_id = task.get('task_id')
    if not task_id:
        return None
    task_type = task.get('task_type')
    if task_type in ('single_cv', 'batch_cv'):
        return url_for('ai_cv_task_progress', task_id=task_id)
    if task_type == 'job_match_analyze':
        return url_for('job_match_analyze_progress', task_id=task_id)
    if task_type == 'batch_ats_friendly':
        return url_for('batch_ats_friendly_progress', task_id=task_id)
    if task_type == 'ats_friendly':
        job_id = task.get('job_id')
        if job_id:
            return url_for('ats_friendly_progress', job_id=job_id)
    if task_type == 'profile_import':
        return url_for('profile_import_progress', task_id=task_id)
    if task_type == 'single_search':
        return url_for('single_search_progress', task_id=task_id)
    if task_type == 'batch_search':
        return url_for('batch_search_progress', task_id=task_id)
    if task_type == 'linkedin_job_import':
        return url_for('linkedin_job_import_progress', task_id=task_id)
    return None


def _sync_queue_task(task_id: str) -> str | None:
    """Return task_id if a queue-backed task is still active."""
    batch_job = batch_queue_repo.get_job_by_task_id(task_id)
    if batch_job:
        if batch_job['status'] in TERMINAL_STATUSES:
            return None
        if batch_job['status'] in ('pending', 'running', 'paused'):
            return task_id
        return None

    ai_job = ai_task_queue_repo.get_job_by_task_id(task_id)
    if ai_job:
        if ai_job['status'] in TERMINAL_STATUSES:
            return None
        if ai_job['status'] in ('pending', 'running', 'paused'):
            return task_id

    urgent_job = urgent_task_queue_repo.get_job_by_task_id(task_id)
    if urgent_job:
        if urgent_job['status'] in TERMINAL_STATUSES:
            return None
        if urgent_job['status'] in ('pending', 'running', 'paused'):
            return task_id
    return None


def _sync_session_background_task(session_key: str) -> str | None:
    """Drop stale session task pointers and return the task id only if still active."""
    task_id = session.get(session_key)
    if not task_id:
        return None

    if session_key == 'batch_search_active':
        active = _sync_queue_task(task_id)
        if active:
            return active
        session.pop(session_key, None)
        return None

    if session_key in (
        'cv_generation_active',
        'ats_friendly_active',
        'batch_ats_friendly_active',
        'job_match_analyze_active',
        'single_search_active',
    ):
        active = _sync_queue_task(task_id)
        if active:
            return active
        session.pop(session_key, None)
        return None

    task = get_task(task_id)
    if not task:
        session.pop(session_key, None)
        return None

    status = task.get('status')
    if status in ('complete', 'error'):
        session.pop(session_key, None)
        return None

    if status in ('pending', 'running', 'paused'):
        if status != 'paused':
            updated_at = task.get('updated_at')
            if updated_at:
                try:
                    last_update = datetime.fromisoformat(updated_at)
                    age_seconds = (datetime.utcnow() - last_update).total_seconds()
                    if age_seconds > BACKGROUND_TASK_STALE_SECONDS:
                        fail_task(task_id, 'Background task timed out or was interrupted.')
                        session.pop(session_key, None)
                        logger.warning('Cleared stale background task %s (%s)', task_id, session_key)
                        return None
                except ValueError:
                    pass
        return task_id

    session.pop(session_key, None)
    return None


def _sync_cv_generation_lock() -> str | None:
    """Drop stale CV generation locks and return the task id only if generation is active."""
    return _sync_session_background_task('cv_generation_active')


def _background_task_progress_label(task: dict) -> str:
    labels = {
        'single_search': 'Job search',
        'batch_search': 'Batch job search',
        'job_match_analyze': 'Analyzing Profile Match',
        'single_cv': 'Generating AI CV',
        'batch_cv': 'Generating AI CVs',
        'ats_friendly': 'ATS Friendly Analysis',
        'batch_ats_friendly': 'Batch ATS Optimization',
        'profile_import': 'Profile import',
        'linkedin_job_import': 'LinkedIn job import',
    }
    return labels.get(task.get('task_type', ''), 'Background task')


def _background_task_progress_url(task_id: str, task: dict) -> str | None:
    task_type = task.get('task_type')
    if task_type == 'single_search':
        return url_for('single_search_progress', task_id=task_id)
    if task_type == 'batch_search':
        return url_for('batch_search_progress', task_id=task_id)
    if task_type == 'job_match_analyze':
        return url_for('job_match_analyze_progress', task_id=task_id)
    if task_type == 'single_cv':
        return url_for('ai_cv_task_progress', task_id=task_id)
    if task_type == 'batch_cv':
        return url_for('ai_cv_task_progress', task_id=task_id)
    if task_type == 'ats_friendly':
        job_id = task.get('job_id')
        if job_id:
            return url_for('ats_friendly_progress', job_id=job_id)
    if task_type == 'batch_ats_friendly':
        return url_for('batch_ats_friendly_progress', task_id=task_id)
    return None


def _active_background_tasks() -> list[dict]:
    """Build UI entries for in-progress background tasks the user can return to."""
    active_tasks: list[dict] = []
    for session_key in BACKGROUND_TASK_SESSION_KEYS:
        task_id = session.get(session_key)
        if not task_id:
            continue
        task = _resolve_task(task_id)
        if not task or task.get('status') not in ('pending', 'running', 'paused'):
            continue
        progress_url = _background_task_progress_url(task_id, task)
        if not progress_url:
            continue
        entry = {
            'label': _background_task_progress_label(task),
            'progress_url': progress_url,
            'message': task.get('message', ''),
        }
        active_tasks.append(entry)
    return active_tasks


@app.before_request
def _refresh_background_task_sessions():
    for session_key in BACKGROUND_TASK_SESSION_KEYS:
        _sync_session_background_task(session_key)


@app.context_processor
def _inject_cv_generation_lock():
    profile = profile_repo.get_profile()
    accounts = list_smtp_accounts(profile)
    return {
        'cv_generation_active': session.get('cv_generation_active'),
        'ats_friendly_active': session.get('ats_friendly_active'),
        'active_background_tasks': _active_background_tasks(),
        'profile_ready': profile_is_ready(profile),
        'smtp_configured': bool(accounts),
        'smtp_accounts': accounts,
        'google_oauth_configured': google_oauth_configured(),
        'microsoft_oauth_configured': microsoft_oauth_configured(),
        'job_sort_options': JOB_SORT_OPTIONS,
        'default_job_sort': DEFAULT_JOB_SORT,
        'job_statuses': JOB_WORKFLOW_STATUSES,
        'status_labels': JOB_STATUS_LABELS,
        'status_icons': JOB_STATUS_ICONS,
        'status_badges': JOB_STATUS_BADGE_CLASSES,
        'dev_mode': app_settings_repo.get_dev_mode(),
        'default_job_sources': UI_DEFAULT_JOB_SOURCES,
        'job_source_options': job_source_options_for_ui(),
        'selected_source_ids_from_csv': selected_source_ids_from_csv,
    }


def _parse_job_sources_from_form() -> str:
    """Parse checkbox (or legacy text) job source fields into a CSV string."""
    if request.form.get('job_sources_field') == '1':
        selected = request.form.getlist('sources')
        valid = [source for source in selected if source in UI_JOB_SOURCE_OPTIONS]
        if not valid:
            raise ValueError('Select at least one job source.')
        return format_sources_csv(valid)

    legacy = (request.form.get('sources') or UI_DEFAULT_JOB_SOURCES).strip()
    parsed = parse_sources_csv(legacy)
    if not parsed:
        raise ValueError('Select at least one job source.')
    return format_sources_csv(parsed)


def _oauth_redirect_uri(endpoint: str) -> str:
    provider = 'google' if 'google' in endpoint else 'microsoft'
    env_key = f"{provider.upper()}_OAUTH_REDIRECT_URI"
    override = os.environ.get(env_key, '').strip()
    if override:
        return override
    return url_for(endpoint, _external=True)


def _enrich_jobs_with_skills(jobs: list[dict]) -> list[dict]:
    """Extract matched skills from job descriptions."""
    analyzer = CVAnalyzer()
    enriched = []
    for job in jobs:
        if job.get('description'):
            matched_skills, _, matched_categories = analyzer.extract_skills_from_description(
                job['description']
            )
            job['matched_skills'] = matched_skills
            job['matched_categories'] = matched_categories
        enriched.append(job)
    return enriched


def _cv_output_path(job: dict) -> tuple[str, str]:
    """Build output filename and full path for a generated CV."""
    today_date = datetime.today().strftime("%Y-%m-%d")
    safe_company = sanitize_filename(job.get('company', 'Company'))
    safe_title = sanitize_filename(job.get('title', 'Role'))
    output_filename = f"CV_{today_date}_{safe_company}_{safe_title}.docx"
    output_path = os.path.join(app.config['CV_OUTPUT_DIR'], output_filename)
    return output_filename, output_path


def _cover_letter_output_path(job: dict) -> tuple[str, str]:
    """Build output filename and full path for a generated cover letter."""
    today_date = datetime.today().strftime("%Y-%m-%d")
    safe_company = sanitize_filename(job.get('company', 'Company'))
    safe_title = sanitize_filename(job.get('title', 'Role'))
    output_filename = f"CoverLetter_{today_date}_{safe_company}_{safe_title}.docx"
    output_path = os.path.join(app.config['CV_OUTPUT_DIR'], output_filename)
    return output_filename, output_path


def _save_job_cv_content(cv_filename: str, tailored_content: dict, **kwargs) -> None:
    save_cv_content(app.config['CV_OUTPUT_DIR'], cv_filename, tailored_content, **kwargs)


def _document_chat_payload(
    store: dict,
    document_type: str,
    *,
    extra: dict | None = None,
) -> dict:
    """Build JSON payload with active session messages and session metadata."""
    document = 'cover_letter' if document_type == 'cover_letter' else 'cv'
    payload = {
        'document': document,
        'chat_history': get_active_chat_messages(store, 'cv'),
        'cover_letter_chat_history': get_active_chat_messages(store, 'cover_letter'),
        'cv_ask_chat_history': get_active_chat_messages(store, 'cv_ask'),
        'cv_chat_sessions': get_chat_sessions(store, 'cv'),
        'cover_letter_chat_sessions': get_chat_sessions(store, 'cover_letter'),
        'cv_ask_chat_sessions': get_chat_sessions(store, 'cv_ask'),
        'cv_chat_active_session_id': get_active_chat_session_id(store, 'cv'),
        'cover_letter_chat_active_session_id': get_active_chat_session_id(store, 'cover_letter'),
        'cv_ask_chat_active_session_id': get_active_chat_session_id(store, 'cv_ask'),
    }
    if extra:
        payload.update(extra)
    return payload


def _resolve_chat_document(document_type: str) -> str:
    """Map API document type strings to storage document kinds."""
    normalized = str(document_type or 'cv').strip().lower()
    if normalized in {'cover_letter', 'cover-letter'}:
        return 'cover_letter'
    if normalized in {'cv_ask', 'ask'}:
        return 'cv_ask'
    return 'cv'


def _load_job_cv_store(cv_filename: str) -> dict | None:
    return load_cv_content(app.config['CV_OUTPUT_DIR'], cv_filename)


def _clear_job_cv(job: dict, job_id: int) -> bool:
    """Remove generated CV artifacts and reset job CV fields."""
    cv_filename = job.get('cv_filename', '')
    cover_letter_filename = job.get('cover_letter_filename', '')
    if not cv_filename and not cover_letter_filename:
        return False

    delete_cv_artifacts(
        app.config['CV_OUTPUT_DIR'],
        cv_filename,
        cover_letter_filename=cover_letter_filename,
    )

    job['cv_filename'] = ''
    job['cover_letter_filename'] = ''
    job['matched_categories'] = {}
    job_repo.update_job(job_id, job)

    if session.get('current_cv_filename') == cv_filename:
        session.pop('current_cv', None)
        session.pop('current_cv_filename', None)

    return True


def _sync_job_cv_docx_from_preview(
    cv_filename: str,
    store: dict,
    profile: dict,
    *,
    persist_store: bool = True,
) -> dict:
    """Rebuild the Word CV and structured content from preview lines."""
    content = store.get('tailored_content') or {}
    profile_name = profile.get('full_name', '')
    preview_lines = resolve_cv_preview_lines(
        content,
        profile_name,
        stored_lines=store.get('cv_preview_lines'),
        customized=bool(store.get('cv_preview_customized')),
    )
    if not preview_lines:
        return store

    updated_content = preview_lines_to_content(content, preview_lines, profile_name)
    store['tailored_content'] = updated_content

    cv_path = os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)
    builder = CVDocumentBuilder(get_default_cv_template_path())
    builder.build_from_preview_lines(cv_path, preview_lines, profile, updated_content)
    build_cv_pdf(cv_path, preview_lines, profile, updated_content)

    if persist_store:
        _save_job_cv_content(cv_filename, updated_content, store=store)

    return store


def _clear_cv_preview_customization(store: dict, content: dict) -> None:
    """Drop stale preview-line overrides after structured CV content changes."""
    store['tailored_content'] = content
    store['cv_preview_lines'] = []
    store['cv_preview_customized'] = False


def get_job_ats_analysis(job: dict) -> dict | None:
    """Return stored ATS analysis for a job's generated CV, if any."""
    cv_filename = job.get('cv_filename', '')
    if not cv_filename:
        return None
    store = _load_job_cv_store(cv_filename)
    if not store:
        return None
    analysis = store.get('ats_analysis') or {}
    if not analysis:
        return None
    normalized = normalize_ats_analysis(analysis)
    if not normalized.get('ats_score') and not normalized.get('suggestions'):
        return None
    return normalized


def _job_has_sendable_cv(job: dict) -> bool:
    cv_filename = job.get('cv_filename', '')
    if not cv_filename:
        return False
    return os.path.exists(os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename))


def _can_send_application(job: dict, profile: dict) -> bool:
    return bool(
        smtp_is_configured(profile)
        and parse_recipient_emails(job)
        and _job_has_sendable_cv(job)
    )


def _send_application_for_job(job_id: int, account_id: str | None = None) -> tuple[dict, int]:
    """Send CV (and cover letter when available) to the job contact emails."""
    job = job_repo.get_job(job_id)
    if not job:
        return {'error': 'Job not found'}, 404

    profile = profile_repo.get_profile()
    account = get_send_account(profile, account_id)
    if not account:
        if account_id:
            return {'error': 'Selected sending account was not found'}, 400
        return {
            'error': (
                'No sending account configured. Connect Gmail or Outlook in your profile, '
                'or add an app-password account.'
            ),
        }, 400

    recipients = parse_recipient_emails(job)
    if not recipients:
        return {'error': 'No contact email found for this job'}, 400

    cv_filename = job.get('cv_filename', '')
    if not cv_filename:
        return {'error': 'Generate a CV before sending an application email'}, 400

    cv_path = os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)
    if not os.path.exists(cv_path):
        return {'error': 'CV file not found on disk. Regenerate the CV first.'}, 400

    attachments = [(cv_filename, cv_path)]
    cl_filename = job.get('cover_letter_filename', '')
    if cl_filename:
        cl_path = os.path.join(app.config['CV_OUTPUT_DIR'], cl_filename)
        if os.path.exists(cl_path):
            attachments.append((cl_filename, cl_path))

    store = _load_job_cv_store(cv_filename) or {}
    cover_letter = store.get('cover_letter', {})
    subject = build_application_subject(job, profile)
    body = build_application_body(job, profile, cover_letter)

    try:
        token_updates = send_application_email(
            account,
            to_emails=recipients,
            subject=subject,
            body=body,
            attachments=attachments,
            google_settings=google_oauth_settings(_oauth_redirect_uri('oauth_google_callback')),
            microsoft_settings=microsoft_oauth_settings(_oauth_redirect_uri('oauth_microsoft_callback')),
        )
        if token_updates:
            profile = update_smtp_account_tokens(profile, account['id'], token_updates)
            profile_repo.save_profile(profile)
    except Exception as exc:
        logger.error('Application email failed for job %s: %s', job_id, exc)
        return {'error': f'Failed to send email: {exc}'}, 500

    _move_jobs_with_history(
        [job_id],
        'cv_sent',
        history_label='Mark as CV Sent after send',
    )
    from_email = str(account.get('email') or '')
    return {
        'ok': True,
        'message': (
            f'Application sent from {from_email} to {", ".join(recipients)}'
        ),
        'recipients': recipients,
        'from_email': from_email,
        'workflow_status': 'cv_sent',
    }, 200


def _generate_and_save_cover_letter(
    job: dict,
    job_id: int | None,
    profile: dict,
    tailored_content: dict,
    cv_filename: str,
    *,
    reset_cover_letter_chat: bool = False,
) -> tuple[str, str, dict]:
    """Generate a cover letter docx and persist its structured content."""
    cl_filename, cl_path = _cover_letter_output_path(job)
    generator = CoverLetterGenerator()
    with dev_agent("CoverLetterGenerator", job_id=job_id):
        cl_content = generator.generate(job, profile, tailored_content)
    CoverLetterBuilder().build(cl_path, cl_content)
    build_cover_letter_pdf(cl_path, cl_content)

    if job_id:
        job['cover_letter_filename'] = cl_filename
        job_repo.update_job(job_id, job)

    store = normalize_store(_load_job_cv_store(cv_filename) or {})
    if reset_cover_letter_chat:
        start_chat_session(store, 'cover_letter')
    _save_job_cv_content(
        cv_filename,
        store.get('tailored_content', tailored_content),
        chat_history=store.get('chat_history', []),
        cover_letter=cl_content,
        cover_letter_chat_history=store.get('cover_letter_chat_history', []),
        store=store,
    )
    return cl_filename, cl_path, cl_content


def _cv_preview_context(
    job: dict,
    *,
    tailored_content: dict | None = None,
    matched_categories: dict | None = None,
    analysis: dict | None = None,
    generation_meta: dict | None = None,
    rag_chunk_count: int = 0,
    show_success_banner: bool = True,
    return_folder: str = 'all',
    return_search: str = '',
    return_sort: str = '',
    return_from_manage: bool = False,
) -> dict:
    """Build template context for the CV preview / success page."""
    cv_filename = job.get('cv_filename', '')
    store = normalize_store(_load_job_cv_store(cv_filename) if cv_filename else None)
    profile = profile_repo.get_profile()
    content = tailored_content or store.get('tailored_content', {})
    profile_name = profile.get('full_name', '')
    cv_preview_lines = resolve_cv_preview_lines(
        content,
        profile_name,
        stored_lines=store.get('cv_preview_lines'),
        customized=bool(store.get('cv_preview_customized')),
    )
    chat_history = get_active_chat_messages(store, 'cv')
    cover_letter = store.get('cover_letter', {})
    cover_letter_chat_history = get_active_chat_messages(store, 'cover_letter')
    cv_ask_chat_history = get_active_chat_messages(store, 'cv_ask')
    cv_chat_sessions = get_chat_sessions(store, 'cv')
    cover_letter_chat_sessions = get_chat_sessions(store, 'cover_letter')
    cv_ask_chat_sessions = get_chat_sessions(store, 'cv_ask')
    cv_chat_active_session_id = get_active_chat_session_id(store, 'cv')
    cover_letter_chat_active_session_id = get_active_chat_session_id(store, 'cover_letter')
    cv_ask_chat_active_session_id = get_active_chat_session_id(store, 'cv_ask')
    categories = matched_categories or CVChatEditor.content_to_matched_categories(content)
    job_id = job.get('id')
    cover_letter_filename = job.get('cover_letter_filename', '')

    return {
        'job': job,
        'job_id': job_id,
        'profile_name': profile_name,
        'profile_work_experience': profile.get('work_experience', []),
        'profile_personal_projects': profile.get('personal_projects', []),
        'cv_preview_lines': cv_preview_lines,
        'cv_filename': cv_filename,
        'cover_letter_filename': cover_letter_filename,
        'tailored_content': content,
        'cover_letter': cover_letter,
        'matched_categories': categories,
        'analysis': analysis or {},
        'generation_meta': generation_meta or {},
        'rag_chunk_count': rag_chunk_count,
        'chat_history': chat_history,
        'cover_letter_chat_history': cover_letter_chat_history,
        'cv_ask_chat_history': cv_ask_chat_history,
        'cv_chat_sessions': cv_chat_sessions,
        'cover_letter_chat_sessions': cover_letter_chat_sessions,
        'cv_ask_chat_sessions': cv_ask_chat_sessions,
        'cv_chat_active_session_id': cv_chat_active_session_id,
        'cover_letter_chat_active_session_id': cover_letter_chat_active_session_id,
        'cv_ask_chat_active_session_id': cv_ask_chat_active_session_id,
        'chat_sessions_api_url': (
            url_for('document_chat_sessions', job_id=job_id) if job_id else None
        ),
        'has_cover_letter': bool(cover_letter_filename and cover_letter.get('body_paragraphs')),
        'show_success_banner': show_success_banner,
        'return_folder': return_folder,
        'return_search': return_search,
        'return_sort': return_sort,
        'return_from_manage': return_from_manage,
        'back_url': _cv_generation_back_url(
            return_from_manage,
            return_folder,
            return_search,
            return_sort,
        ),
        'back_label': 'Back to Jobs' if return_from_manage else 'Back to Job List',
        'download_url': (
            url_for('download_job_cv', job_id=job_id)
            if job_id
            else url_for('download_cv')
        ),
        'download_pdf_url': (
            url_for('download_job_cv_pdf', job_id=job_id)
            if job_id
            else None
        ),
        'cover_letter_download_url': (
            url_for('download_job_cover_letter', job_id=job_id)
            if job_id and cover_letter_filename
            else None
        ),
        'cover_letter_download_pdf_url': (
            url_for('download_job_cover_letter_pdf', job_id=job_id)
            if job_id and cover_letter_filename
            else None
        ),
        'chat_api_url': url_for('document_chat', job_id=job_id) if job_id else None,
        'cv_ask_api_url': url_for('cv_ask_chat', job_id=job_id) if job_id else None,
        'preview_lines_api_url': (
            url_for('cv_preview_lines_update', job_id=job_id) if job_id else None
        ),
        'generate_cover_letter_url': (
            url_for('generate_job_cover_letter', job_id=job_id) if job_id else None
        ),
        'can_send_application': _can_send_application(job, profile) if job_id else False,
        'send_application_url': (
            url_for('send_job_application', job_id=job_id) if job_id else None
        ),
        'job_recipient_emails': ', '.join(parse_recipient_emails(job)) if job_id else '',
    }


def _generate_rag_cv(
    job: dict,
    profile: dict,
    *,
    reindex: bool = True,
    task_id: str | None = None,
) -> dict:
    """Generate a tailored CV using RAG + Ollama and the stored user profile."""
    output_filename, output_path = _cv_output_path(job)
    generator = RAGCVGenerator()
    job_id = job.get("id") if isinstance(job.get("id"), int) else None

    def on_progress(step: str, message: str, percent: int) -> None:
        if task_id:
            update_task(
                task_id,
                status='running',
                step=step,
                message=message,
                percent=percent,
            )

    with dev_agent(
        "RAGCVGenerator",
        task_id=task_id,
        job_id=job_id,
        context={
            "job_title": job.get("title", ""),
            "job_company": job.get("company", ""),
        },
    ):
        result = generator.generate_cv(
            job,
            output_path,
            profile=profile,
            cv_template_path=get_default_cv_template_path(),
            reindex=reindex,
            on_progress=on_progress,
        )
    result['output_filename'] = output_filename
    return result


def _cv_generation_back_url(
    return_from_manage: bool,
    return_folder: str,
    return_search: str,
    return_sort: str = '',
):
    if return_from_manage:
        return url_for(
            'manage_jobs',
            **_manage_jobs_url_kwargs(return_folder, return_search, return_sort),
        )
    if return_sort and return_sort != DEFAULT_JOB_SORT:
        return url_for('job_list', sort=return_sort)
    return url_for('job_list')


def _run_single_cv_task(
    task_id: str,
    profile: dict,
    job: dict,
    job_id: int,
    return_folder: str,
    return_search: str,
    return_from_manage: bool,
    return_sort: str = '',
) -> None:
    with dev_task(task_id, "cv_generation", job_id=job_id):
        result = _generate_rag_cv(job, profile, task_id=task_id)
        output_path = result['output_path']
        output_filename = result['output_filename']
        tailored_content = result.get('tailored_content', {})

        matched_categories = {
            'Skills Matching Job Description': tailored_content.get('job_matched_skills', []),
            'Job Skills Not In CV': tailored_content.get('job_skills_not_in_cv', []),
            'Technical Skills': tailored_content.get('technical_skills', tailored_content.get('key_skills', [])),
            'Tools & Platforms': tailored_content.get('tools_platforms', []),
        }
        job['matched_categories'] = matched_categories
        job['cv_filename'] = output_filename
        job_repo.update_job(job_id, job)
        _save_job_cv_content(output_filename, tailored_content, chat_history=[])

        try:
            cl_filename, _, cl_content = _generate_and_save_cover_letter(
                job, job_id, profile, tailored_content, output_filename
            )
        except Exception as cl_error:
            logger.error('Cover letter generation failed for job %s: %s', job_id, cl_error)
            cl_filename = ''
            cl_content = {}

        complete_task(
            task_id,
            {
                'job': job,
                'cv_filename': output_filename,
                'cover_letter_filename': cl_filename,
                'cover_letter': cl_content,
                'output_path': output_path,
                'matched_categories': matched_categories,
                'tailored_content': tailored_content,
                'analysis': result.get('analysis', {}),
                'generation_meta': result.get('models', {}),
                'rag_chunk_count': result.get('chunk_count', 0),
                'return_folder': return_folder,
                'return_search': return_search,
                'return_sort': return_sort,
                'return_from_manage': return_from_manage,
            },
        )


def _run_ats_friendly_task(
    task_id: str,
    job: dict,
    job_id: int,
    profile: dict,
    cv_filename: str,
    return_folder: str,
    return_search: str,
    return_from_manage: bool,
    return_sort: str = '',
) -> None:
    update_task(
        task_id,
        status='running',
        step='loading_cv',
        message='Loading your CV content…',
        percent=10,
    )
    store = normalize_store(_load_job_cv_store(cv_filename) or {})
    profile_name = profile.get('full_name', '')
    cv_content = resolve_effective_tailored_content(
        store.get('tailored_content', {}),
        profile_name,
        stored_lines=store.get('cv_preview_lines'),
        customized=bool(store.get('cv_preview_customized')),
    )
    if not cv_content:
        fail_task(task_id, 'CV content not found. Regenerate the CV first.')
        return

    update_task(
        task_id,
        status='running',
        step='analyzing_ats',
        message='Comparing your CV against ATS rules and the job description…',
        percent=35,
    )
    analyzer = ATSFriendlyAnalyzer()
    try:
        with dev_agent("ATSFriendlyAnalyzer", task_id=task_id, job_id=job_id):
            analysis = analyzer.analyze(job=job, cv_content=cv_content, profile=profile)
    except Exception as exc:
        logger.error('ATS analysis failed for job %s: %s', job_id, exc)
        fail_task(task_id, str(exc))
        return

    analysis['analyzed_at'] = datetime.utcnow().isoformat(timespec='seconds')
    store['ats_analysis'] = analysis
    update_task(
        task_id,
        status='running',
        step='saving',
        message='Saving ATS report…',
        percent=90,
    )
    _save_job_cv_content(cv_filename, cv_content, store=store)

    complete_task(
        task_id,
        {
            'job': job,
            'job_id': job_id,
            'cv_filename': cv_filename,
            'ats_analysis': analysis,
            'return_folder': return_folder,
            'return_search': return_search,
            'return_sort': return_sort,
            'return_from_manage': return_from_manage,
        },
    )


def _jobs_with_cv(jobs: list[dict]) -> list[dict]:
    """Return jobs in folder order that have a generated CV file on disk."""
    return [job for job in jobs if _job_has_sendable_cv(job)]


def _run_ats_pass_for_job(
    *,
    job: dict,
    job_id: int,
    profile: dict,
    cv_filename: str,
    analyzer: ATSFriendlyAnalyzer,
    apply_all: bool,
) -> dict[str, Any]:
    """Run one ATS pass for a job: analyze, optionally accept all suggestions."""
    cv_path = os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)
    store = normalize_store(_load_job_cv_store(cv_filename) or {})
    profile_name = profile.get('full_name', '')
    cv_content = resolve_effective_tailored_content(
        store.get('tailored_content', {}),
        profile_name,
        stored_lines=store.get('cv_preview_lines'),
        customized=bool(store.get('cv_preview_customized')),
    )
    if not cv_content:
        return {'status': 'skipped', 'reason': 'CV content not found'}

    with dev_agent("ATSFriendlyAnalyzer", job_id=job_id):
        analysis = analyzer.analyze(job=job, cv_content=cv_content, profile=profile)
    analysis['analyzed_at'] = datetime.utcnow().isoformat(timespec='seconds')

    result: dict[str, Any] = {
        'status': 'ok',
        'analyzed': True,
        'applied': 0,
        'suggestion_count': len(analysis.get('suggestions', [])),
    }

    if apply_all:
        to_apply = pending_suggestions(analysis)
        if not to_apply:
            result['apply_skipped'] = True
        else:
            suggestion_ids = [item['id'] for item in to_apply]
            try:
                with dev_agent("ATSFriendlyAnalyzer", job_id=job_id):
                    updated_content = analyzer.apply_all_suggestions(
                        job=job,
                        cv_content=cv_content,
                        profile=profile,
                        suggestions=to_apply,
                    )
                editor = CVChatEditor()
                editor.rebuild_document(cv_path, updated_content, profile)
                matched_categories = CVChatEditor.content_to_matched_categories(updated_content)
                job['matched_categories'] = matched_categories
                job_repo.update_job(job_id, job)

                analysis = update_suggestions_status(analysis, suggestion_ids, status='applied')
                cv_content = updated_content
                _clear_cv_preview_customization(store, updated_content)
                result['applied'] = len(suggestion_ids)
            except Exception as exc:
                logger.error('Batch ATS apply_all failed for job %s: %s', job_id, exc)
                try:
                    analysis = update_suggestions_status(
                        analysis,
                        suggestion_ids,
                        status='failed',
                        error=str(exc),
                    )
                except KeyError:
                    pass
                store['ats_analysis'] = analysis
                _save_job_cv_content(cv_filename, cv_content, store=store)
                result['status'] = 'apply_failed'
                result['error'] = str(exc)
                return result

    store['ats_analysis'] = analysis
    _save_job_cv_content(cv_filename, cv_content, store=store)
    return result


def _run_batch_ats_friendly_task(
    task_id: str,
    jobs: list[dict],
    profile: dict,
    return_folder: str,
    return_search: str,
    return_sort: str = '',
) -> None:
    """Run three ATS passes on every job in the folder that has a CV."""
    with dev_task(task_id, "batch_ats_friendly"):
        jobs_with_cv = _jobs_with_cv(jobs)
        if not jobs_with_cv:
            fail_task(task_id, 'No jobs with generated CVs found in this folder.')
            return

        analyzer = ATSFriendlyAnalyzer()
        if not analyzer.llm.is_available():
            fail_task(
                task_id,
                f'{analyzer.llm.provider_label} is not reachable. Check your LLM settings.',
            )
            return

        try:
            analyzer.llm.validate_models()
        except Exception as exc:
            fail_task(task_id, str(exc))
            return

        total_passes = len(BATCH_ATS_FRIENDLY_PASSES)
        total_steps = len(jobs_with_cv) * total_passes
        stats: dict[str, Any] = {
            'total_jobs': len(jobs_with_cv),
            'passes': total_passes,
            'analyzed': 0,
            'apply_passes': 0,
            'suggestions_applied': 0,
            'apply_skipped_no_suggestions': 0,
            'skipped': 0,
            'failed': 0,
            'failed_jobs': [],
        }

        step = 0
        for pass_index, pass_config in enumerate(BATCH_ATS_FRIENDLY_PASSES, start=1):
            apply_all = bool(pass_config['apply_all'])
            pass_label = pass_config['label']

            for job_index, job in enumerate(jobs_with_cv, start=1):
                try:
                    task_control_checkpoint(task_id)
                except TaskStopped:
                    current_task = get_task(task_id)
                    stopped = bool(current_task and current_task.get('control') == 'stop')
                    if stats['analyzed'] == 0:
                        fail_task(task_id, 'Batch ATS optimization stopped before any jobs were processed.')
                        return
                    complete_task(
                        task_id,
                        {
                            'stats': stats,
                            'return_folder': return_folder,
                            'return_search': return_search,
                            'return_sort': return_sort,
                            'stopped': stopped,
                        },
                        message=(
                            f'Batch ATS optimization stopped — processed {stats["analyzed"]} '
                            f'analysis pass(es) across {stats["total_jobs"]} job(s)'
                        ),
                    )
                    return

                step += 1
                job_id = job.get('id')
                title = job.get('title', 'Untitled')
                cv_filename = job.get('cv_filename', '')
                percent = 5 + int((step / max(total_steps, 1)) * 90)
                action_label = 'accepting suggestions' if apply_all else 'analyzing'
                update_task(
                    task_id,
                    status='running',
                    step='batch_ats',
                    message=(
                        f'Pass {pass_index} of {total_passes} — job {job_index} of '
                        f'{len(jobs_with_cv)}: {title} — {action_label}…'
                    ),
                    percent=percent,
                    meta={
                        'current_pass': pass_index,
                        'total_passes': total_passes,
                        'current_index': job_index,
                        'total_jobs': len(jobs_with_cv),
                        'current_job_title': title,
                        'apply_all': apply_all,
                    },
                )

                if not job_id or not cv_filename:
                    stats['skipped'] += 1
                    continue

                try:
                    pass_result = _run_ats_pass_for_job(
                        job=job,
                        job_id=job_id,
                        profile=profile,
                        cv_filename=cv_filename,
                        analyzer=analyzer,
                        apply_all=apply_all,
                    )
                except Exception as exc:
                    logger.error('Batch ATS pass failed for job %s: %s', job_id, exc)
                    stats['failed'] += 1
                    stats['failed_jobs'].append({'job_id': job_id, 'title': title, 'error': str(exc)})
                    continue

                if pass_result.get('status') == 'skipped':
                    stats['skipped'] += 1
                    continue
                if pass_result.get('status') == 'apply_failed':
                    stats['failed'] += 1
                    stats['failed_jobs'].append({
                        'job_id': job_id,
                        'title': title,
                        'error': pass_result.get('error', 'Apply all failed'),
                    })
                    stats['analyzed'] += 1
                    continue

                stats['analyzed'] += 1
                if apply_all:
                    stats['apply_passes'] += 1
                    if pass_result.get('apply_skipped'):
                        stats['apply_skipped_no_suggestions'] += 1
                    else:
                        stats['suggestions_applied'] += pass_result.get('applied', 0)

        complete_task(
            task_id,
            {
                'stats': stats,
                'return_folder': return_folder,
                'return_search': return_search,
                'return_sort': return_sort,
            },
            message=(
                f'Batch ATS optimization complete — {stats["analyzed"]} analysis pass(es) on '
                f'{stats["total_jobs"]} job(s), {stats["suggestions_applied"]} suggestion(s) applied'
            ),
        )


def _ats_friendly_results_context(
    job: dict,
    ats_analysis: dict,
    *,
    return_folder: str = 'all',
    return_search: str = '',
    return_sort: str = '',
    return_from_manage: bool = False,
) -> dict:
    job_id = job.get('id')
    cv_filename = job.get('cv_filename', '')
    store = normalize_store(_load_job_cv_store(cv_filename) if cv_filename else None)
    profile = profile_repo.get_profile()
    profile_name = profile.get('full_name', '')
    content = resolve_effective_tailored_content(
        store.get('tailored_content', {}),
        profile_name,
        stored_lines=store.get('cv_preview_lines'),
        customized=bool(store.get('cv_preview_customized')),
    )
    return {
        'job': job,
        'job_id': job_id,
        'ats_analysis': normalize_ats_analysis(ats_analysis),
        'cv_preview_lines': cv_content_to_preview_lines(content, profile_name),
        'suggestions_api_url': url_for('ats_suggestion_action', job_id=job_id) if job_id else None,
        'reanalyze_url': url_for(
            'ats_friendly_progress',
            job_id=job_id,
            folder=return_folder,
            q=return_search or None,
            sort=return_sort or None,
            force='1',
        ) if job_id else None,
        'preview_cv_url': url_for(
            'preview_job_cv',
            job_id=job_id,
            folder=return_folder,
            q=return_search or None,
            sort=return_sort or None,
        ) if job_id else None,
        'return_folder': return_folder,
        'return_search': return_search,
        'return_sort': return_sort,
        'return_from_manage': return_from_manage,
        'back_url': _cv_generation_back_url(
            return_from_manage,
            return_folder,
            return_search,
            return_sort,
        ),
    }


CONTROLLABLE_TASK_TYPES = frozenset(
    {
        'single_search',
        'batch_search',
        'job_match_analyze',
        'batch_ats_friendly',
        'batch_cv',
    }
) | CONTROLLABLE_AI_TASK_TYPES | CONTROLLABLE_URGENT_TASK_TYPES


def _read_batch_lines_from_request(field_name: str, text_field_name: str) -> list[str]:
    """Read newline-separated values from an uploaded file or text area."""
    upload = request.files.get(field_name)
    if upload and upload.filename:
        return parse_lines(decode_uploaded_text(upload.read()))

    text = (request.form.get(text_field_name) or '').strip()
    if text:
        return parse_lines(text)
    return []


def _run_batch_cv_task(task_id: str, profile: dict, jobs: list[dict]) -> None:
    generator = RAGCVGenerator()

    def on_progress(step: str, message: str, percent: int) -> None:
        update_task(task_id, status='running', step=step, message=message, percent=percent)

    on_progress('validating_ollama', f'Checking {generator.llm.provider_label} and models…', 5)
    if not generator.llm.is_available():
        raise RuntimeError(f'{generator.llm.provider_label} is not reachable.')

    generator.llm.validate_models()
    on_progress('indexing_cv', 'Indexing your profile with RAG…', 12)
    generator.prepare_profile_index(profile)

    successful_jobs = []
    failed_jobs = []
    generated_cvs = []
    total = len(jobs)

    for index, job in enumerate(jobs, start=1):
        job_id = job.get('id')
        title = job.get('title', 'Job')
        base_percent = 15 + int(((index - 1) / max(total, 1)) * 80)
        update_task(
            task_id,
            message=f'Generating CV {index} of {total}: {title}',
            percent=base_percent,
            meta={
                'current_index': index,
                'total_jobs': total,
                'current_job_title': title,
            },
        )

        try:
            output_filename, output_path = _cv_output_path(job)
            with dev_agent(
                "RAGCVGenerator",
                job_id=job_id,
                context={
                    "job_title": title,
                    "batch_index": index,
                    "batch_total": total,
                },
            ):
                result = generator.generate_cv(
                    job,
                    output_path,
                    profile=profile,
                    cv_template_path=get_default_cv_template_path(),
                    reindex=False,
                    on_progress=lambda step, message, percent, bp=base_percent: update_task(
                        task_id,
                        step=step,
                        message=message,
                        percent=min(95, bp + percent // 10),
                        meta={
                            'current_index': index,
                            'total_jobs': total,
                            'current_job_title': title,
                        },
                    ),
                )
            generated_cvs.append(result['output_path'])
            successful_jobs.append(job)
            if job_id:
                tailored_content = result.get('tailored_content', {})
                job['matched_categories'] = {
                    'Skills Matching Job Description': tailored_content.get('job_matched_skills', []),
                    'Job Skills Not In CV': tailored_content.get('job_skills_not_in_cv', []),
                    'Technical Skills': tailored_content.get('technical_skills', tailored_content.get('key_skills', [])),
                    'Tools & Platforms': tailored_content.get('tools_platforms', []),
                }
                job['cv_filename'] = output_filename
                job_repo.update_job(job_id, job)
                _save_job_cv_content(output_filename, tailored_content, chat_history=[])
                try:
                    _generate_and_save_cover_letter(
                        job, job_id, profile, tailored_content, output_filename
                    )
                except Exception as cl_error:
                    logger.error('Batch cover letter failed for %s: %s', title, cl_error)
        except Exception as job_error:
            logger.error('Batch CV failed for %s: %s', title, job_error)
            failed_jobs.append(job)

    if not generated_cvs:
        raise RuntimeError('No CVs were generated.')

    complete_task(
        task_id,
        {
            'successful_jobs': successful_jobs,
            'failed_jobs': failed_jobs,
            'generated_cvs': generated_cvs,
            'generated_count': len(generated_cvs),
        },
    )


def _get_jobs_for_view(
    search_run_id: int | None = None,
    sort_by: str | None = None,
) -> list[dict]:
    """Load jobs from SQLite for the current view."""
    if search_run_id is not None:
        jobs = job_repo.list_jobs(search_run_id=search_run_id)
        if jobs:
            return sort_jobs(jobs, sort_by)
    return sort_jobs(job_repo.list_jobs(), sort_by)


def _parse_bulk_job_ids(raw_job_ids: list[str]) -> list[int] | None:
    """Parse submitted bulk job IDs, or None when empty or invalid."""
    if not raw_job_ids:
        return None
    try:
        return [int(job_id) for job_id in raw_job_ids]
    except ValueError:
        return None


def _get_jobs_by_ids(job_ids: list[int]) -> list[dict]:
    """Load jobs by ID, preserving the submitted selection order."""
    jobs = []
    for job_id in job_ids:
        job = job_repo.get_job(job_id)
        if job:
            jobs.append(job)
    return jobs


def _batch_cv_back_url(
    return_view: str,
    return_folder: str,
    return_search: str,
    return_sort: str,
) -> str:
    """Build the return URL after selected-job batch CV generation."""
    if return_view == 'manage':
        kwargs = {}
        if return_search:
            kwargs['q'] = return_search
        if return_sort and return_sort != DEFAULT_JOB_SORT:
            kwargs['sort'] = return_sort
        if return_folder and return_folder != 'all':
            return url_for('manage_jobs', folder=return_folder, **kwargs)
        return url_for('manage_jobs', **kwargs)
    return url_for('job_list', sort=return_sort or None)


def _job_form_data() -> dict:
    """Read job fields from the current request form."""
    data = {column: request.form.get(column, '') for column in JOB_COLUMNS}
    workflow_status = request.form.get('workflow_status', DEFAULT_JOB_STATUS)
    data['workflow_status'] = (
        workflow_status if is_valid_job_status(workflow_status) else DEFAULT_JOB_STATUS
    )
    return data


def _manage_jobs_redirect(folder: str = 'all', search: str = '', sort: str = ''):
    """Redirect back to the manage jobs view preserving folder context."""
    kwargs = {}
    if search:
        kwargs['q'] = search
    if sort and sort != DEFAULT_JOB_SORT:
        kwargs['sort'] = sort
    if folder and folder != 'all':
        kwargs['folder'] = folder
    return redirect(url_for('manage_jobs', **kwargs))


def _job_move_history_context() -> dict[str, Any]:
    return {
        'can_undo_job_moves': can_undo_job_moves(session),
        'can_redo_job_moves': can_redo_job_moves(session),
        'undo_job_move_label': undo_job_move_label(session),
        'redo_job_move_label': redo_job_move_label(session),
    }


def _move_jobs_with_history(
    job_ids: list[int],
    workflow_status: str,
    *,
    history_label: str | None = None,
) -> int:
    changes = job_repo.move_jobs_status(job_ids, workflow_status)
    if changes:
        label = history_label or f'Move to {job_status_label(workflow_status)}'
        record_job_moves(session, changes, label)
    return len(changes)


def _record_job_status_changes(
    changes: list[dict],
    *,
    history_label: str,
) -> None:
    if changes:
        record_job_moves(session, changes, history_label)


def _undo_job_moves_redirect(
    return_view: str,
    return_folder: str,
    return_search: str,
    return_sort: str,
):
    entry = pop_undo_job_moves(session)
    if not entry:
        flash('Nothing to undo', 'info')
    else:
        restored = job_repo.apply_job_status_changes(
            entry.get('changes', []),
            use_from_status=True,
        )
        if restored:
            flash(f'Undid: {entry.get("label", "job move")}', 'success')
        else:
            flash('Could not undo the last job move', 'warning')

    if return_view == 'list':
        kwargs = {}
        if return_sort and return_sort != DEFAULT_JOB_SORT:
            kwargs['sort'] = return_sort
        return redirect(url_for('job_list', **kwargs))
    return _manage_jobs_redirect(return_folder, return_search, return_sort)


def _redo_job_moves_redirect(
    return_view: str,
    return_folder: str,
    return_search: str,
    return_sort: str,
):
    entry = pop_redo_job_moves(session)
    if not entry:
        flash('Nothing to redo', 'info')
    else:
        restored = job_repo.apply_job_status_changes(
            entry.get('changes', []),
            use_from_status=False,
        )
        if restored:
            flash(f'Redid: {entry.get("label", "job move")}', 'success')
        else:
            flash('Could not redo the last job move', 'warning')

    if return_view == 'list':
        kwargs = {}
        if return_sort and return_sort != DEFAULT_JOB_SORT:
            kwargs['sort'] = return_sort
        return redirect(url_for('job_list', **kwargs))
    return _manage_jobs_redirect(return_folder, return_search, return_sort)


def _redirect_after_job_status_update(
    return_view: str,
    job_id: int,
    workflow_status: str,
    return_folder: str,
    return_search: str,
    return_sort: str,
    return_from_manage: bool,
):
    """Redirect after a single job status change based on where the action started."""
    if return_view == 'list':
        kwargs = {}
        if return_sort and return_sort != DEFAULT_JOB_SORT:
            kwargs['sort'] = return_sort
        return redirect(url_for('job_list', **kwargs))

    if return_view == 'preview':
        if workflow_status in ('archived', 'shortlisted'):
            if return_from_manage:
                return _manage_jobs_redirect(workflow_status, return_search, return_sort)
            kwargs = {}
            if return_sort and return_sort != DEFAULT_JOB_SORT:
                kwargs['sort'] = return_sort
            return redirect(url_for('job_list', **kwargs))

        kwargs = {}
        if return_from_manage or return_folder != 'all' or return_search:
            if return_folder:
                kwargs['folder'] = return_folder
            if return_search:
                kwargs['q'] = return_search
        if return_sort and return_sort != DEFAULT_JOB_SORT:
            kwargs['sort'] = return_sort
        return redirect(url_for('preview_job_cv', job_id=job_id, **kwargs))

    return _manage_jobs_redirect(return_folder, return_search, return_sort)


def _redirect_after_clear_cv(
    return_view: str,
    return_folder: str,
    return_search: str,
    return_sort: str,
):
    """Redirect after clearing a job's generated CV."""
    if return_view in ('list', 'preview'):
        kwargs = {}
        if return_sort and return_sort != DEFAULT_JOB_SORT:
            kwargs['sort'] = return_sort
        return redirect(url_for('job_list', **kwargs))
    return _manage_jobs_redirect(return_folder, return_search, return_sort)


def _manage_jobs_url_kwargs(
    folder: str = 'all',
    search: str = '',
    sort: str = '',
) -> dict:
    """Build query kwargs for manage jobs links."""
    kwargs: dict = {}
    if search:
        kwargs['q'] = search
    if sort and sort != DEFAULT_JOB_SORT:
        kwargs['sort'] = sort
    if folder and folder != 'all':
        kwargs['folder'] = folder
    return kwargs


@app.template_filter('job_status_label')
def _job_status_label_filter(status):
    return job_status_label(status)


@app.template_filter('profile_match_score')
def _profile_match_score_filter(job):
    return get_profile_match_score(job)


@app.template_filter('profile_match_analysis')
def _profile_match_analysis_filter(job):
    return get_profile_match_analysis(job)


@app.template_filter('job_ats_analysis')
def _job_ats_analysis_filter(job):
    return get_job_ats_analysis(job)


@app.route('/')
def index():
    """Render the home page."""
    profile = profile_repo.get_profile()
    status_counts = job_repo.count_jobs_by_status()
    return render_template(
        'index.html',
        profile_ready=profile_is_ready(profile),
        profile_name=(profile.get('full_name') or '').strip(),
        profile_has_skills=profile_has_matchable_skills(profile),
        email_configured=smtp_is_configured(profile),
        total_jobs=sum(status_counts.values()),
        status_counts=status_counts,
        status_labels=JOB_STATUS_LABELS,
    )

@app.route('/search', methods=['GET', 'POST'])
def search_jobs():
    """Queue a single job search and show progress while it runs."""
    if request.method == 'POST':
        keyword = (request.form.get('keyword') or '').strip()
        location = (request.form.get('location') or '').strip()
        max_jobs = int(request.form.get('max_jobs', 10))
        try:
            sources = _parse_job_sources_from_form()
        except ValueError as exc:
            flash(str(exc), 'error')
            return redirect(url_for('index'))
        mode = request.form.get('mode', 'both')
        search_filters = SearchFilters.from_mapping(request.form)

        if not keyword or not location:
            flash('Please enter both job title and location', 'error')
            return redirect(url_for('index'))

        source_list = parse_sources_csv(sources)

        task_id = _enqueue_urgent_task(
            'single_search',
            {
                'keyword': keyword,
                'location': location,
                'max_jobs': max_jobs,
                'sources': sources,
                'source_list': source_list,
                'mode': mode,
                'search_filters': {
                    'remote': search_filters.remote,
                    'relocation': search_filters.relocation,
                    'visa_sponsorship': search_filters.visa_sponsorship,
                },
                'meta': {'keyword': keyword, 'location': location},
            },
        )
        session['single_search_active'] = task_id
        flash(
            'Job search queued — the urgent worker will start it immediately '
            '(run: job-apply-ai urgent-worker)',
            'success',
        )
        return redirect(url_for('single_search_progress', task_id=task_id))

    return redirect(url_for('index'))


@app.route('/search/batch', methods=['POST'])
def batch_search_jobs():
    """Enqueue a one-time batch search for the worker process."""
    titles = _read_batch_lines_from_request('titles_file', 'titles_text')
    locations = _read_batch_lines_from_request('locations_file', 'locations_text')
    if not titles or not locations:
        flash('Provide at least one job title and one location.', 'error')
        return redirect(url_for('index'))

    max_jobs = int(request.form.get('max_jobs', 5))
    try:
        sources = _parse_job_sources_from_form()
    except ValueError as exc:
        flash(str(exc), 'error')
        return redirect(url_for('index'))
    mode = request.form.get('mode', 'both')
    search_filters = SearchFilters.from_mapping(request.form)
    shuffle_queue = request.form.get('shuffle_queue') == 'on'
    total_combinations = len(titles) * len(locations)

    try:
        jobs = batch_queue_repo.create_jobs(
            name=f"Dashboard batch ({total_combinations} searches)",
            titles=titles,
            locations=locations,
            schedule_type='once',
            shuffle_queue=shuffle_queue,
            max_jobs=max_jobs,
            sources=sources,
            mode=mode,
            search_filters=search_filters,
            run_immediately=True,
        )
    except ValueError as exc:
        flash(str(exc), 'error')
        return redirect(url_for('index'))

    worker_hint = (
        'start it with: python -m job_apply_ai batch-worker'
    )
    if len(jobs) == 1:
        session['batch_search_active'] = jobs[0]['task_id']
        flash(
            f'Batch search queued. The worker will pick it up shortly — {worker_hint}',
            'success',
        )
        return redirect(url_for('batch_search_progress', task_id=jobs[0]['task_id']))

    flash(
        f'Large batch split into {len(jobs)} queue jobs '
        f'({total_combinations} searches total). The worker will process them in order — '
        f'{worker_hint}',
        'success',
    )
    return redirect(url_for('batch_queue_list'))


@app.route('/ai-queue')
def ai_queue_list():
    """List and manage AI tasks in the worker queue."""
    jobs = ai_task_queue_repo.list_jobs()
    for job in jobs:
        snapshot = to_ai_task_snapshot(job)
        job['progress_url'] = _ai_task_progress_url(snapshot)
    finished_count = sum(1 for job in jobs if job['status'] in TERMINAL_STATUSES)
    stoppable_count = sum(1 for job in jobs if job['status'] in ('pending', 'running', 'paused'))
    return render_template(
        'ai_queue.html',
        jobs=jobs,
        task_type_labels=AI_TASK_TYPE_LABELS,
        status_labels=AI_STATUS_LABELS,
        finished_count=finished_count,
        stoppable_count=stoppable_count,
    )


@app.route('/ai-queue/clear', methods=['POST'])
def ai_queue_clear():
    deleted = ai_task_queue_repo.clear_finished_jobs()
    flash(f'Removed {deleted} finished AI task(s) from the queue.', 'success')
    return redirect(url_for('ai_queue_list'))


@app.route('/ai-queue/stop-all', methods=['POST'])
def ai_queue_stop_all():
    cancelled, stop_requested = ai_task_queue_repo.stop_all_active_jobs()
    parts = []
    if cancelled:
        parts.append(f'{cancelled} pending cancelled')
    if stop_requested:
        parts.append(f'{stop_requested} running/paused stopping')
    flash(
        f'AI queue: {", ".join(parts) or "no active tasks to stop"}.',
        'success' if parts else 'info',
    )
    return redirect(url_for('ai_queue_list'))


@app.route('/ai-queue/<int:job_id>/delete', methods=['POST'])
def ai_queue_delete(job_id):
    try:
        if not ai_task_queue_repo.delete_job(job_id):
            flash('AI task not found.', 'warning')
    except ValueError as exc:
        flash(str(exc), 'error')
    else:
        flash('AI task deleted.', 'success')
    return redirect(url_for('ai_queue_list'))


@app.route('/ai-queue/<int:job_id>/pause', methods=['POST'])
def ai_queue_pause(job_id):
    if ai_task_queue_repo.pause_job(job_id):
        flash('AI task paused.', 'success')
    else:
        flash('Task cannot be paused in its current state.', 'warning')
    return redirect(url_for('ai_queue_list'))


@app.route('/ai-queue/<int:job_id>/resume', methods=['POST'])
def ai_queue_resume(job_id):
    if ai_task_queue_repo.resume_job(job_id):
        flash('AI task resumed.', 'success')
    else:
        flash('Task cannot be resumed in its current state.', 'warning')
    return redirect(url_for('ai_queue_list'))


@app.route('/ai-queue/<int:job_id>/stop', methods=['POST'])
def ai_queue_stop(job_id):
    job = ai_task_queue_repo.get_job(job_id)
    if not job:
        flash('AI task not found.', 'warning')
    elif job['status'] == 'pending':
        ai_task_queue_repo.cancel_job(job_id)
        flash('Pending AI task cancelled.', 'success')
    elif ai_task_queue_repo.request_stop(job_id):
        flash('Stop requested — worker will halt when possible.', 'success')
    else:
        flash('Task cannot be stopped in its current state.', 'warning')
    return redirect(url_for('ai_queue_list'))


@app.route('/api/search/suggest-titles', methods=['POST'])
def suggest_search_titles():
    """Suggest job board search titles from the stored profile."""
    payload = request.get_json(silent=True) or {}
    try:
        max_titles = int(payload.get('max_titles', 10))
    except (TypeError, ValueError):
        max_titles = 10

    profile = profile_repo.get_profile()
    result = suggest_job_titles(profile, max_titles=max_titles)
    status = 200 if result.get('titles') else 400
    return jsonify(result), status


@app.route('/search/batch/<task_id>')
def batch_search_progress(task_id):
    """Show progress while batch job search runs."""
    task = _resolve_task(task_id)
    if not task:
        flash('Batch search task not found', 'error')
        return redirect(url_for('index'))

    total_searches = task.get('meta', {}).get('total_searches', 0)
    return render_template(
        'batch_search_progress.html',
        task_id=task_id,
        total_searches=total_searches,
        status_url=url_for('cv_task_status', task_id=task_id),
        complete_url=url_for('batch_search_complete', task_id=task_id),
        control_url=url_for('control_background_task', task_id=task_id),
        back_url=f"{url_for('index')}#batch-search-jobs",
        back_label='Back to home',
    )


@app.route('/search/batch/complete/<task_id>')
def batch_search_complete(task_id):
    """Show jobs found by a completed batch search."""
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('Batch search result not found', 'error')
        return redirect(url_for('index'))

    result = task['result']
    search_run_id = result.get('search_run_id')
    processed_jobs = sort_jobs(
        job_repo.list_jobs(search_run_id=search_run_id),
        'match_desc',
    )
    session['search_run_id'] = search_run_id

    failed_searches = result.get('failed_searches') or []
    if result.get('stopped'):
        flash(task.get('message', 'Batch search stopped.'), 'warning')
    elif failed_searches:
        flash(
            f'Batch search finished with {len(failed_searches)} failed combination(s).',
            'warning',
        )

    session.pop('batch_search_active', None)

    if not processed_jobs:
        flash('Batch search completed but no jobs were saved.', 'warning')
        return redirect(url_for('index'))

    return render_template(
        'job_list.html',
        jobs=processed_jobs,
        search_run_id=search_run_id,
        current_sort='match_desc',
        job_sort_options=JOB_SORT_OPTIONS,
    )


def _batch_queue_form_payload(sources: str) -> dict:
    """Parse shared batch queue create/edit form fields."""
    titles = parse_lines(request.form.get('titles_text') or '')
    locations = parse_lines(request.form.get('locations_text') or '')
    return {
        'name': (request.form.get('name') or '').strip(),
        'titles': titles,
        'locations': locations,
        'schedule_type': request.form.get('schedule_type', 'once'),
        'shuffle_queue': request.form.get('shuffle_queue') == 'on',
        'max_jobs': int(request.form.get('max_jobs', 5)),
        'sources': sources,
        'mode': request.form.get('mode', 'both'),
        'search_filters': SearchFilters.from_mapping(request.form),
    }


def _batch_queue_form_from_request(sources: str | None = None) -> dict:
    """Build template form dict from the current request (for validation errors)."""
    filters = SearchFilters.from_mapping(request.form)
    source_ids = request.form.getlist('sources') if request.form.get('job_sources_field') == '1' else None
    return {
        'name': (request.form.get('name') or '').strip(),
        'titles_text': request.form.get('titles_text') or '',
        'locations_text': request.form.get('locations_text') or '',
        'schedule_type': request.form.get('schedule_type', 'once'),
        'shuffle_queue': request.form.get('shuffle_queue') == 'on',
        'max_jobs': int(request.form.get('max_jobs', 5) or 5),
        'sources': sources if sources is not None else format_sources_csv(source_ids or []),
        'source_ids': source_ids if source_ids is not None else None,
        'mode': request.form.get('mode', 'both'),
        'filter_remote': filters.remote,
        'filter_relocation': filters.relocation,
        'filter_visa_sponsorship': filters.visa_sponsorship,
    }


@app.route('/batch-queue')
def batch_queue_list():
    """List batch search queue jobs."""
    jobs = batch_queue_repo.list_jobs()
    finished_count = sum(
        1 for job in jobs if job['status'] in ('completed', 'failed', 'cancelled')
    )
    stoppable_count = sum(
        1 for job in jobs if job['status'] in ('pending', 'running', 'paused')
    )
    return render_template(
        'batch_queue.html',
        jobs=jobs,
        finished_count=finished_count,
        stoppable_count=stoppable_count,
        status_labels=STATUS_LABELS,
        schedule_labels=SCHEDULE_LABELS,
    )


@app.route('/batch-queue/new', methods=['GET', 'POST'])
def batch_queue_create():
    """Create a batch search queue job."""
    if request.method == 'POST':
        try:
            sources = _parse_job_sources_from_form()
        except ValueError as exc:
            flash(str(exc), 'error')
            return render_template(
                'batch_queue_form.html',
                job=None,
                form=_batch_queue_form_from_request(),
                schedule_labels=SCHEDULE_LABELS,
            )
        payload = _batch_queue_form_payload(sources)
        try:
            jobs = batch_queue_repo.create_jobs(
                name=payload['name'],
                titles=payload['titles'],
                locations=payload['locations'],
                schedule_type=payload['schedule_type'],
                shuffle_queue=payload['shuffle_queue'],
                max_jobs=payload['max_jobs'],
                sources=payload['sources'],
                mode=payload['mode'],
                search_filters=payload['search_filters'],
                run_immediately=True,
            )
        except ValueError as exc:
            flash(str(exc), 'error')
            filters = payload['search_filters']
            return render_template(
                'batch_queue_form.html',
                job=None,
                form={
                    **payload,
                    'titles_text': '\n'.join(payload['titles']),
                    'locations_text': '\n'.join(payload['locations']),
                    'filter_remote': filters.remote,
                    'filter_relocation': filters.relocation,
                    'filter_visa_sponsorship': filters.visa_sponsorship,
                },
                schedule_labels=SCHEDULE_LABELS,
            )
        total_combinations = len(payload['titles']) * len(payload['locations'])
        if len(jobs) == 1:
            flash('Batch search job queued.', 'success')
        else:
            flash(
                f'Large batch split into {len(jobs)} queue jobs '
                f'({total_combinations} searches total).',
                'success',
            )
        return redirect(url_for('batch_queue_list'))

    return render_template(
        'batch_queue_form.html',
        job=None,
        form={},
        schedule_labels=SCHEDULE_LABELS,
    )


@app.route('/batch-queue/<int:job_id>/edit', methods=['GET', 'POST'])
def batch_queue_edit(job_id):
    """Edit a pending or paused batch search queue job."""
    job = batch_queue_repo.get_job(job_id)
    if not job:
        flash('Queue job not found', 'error')
        return redirect(url_for('batch_queue_list'))
    if job['status'] not in ('pending', 'paused'):
        flash('Only pending or paused jobs can be edited.', 'error')
        return redirect(url_for('batch_queue_list'))

    if request.method == 'POST':
        try:
            sources = _parse_job_sources_from_form()
        except ValueError as exc:
            flash(str(exc), 'error')
            return render_template(
                'batch_queue_form.html',
                job=job,
                form=_batch_queue_form_from_request(),
                schedule_labels=SCHEDULE_LABELS,
            )
        payload = _batch_queue_form_payload(sources)
        try:
            batch_queue_repo.update_job(
                job_id,
                name=payload['name'],
                titles=payload['titles'],
                locations=payload['locations'],
                schedule_type=payload['schedule_type'],
                shuffle_queue=payload['shuffle_queue'],
                max_jobs=payload['max_jobs'],
                sources=payload['sources'],
                mode=payload['mode'],
                search_filters=payload['search_filters'],
            )
        except ValueError as exc:
            flash(str(exc), 'error')
            return render_template(
                'batch_queue_form.html',
                job=job,
                form=payload,
                schedule_labels=SCHEDULE_LABELS,
            )
        flash('Batch search job updated.', 'success')
        return redirect(url_for('batch_queue_list'))

    form = {
        'name': job['name'],
        'titles_text': '\n'.join(job['titles']),
        'locations_text': '\n'.join(job['locations']),
        'schedule_type': job['schedule_type'],
        'shuffle_queue': job['shuffle_queue'],
        'max_jobs': job['max_jobs'],
        'sources': job['sources'],
        'mode': job['mode'],
        'filter_remote': job['search_filters'].get('remote'),
        'filter_relocation': job['search_filters'].get('relocation'),
        'filter_visa_sponsorship': job['search_filters'].get('visa_sponsorship'),
    }
    return render_template(
        'batch_queue_form.html',
        job=job,
        form=form,
        schedule_labels=SCHEDULE_LABELS,
    )


@app.route('/batch-queue/clear', methods=['POST'])
def batch_queue_clear():
    """Remove finished batch search queue jobs (completed, failed, cancelled)."""
    deleted = batch_queue_repo.clear_finished_jobs()
    if deleted:
        flash(f'Removed {deleted} finished batch search job(s).', 'success')
    else:
        flash('No finished batch search jobs to clear.', 'info')
    return redirect(url_for('batch_queue_list'))


@app.route('/batch-queue/stop-all', methods=['POST'])
def batch_queue_stop_all():
    """Cancel pending jobs and request stop for all running or paused queue jobs."""
    cancelled, stop_requested = batch_queue_repo.stop_all_active_jobs()
    if cancelled or stop_requested:
        parts = []
        if cancelled:
            parts.append(f'{cancelled} pending job(s) cancelled')
        if stop_requested:
            parts.append(
                f'stop requested for {stop_requested} running or paused job(s)'
            )
        flash(f'Stop all: {", ".join(parts)}.', 'success')
    else:
        flash('No active batch search jobs to stop.', 'info')
    return redirect(url_for('batch_queue_list'))


@app.route('/batch-queue/<int:job_id>/delete', methods=['POST'])
def batch_queue_delete(job_id):
    """Delete a batch search queue job."""
    try:
        if not batch_queue_repo.delete_job(job_id):
            flash('Queue job not found', 'error')
        else:
            flash('Batch search job deleted.', 'success')
    except ValueError as exc:
        flash(str(exc), 'error')
    return redirect(url_for('batch_queue_list'))


@app.route('/batch-queue/<int:job_id>/pause', methods=['POST'])
def batch_queue_pause(job_id):
    """Pause a running batch search queue job."""
    if batch_queue_repo.pause_job(job_id):
        flash('Batch search paused.', 'success')
    else:
        flash('Job is not running.', 'warning')
    return redirect(url_for('batch_queue_list'))


@app.route('/batch-queue/<int:job_id>/resume', methods=['POST'])
def batch_queue_resume(job_id):
    """Resume a paused batch search queue job."""
    if batch_queue_repo.resume_job(job_id):
        flash('Batch search resumed.', 'success')
    else:
        flash('Job is not paused.', 'warning')
    return redirect(url_for('batch_queue_list'))


@app.route('/batch-queue/<int:job_id>/stop', methods=['POST'])
def batch_queue_stop(job_id):
    """Request stop for a batch search queue job."""
    job = batch_queue_repo.get_job(job_id)
    if not job:
        flash('Queue job not found', 'error')
    elif job['status'] == 'pending':
        batch_queue_repo.cancel_job(job_id)
        flash('Pending job cancelled.', 'success')
    elif batch_queue_repo.request_stop(job_id):
        flash('Stop requested — worker will halt after the current search.', 'success')
    else:
        flash('Job cannot be stopped in its current state.', 'warning')
    return redirect(url_for('batch_queue_list'))


@app.route('/search/<task_id>')
def single_search_progress(task_id):
    """Show progress while a single job search runs."""
    task = _resolve_task(task_id)
    if not task:
        flash('Job search task not found', 'error')
        return redirect(url_for('index'))

    meta = task.get('meta', {})
    return render_template(
        'single_search_progress.html',
        task_id=task_id,
        keyword=meta.get('keyword', ''),
        location=meta.get('location', ''),
        status_url=url_for('cv_task_status', task_id=task_id),
        complete_url=url_for('single_search_complete', task_id=task_id),
        control_url=url_for('control_background_task', task_id=task_id),
        back_url=f"{url_for('index')}#search-jobs",
        back_label='Back to home',
    )


@app.route('/search/complete/<task_id>')
def single_search_complete(task_id):
    """Show jobs found by a completed single search."""
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('Job search result not found', 'error')
        return redirect(url_for('index'))

    result = task['result']
    search_run_id = result.get('search_run_id')
    processed_jobs = sort_jobs(
        job_repo.list_jobs(search_run_id=search_run_id),
        'match_desc',
    )
    session['search_run_id'] = search_run_id

    if result.get('stopped'):
        flash(task.get('message', 'Job search stopped.'), 'warning')

    session.pop('single_search_active', None)

    if not processed_jobs:
        flash('Search completed but no jobs were saved.', 'warning')
        return redirect(url_for('index'))

    return render_template(
        'job_list.html',
        jobs=processed_jobs,
        search_run_id=search_run_id,
        current_sort='match_desc',
        job_sort_options=JOB_SORT_OPTIONS,
    )


def _clear_profile_import_session() -> None:
    session.pop('profile_draft', None)
    session.pop('profile_import_summary', None)


@app.route('/profile', methods=['GET', 'POST'], endpoint='user_profile')
@app.route('/upload_cv', methods=['GET', 'POST'], endpoint='upload_cv')
def user_profile():
    """Create or update the stored CV profile used for generation."""
    if request.method == 'POST':
        if request.form.get('action') == 'discard_import':
            _clear_profile_import_session()
            flash('Import review discarded.', 'info')
            return redirect(url_for('user_profile'))

        profile = profile_from_form(request.form, profile_repo.get_profile())
        if not profile.get('full_name'):
            flash('Full name is required', 'error')
            return render_template(
                'profile_form.html',
                form=profile_to_form_fields(profile),
                import_review=bool(session.get('profile_draft')),
                import_summary=session.get('profile_import_summary'),
            )

        profile_repo.save_profile(profile)
        _clear_profile_import_session()
        flash('Profile saved successfully. You can now generate tailored CVs.', 'success')
        return redirect(url_for('user_profile'))

    form = _profile_form_data()
    return render_template(
        'profile_form.html',
        form=form,
        import_review=bool(session.get('profile_draft')),
        import_summary=session.get('profile_import_summary'),
    )


def _profile_form_data() -> dict:
    """Build form field dict, merging any import draft with stored profile defaults."""
    stored = profile_repo.get_profile()
    defaults = profile_to_form_fields(stored)
    draft = session.get('profile_draft')
    if draft:
        defaults.update(draft)
    return defaults


def _llm_settings_context() -> dict:
    """Build template context for the settings page."""
    all_settings = app_settings_repo.get_settings()
    fast_provider = all_settings["fast_model_provider"]
    main_provider = all_settings["main_model_provider"]
    ollama_settings = all_settings["ollama"]
    alibaba_settings = all_settings["alibaba"]

    ollama_client = OllamaClient(
        base_url=ollama_settings["base_url"],
        fast_model=ollama_settings["fast_model"],
        main_model=ollama_settings["main_model"],
        num_predict=ollama_settings["num_predict"],
    )
    ollama_available = ollama_client.is_available()
    ollama_models = ollama_client.list_models(refresh=True) if ollama_available else []

    alibaba_client = AlibabaClient(
        api_key=alibaba_settings["api_key"],
        base_url=alibaba_settings["base_url"],
        fast_model=alibaba_settings["fast_model"],
        main_model=alibaba_settings["main_model"],
        num_predict=alibaba_settings["num_predict"],
        model_mode=alibaba_settings["model_mode"],
    )
    alibaba_available = alibaba_client.is_available()
    alibaba_models = alibaba_client.list_models(refresh=True) if alibaba_available else list(KNOWN_MODELS)

    from job_apply_ai.cv_modifier.llm_client import build_llm_client

    from job_apply_ai.cv_modifier.alibaba_client import parse_model_pool

    active_client = build_llm_client(all_settings)
    llm_available = active_client.is_available()

    alibaba_fast_pool = parse_model_pool(alibaba_settings["fast_model"])
    alibaba_main_pool = parse_model_pool(alibaba_settings["main_model"])
    alibaba_fast_rotating_count = len(alibaba_client.rotation_pool("fast"))
    alibaba_main_rotating_count = len(alibaba_client.rotation_pool("main"))

    freellmapi_settings = all_settings["freellmapi"]
    freellmapi_client = FreeLLMAPIClient(
        api_key=freellmapi_settings["api_key"],
        base_url=freellmapi_settings["base_url"],
        fast_model=freellmapi_settings["fast_model"],
        main_model=freellmapi_settings["main_model"],
        num_predict=freellmapi_settings["num_predict"],
        model_mode=freellmapi_settings["model_mode"],
        model_state=freellmapi_settings.get("model_state"),
    )
    freellmapi_available = freellmapi_client.is_available()
    freellmapi_models = (
        freellmapi_client.list_models(refresh=True) if freellmapi_available else [AUTO_MODEL]
    )
    freellmapi_fast_pool = parse_model_pool(freellmapi_settings["fast_model"])
    freellmapi_main_pool = parse_model_pool(freellmapi_settings["main_model"])
    freellmapi_fast_rotating_count = len(freellmapi_client.rotation_pool("fast"))
    freellmapi_main_rotating_count = len(freellmapi_client.rotation_pool("main"))

    return {
        "llm_provider": all_settings["llm_provider"],
        "fast_model_provider": fast_provider,
        "main_model_provider": main_provider,
        "ollama_settings": ollama_settings,
        "alibaba_settings": alibaba_settings,
        "has_alibaba_api_key": bool(alibaba_settings.get("api_key")),
        "ollama_available": ollama_available,
        "alibaba_available": alibaba_available,
        "llm_available": llm_available,
        "installed_models": ollama_models + [m for m in alibaba_models if m not in ollama_models],
        "ollama_models": ollama_models,
        "alibaba_models": alibaba_models,
        "active_provider_label": active_client.provider_label,
        "known_alibaba_models": list(KNOWN_MODELS),
        "alibaba_fast_pool": alibaba_fast_pool,
        "alibaba_main_pool": alibaba_main_pool,
        "alibaba_fast_rotating_count": alibaba_fast_rotating_count,
        "alibaba_main_rotating_count": alibaba_main_rotating_count,
        "freellmapi_settings": freellmapi_settings,
        "has_freellmapi_api_key": bool(freellmapi_settings.get("api_key")),
        "freellmapi_available": freellmapi_available,
        "freellmapi_models": freellmapi_models,
        "freellmapi_fast_pool": freellmapi_fast_pool,
        "freellmapi_main_pool": freellmapi_main_pool,
        "freellmapi_fast_rotating_count": freellmapi_fast_rotating_count,
        "freellmapi_main_rotating_count": freellmapi_main_rotating_count,
        "settings": ollama_settings,
        "dev_mode": all_settings.get("dev_mode", False),
        "worker_settings": all_settings["workers"],
    }


@app.route('/settings', methods=['GET', 'POST'], endpoint='app_settings')
def app_settings():
    """Configure LLM provider, models, and generation settings."""
    if request.method == 'POST':
        current = app_settings_repo.get_settings()
        llm_settings = llm_settings_from_form(
            request.form,
            existing_alibaba_api_key=current["alibaba"].get("api_key", ""),
            existing_freellmapi_api_key=current["freellmapi"].get("api_key", ""),
        )
        fast_provider = llm_settings["fast_model_provider"]
        main_provider = llm_settings["main_model_provider"]

        for role, provider in (("fast", fast_provider), ("main", main_provider)):
            provider_settings = llm_settings[provider]
            model_key = f"{role}_model"
            if not provider_settings.get(model_key):
                flash(f'{role.title()} model is required for {provider}.', 'error')
                return render_template('settings.html', **_llm_settings_context())

        if uses_alibaba_provider(llm_settings) and not llm_settings["alibaba"].get("api_key"):
            flash('Alibaba Cloud API key is required when Alibaba is selected for fast or main models.', 'error')
            return render_template('settings.html', **_llm_settings_context())

        if uses_freellmapi_provider(llm_settings) and not llm_settings["freellmapi"].get("api_key"):
            flash('FreeLLMAPI unified API key is required when FreeLLMAPI is selected for fast or main models.', 'error')
            return render_template('settings.html', **_llm_settings_context())

        alibaba = llm_settings["alibaba"]
        if alibaba.get("model_mode") in ("round_robin", "auto"):
            from job_apply_ai.cv_modifier.alibaba_client import parse_model_pool

            for role, provider in (("fast", fast_provider), ("main", main_provider)):
                if provider != "alibaba":
                    continue
                pool = parse_model_pool(alibaba[f"{role}_model"])
                if len(pool) < 2:
                    flash(
                        f'Alibaba {role} model pool was expanded to multiple models for {alibaba["model_mode"]} mode. '
                        'Review the comma-separated list and save again if needed.',
                        'warning',
                    )
                    break

        llm_settings["dev_mode"] = request.form.get("dev_mode") == "on"
        llm_settings["workers"] = worker_settings_from_form(request.form)
        app_settings_repo.save_llm_settings(llm_settings)
        invalidate_dev_mode_cache()
        flash('Settings saved. Model and worker changes apply to the next AI task.', 'success')
        return redirect(url_for('app_settings'))

    current = app_settings_repo.get_settings()
    alibaba = current["alibaba"]
    if alibaba.get("model_mode") in ("round_robin", "auto"):
        from job_apply_ai.cv_modifier.alibaba_client import parse_model_pool

        repaired = ensure_alibaba_rotation_pools(alibaba)
        needs_repair = any(
            len(parse_model_pool(alibaba[key])) < 2 and parse_model_pool(repaired[key]) != parse_model_pool(alibaba[key])
            for key in ("fast_model", "main_model")
        )
        if needs_repair:
            saved = dict(alibaba)
            saved["fast_model"] = repaired["fast_model"]
            saved["main_model"] = repaired["main_model"]
            app_settings_repo.save_alibaba_settings(saved)
            flash(
                'Alibaba model pools were expanded with additional Qwen models so rotation can work. '
                'Review the model lists below.',
                'info',
            )

    return render_template('settings.html', **_llm_settings_context())


@app.route('/dev/logs')
def dev_logs_page():
    """Developer log viewer (requires dev mode)."""
    if not app_settings_repo.get_dev_mode():
        flash('Enable Developer mode in Settings to view developer logs.', 'warning')
        return redirect(url_for('app_settings'))
    repo = DevLogRepository()
    return render_template(
        'dev_logs.html',
        categories=DEV_LOG_CATEGORIES,
        agents=repo.list_agents(),
        total_logs=repo.count_logs(),
    )


@app.route('/api/dev/logs')
def api_dev_logs():
    """JSON API for developer logs."""
    if not app_settings_repo.get_dev_mode():
        return jsonify({'error': 'Developer mode is disabled'}), 403
    repo = DevLogRepository()
    category = request.args.get('category', '').strip()
    agent = request.args.get('agent', '').strip()
    search = request.args.get('search', '').strip()
    task_id = request.args.get('task_id', '').strip()
    since_id = request.args.get('since_id', 0, type=int)
    limit = request.args.get('limit', 200, type=int)
    offset = request.args.get('offset', 0, type=int)
    logs = repo.list_logs(
        category=category,
        agent=agent,
        search=search,
        task_id=task_id,
        since_id=since_id,
        limit=limit,
        offset=offset,
    )
    return jsonify({
        'logs': logs,
        'total': repo.count_logs(category=category, agent=agent, search=search, task_id=task_id),
    })


@app.route('/api/dev/logs', methods=['DELETE'])
def api_dev_logs_clear():
    """Clear developer logs (all or filtered)."""
    if not app_settings_repo.get_dev_mode():
        return jsonify({'error': 'Developer mode is disabled'}), 403
    payload = request.get_json(silent=True) or {}
    category = str(payload.get('category') or request.args.get('category', '')).strip()
    task_id = str(payload.get('task_id') or request.args.get('task_id', '')).strip()
    deleted = DevLogRepository().clear_logs(category=category, task_id=task_id)
    return jsonify({'deleted': deleted})


@app.route('/api/dev/logs/download')
def api_dev_logs_download():
    """Download developer logs as JSON or plain text."""
    if not app_settings_repo.get_dev_mode():
        return jsonify({'error': 'Developer mode is disabled'}), 403
    repo = DevLogRepository()
    category = request.args.get('category', '').strip()
    agent = request.args.get('agent', '').strip()
    search = request.args.get('search', '').strip()
    task_id = request.args.get('task_id', '').strip()
    fmt = request.args.get('format', 'json').strip().lower()
    logs = repo.list_logs(
        category=category,
        agent=agent,
        search=search,
        task_id=task_id,
        limit=10000,
        offset=0,
    )
    logs.reverse()

    if fmt == 'txt':
        lines = []
        for entry in logs:
            lines.append(
                f"[{entry['created_at']}] {entry['category']}/{entry['agent']}/{entry['event']}: {entry['message']}"
            )
            if entry.get('data'):
                lines.append(json.dumps(entry['data'], ensure_ascii=False, indent=2))
            lines.append('')
        body = '\n'.join(lines)
        return Response(
            body,
            mimetype='text/plain; charset=utf-8',
            headers={'Content-Disposition': 'attachment; filename=dev-logs.txt'},
        )

    body = json.dumps(logs, ensure_ascii=False, indent=2)
    return Response(
        body,
        mimetype='application/json; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=dev-logs.json'},
    )


@app.route('/profile/oauth/google/start')
def oauth_google_start():
    """Begin Google OAuth for Gmail sending."""
    if not google_oauth_configured():
        flash('Google OAuth is not configured. Add GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET to .env.', 'error')
        return redirect(url_for('user_profile'))

    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    redirect_uri = _oauth_redirect_uri('oauth_google_callback')
    settings = google_oauth_settings(redirect_uri)
    auth_url, code_verifier = google_authorization_url(settings, state)
    session['oauth_google_code_verifier'] = code_verifier
    return redirect(auth_url)


@app.route('/profile/oauth/google/callback')
def oauth_google_callback():
    """Complete Google OAuth and store the connected Gmail account."""
    expected_state = session.pop('oauth_state', None)
    code_verifier = session.pop('oauth_google_code_verifier', '')
    if request.args.get('state') != expected_state:
        flash('Google sign-in failed: invalid OAuth state. Please try again.', 'error')
        return redirect(url_for('user_profile'))

    if request.args.get('error'):
        flash(f"Google sign-in cancelled: {request.args.get('error_description') or request.args.get('error')}", 'warning')
        return redirect(url_for('user_profile'))

    code = request.args.get('code', '').strip()
    if not code:
        flash('Google sign-in failed: no authorization code received.', 'error')
        return redirect(url_for('user_profile'))

    try:
        redirect_uri = _oauth_redirect_uri('oauth_google_callback')
        settings = google_oauth_settings(redirect_uri)
        oauth_data = exchange_google_code(
            settings,
            code,
            request.args.get('state', ''),
            code_verifier=code_verifier,
        )
        if not oauth_data.get('oauth_refresh_token'):
            raise RuntimeError('Google did not return a refresh token. Remove app access and connect again.')

        profile = upsert_oauth_smtp_account(
            profile_repo.get_profile(),
            provider='gmail',
            email=oauth_data['email'],
            oauth_refresh_token=oauth_data['oauth_refresh_token'],
            oauth_access_token=oauth_data.get('oauth_access_token', ''),
            oauth_expires_at=oauth_data.get('oauth_expires_at', ''),
        )
        profile_repo.save_profile(profile)
        flash(f"Connected Gmail account {oauth_data['email']}.", 'success')
    except Exception as exc:
        logger.error('Google OAuth callback failed: %s', exc)
        flash(f'Google sign-in failed: {exc}', 'error')
    return redirect(url_for('user_profile'))


@app.route('/profile/oauth/microsoft/start')
def oauth_microsoft_start():
    """Begin Microsoft OAuth for Outlook/Hotmail sending."""
    if not microsoft_oauth_configured():
        flash('Microsoft OAuth is not configured. Add MICROSOFT_OAUTH_CLIENT_ID and MICROSOFT_OAUTH_CLIENT_SECRET to .env.', 'error')
        return redirect(url_for('user_profile'))

    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    redirect_uri = _oauth_redirect_uri('oauth_microsoft_callback')
    settings = microsoft_oauth_settings(redirect_uri)
    return redirect(microsoft_authorization_url(settings, state))


@app.route('/profile/oauth/microsoft/callback')
def oauth_microsoft_callback():
    """Complete Microsoft OAuth and store the connected Outlook/Hotmail account."""
    expected_state = session.pop('oauth_state', None)
    if request.args.get('state') != expected_state:
        flash('Microsoft sign-in failed: invalid OAuth state. Please try again.', 'error')
        return redirect(url_for('user_profile'))

    if request.args.get('error'):
        flash(f"Microsoft sign-in cancelled: {request.args.get('error_description') or request.args.get('error')}", 'warning')
        return redirect(url_for('user_profile'))

    code = request.args.get('code', '').strip()
    if not code:
        flash('Microsoft sign-in failed: no authorization code received.', 'error')
        return redirect(url_for('user_profile'))

    try:
        redirect_uri = _oauth_redirect_uri('oauth_microsoft_callback')
        settings = microsoft_oauth_settings(redirect_uri)
        oauth_data = exchange_microsoft_code(settings, code)
        if not oauth_data.get('oauth_refresh_token'):
            raise RuntimeError('Microsoft did not return a refresh token. Reconnect and accept all permissions.')

        provider = 'outlook'
        email = oauth_data['email'].lower()
        if email.endswith('@hotmail.com') or email.endswith('@live.com'):
            provider = 'hotmail'

        profile = upsert_oauth_smtp_account(
            profile_repo.get_profile(),
            provider=provider,
            email=oauth_data['email'],
            oauth_refresh_token=oauth_data['oauth_refresh_token'],
            oauth_access_token=oauth_data.get('oauth_access_token', ''),
            oauth_expires_at=oauth_data.get('oauth_expires_at', ''),
        )
        profile_repo.save_profile(profile)
        flash(f"Connected Microsoft account {oauth_data['email']}.", 'success')
    except Exception as exc:
        logger.error('Microsoft OAuth callback failed: %s', exc)
        flash(f'Microsoft sign-in failed: {exc}', 'error')
    return redirect(url_for('user_profile'))


@app.route('/profile/oauth/disconnect/<account_id>', methods=['POST'])
def oauth_disconnect_account(account_id):
    """Remove a connected OAuth or SMTP account."""
    profile = remove_smtp_account(profile_repo.get_profile(), account_id)
    profile_repo.save_profile(profile)
    flash('Sending account disconnected.', 'info')
    return redirect(url_for('user_profile'))


@app.route('/profile/oauth/default/<account_id>', methods=['POST'])
def oauth_set_default_account(account_id):
    """Set the default sending account."""
    profile = set_default_smtp_account(profile_repo.get_profile(), account_id)
    profile_repo.save_profile(profile)
    flash('Default sending account updated.', 'success')
    return redirect(url_for('user_profile'))


LINKEDIN_SYNC_SESSION_KEY = 'linkedin_sync_profile'


def _linkedin_sync_snapshot() -> dict[str, Any] | None:
    raw = session.get(LINKEDIN_SYNC_SESSION_KEY)
    return raw if isinstance(raw, dict) else None


def _store_linkedin_sync_snapshot(profile: dict[str, Any]) -> None:
    snapshot = dict(profile)
    snapshot.pop('_raw_sections', None)
    session[LINKEDIN_SYNC_SESSION_KEY] = snapshot


def _linkedin_sync_response(local_profile: dict[str, Any], linkedin_profile: dict[str, Any]) -> dict[str, Any]:
    diffs = compare_profiles(local_profile, linkedin_profile)
    return {
        'ok': True,
        'diffs': diffs,
        'summary': diff_summary(diffs),
        'linkedin_url': linkedin_profile.get('_linkedin_url') or linkedin_profile.get('linkedin') or '',
    }


@app.route('/profile/linkedin-sync')
def linkedin_profile_sync():
    """Compare HermesHire profile with LinkedIn and reconcile differences."""
    local_profile = profile_repo.get_profile()
    linkedin_profile = _linkedin_sync_snapshot()
    initial_diffs = compare_profiles(local_profile, linkedin_profile) if linkedin_profile else []
    return render_template(
        'profile_linkedin_sync.html',
        mcp_healthy=check_linkedin_mcp_health(),
        initial_diffs=initial_diffs,
        initial_summary=diff_summary(initial_diffs) if initial_diffs else None,
        linkedin_url=(linkedin_profile or {}).get('_linkedin_url') or (linkedin_profile or {}).get('linkedin') or '',
    )


@app.route('/profile/linkedin-sync/fetch', methods=['POST'])
def linkedin_profile_sync_fetch():
    """Fetch LinkedIn profile via MCP and return diff rows."""
    try:
        linkedin_profile = fetch_linkedin_profile()
    except LinkedInMcpError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 503

    _store_linkedin_sync_snapshot(linkedin_profile)
    local_profile = profile_repo.get_profile()
    return jsonify(_linkedin_sync_response(local_profile, linkedin_profile))


@app.route('/profile/linkedin-sync/apply', methods=['POST'])
def linkedin_profile_sync_apply():
    """Apply one sync action to the local profile or prepare a manual LinkedIn edit."""
    payload = request.get_json(silent=True) or {}
    diff_id = str(payload.get('diff_id') or '').strip()
    action = str(payload.get('action') or '').strip()
    if not diff_id or not action:
        return jsonify({'ok': False, 'error': 'diff_id and action are required.'}), 400

    linkedin_profile = _linkedin_sync_snapshot()
    if not linkedin_profile:
        return jsonify({'ok': False, 'error': 'Fetch your LinkedIn profile first.'}), 400

    local_profile = profile_repo.get_profile()
    try:
        updated_profile, result = apply_sync_action(local_profile, linkedin_profile, diff_id, action)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400

    if result.get('applied'):
        profile_repo.save_profile(updated_profile)

    response = _linkedin_sync_response(updated_profile if result.get('applied') else local_profile, linkedin_profile)
    response['result'] = result
    return jsonify(response)


@app.route('/profile/export')
def export_profile():
    """Download the stored profile as a JSON file."""
    profile = profile_repo.get_profile()
    payload = profile_to_export_dict(profile)
    name_part = sanitize_filename(profile.get('full_name') or 'profile')
    filename = f"{name_part}_profile.json"
    buffer = io.BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'))
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/json',
    )


@app.route('/profile/import/json', methods=['POST'])
def import_profile_json():
    """Import profile data from a JSON file for review before saving."""
    if 'profile_file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('user_profile'))

    file = request.files['profile_file']
    if not file.filename:
        flash('No file selected', 'error')
        return redirect(url_for('user_profile'))
    if not file.filename.lower().endswith('.json'):
        flash('Please upload a .json profile file', 'error')
        return redirect(url_for('user_profile'))

    try:
        raw = json.loads(file.read().decode('utf-8'))
        imported = profile_from_export_dict(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        flash(f'Invalid profile JSON file: {exc}', 'error')
        return redirect(url_for('user_profile'))
    except ValueError as exc:
        flash(str(exc), 'error')
        return redirect(url_for('user_profile'))

    replace = request.form.get('replace') == '1'
    current = profile_repo.get_profile()

    if replace:
        merged = imported
        summary_lines = ['Replaced profile with imported data']
        has_changes = True
    else:
        merged, changes = merge_profiles(current, imported)
        summary_lines = summarize_import_changes(changes)
        has_changes = import_has_changes(changes)

    session['profile_draft'] = profile_to_form_fields(merged)
    session['profile_import_summary'] = summary_lines

    if has_changes:
        flash('Profile imported. Review the updates and click Save Profile to apply them.', 'success')
    else:
        flash('Profile imported, but no new details were found beyond your current profile.', 'info')

    return redirect(url_for('user_profile'))


@app.route('/backup/export')
def export_backup_route():
    """Download a full backup zip (profile, jobs, search history, CV artifacts)."""
    from job_apply_ai.storage.backup import backup_filename, export_backup

    profile = profile_repo.get_profile()
    buffer = export_backup(data_dir=app.config['UPLOAD_FOLDER'])
    return send_file(
        buffer,
        as_attachment=True,
        download_name=backup_filename(profile),
        mimetype='application/zip',
    )


@app.route('/backup/import', methods=['POST'])
def import_backup_route():
    """Restore profile, jobs, and CV artifacts from a backup zip."""
    from job_apply_ai.storage.backup import restore_backup

    if 'backup_file' not in request.files:
        flash('No backup file selected', 'error')
        return redirect(url_for('user_profile'))

    file = request.files['backup_file']
    if not file.filename:
        flash('No backup file selected', 'error')
        return redirect(url_for('user_profile'))
    if not file.filename.lower().endswith('.zip'):
        flash('Please upload a .zip backup file', 'error')
        return redirect(url_for('user_profile'))

    replace = request.form.get('replace') == '1'
    include_task_queue = request.form.get('restore_task_queue') == '1'
    include_settings = request.form.get('restore_settings') == '1'
    include_all_others = request.form.get('restore_all_others') == '1'
    if not (include_task_queue or include_settings or include_all_others):
        flash('Select at least one section to restore.', 'error')
        return redirect(url_for('user_profile'))

    try:
        stats = restore_backup(
            file.read(),
            data_dir=app.config['UPLOAD_FOLDER'],
            replace=replace,
            merge_profile=not replace,
            include_task_queue=include_task_queue,
            include_settings=include_settings,
            include_all_others=include_all_others,
        )
    except (ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        flash(f'Could not restore backup: {exc}', 'error')
        return redirect(url_for('user_profile'))

    parts: list[str] = []
    if include_all_others:
        parts.extend([
            f"{stats['jobs_restored']} job(s)",
            f"{stats['search_runs_restored']} search run(s)",
            f"{stats['cv_sidecars_restored']} CV history file(s)",
            f"{stats['dev_logs_restored']} dev log(s)",
            f"{stats['files_restored']} file(s)",
        ])
    if include_task_queue:
        parts.append(f"{stats['batch_jobs_restored']} batch job(s)")
    if include_settings and stats['settings_restored']:
        parts.append('settings updated')
    flash(f"Backup restored: {', '.join(parts) or 'no changes'}.", 'success')
    return redirect(url_for('user_profile'))


@app.route('/profile/import/start', methods=['POST'])
def start_profile_import():
    """Upload a CV and extract profile data in the background."""
    if 'cv_file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('user_profile'))

    file = request.files['cv_file']
    if not file.filename:
        flash('No file selected', 'error')
        return redirect(url_for('user_profile'))
    if not file.filename.lower().endswith('.docx'):
        flash('Please upload a .docx CV file', 'error')
        return redirect(url_for('user_profile'))

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'profile_import_{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.docx')
    file.save(temp_path)

    task_id = _enqueue_ai_task(
        'profile_import',
        {'cv_path': temp_path},
    )
    return redirect(url_for('profile_import_progress', task_id=task_id))


@app.route('/profile/import/<task_id>')
def profile_import_progress(task_id):
    """Show progress while a CV import is being parsed."""
    task = _resolve_task(task_id)
    if not task:
        flash('Import task not found', 'error')
        return redirect(url_for('user_profile'))

    return render_template(
        'profile_import_progress.html',
        task_id=task_id,
        status_url=url_for('cv_task_status', task_id=task_id),
        complete_url=url_for('profile_import_complete', task_id=task_id),
        back_url=url_for('user_profile'),
    )


@app.route('/profile/import/<task_id>/status')
def profile_import_status(task_id):
    """Compatibility endpoint for import progress polling."""
    return cv_task_status(task_id)


@app.route('/profile/import/complete/<task_id>')
def profile_import_complete(task_id):
    """Load merged profile draft for user review and approval."""
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('Profile import result not found', 'error')
        return redirect(url_for('user_profile'))

    result = task['result']
    session['profile_draft'] = result.get('form', {})
    session['profile_import_summary'] = result.get('import_summary', [])

    if result.get('has_changes'):
        flash('CV imported. Review the highlighted updates and click Save Profile to apply them.', 'success')
    else:
        flash('CV imported, but no new details were found beyond your current profile.', 'info')

    return redirect(url_for('user_profile'))

@app.route('/job_list')
def job_list():
    """Display the list of jobs with Make CV buttons."""
    search_run_id = session.get('search_run_id')
    sort_by = validate_job_sort(request.args.get('sort'))
    jobs = _get_jobs_for_view(search_run_id, sort_by=sort_by)

    if not jobs:
        flash('No jobs found. Please search for jobs first.', 'warning')
        return redirect(url_for('index'))

    sort_query = sort_by if sort_by != DEFAULT_JOB_SORT else None

    return render_template(
        'job_list.html',
        jobs=jobs,
        search_run_id=search_run_id,
        current_sort=sort_by,
        sort_query=sort_query,
        job_sort_options=JOB_SORT_OPTIONS,
        job_statuses=JOB_WORKFLOW_STATUSES,
        status_labels=JOB_STATUS_LABELS,
        **_job_move_history_context(),
    )


@app.route('/jobs/manage')
@app.route('/jobs/manage/<folder>')
def manage_jobs(folder='all'):
    """Display jobs grouped by workflow status with folder navigation."""
    search = request.args.get('q', '').strip()
    sort_by = validate_job_sort(request.args.get('sort'))

    if folder != 'all' and not is_valid_job_status(folder):
        flash('Unknown job folder', 'warning')
        return redirect(url_for('manage_jobs'))

    workflow_status = None if folder == 'all' else folder
    exclude_statuses = ['archived'] if folder == 'all' else None
    jobs = sort_jobs(
        job_repo.list_jobs(
            workflow_status=workflow_status,
            search=search or None,
            exclude_workflow_statuses=exclude_statuses,
        ),
        sort_by,
    )
    status_counts = job_repo.count_jobs_by_status()
    total_count = sum(status_counts.values())

    folder_counts = {
        'all': total_count - status_counts.get('archived', 0),
    }
    for status in JOB_WORKFLOW_STATUSES:
        folder_counts[status] = status_counts.get(status, 0)

    sort_query = sort_by if sort_by != DEFAULT_JOB_SORT else None
    jobs_with_cv_count = sum(1 for job in jobs if _job_has_sendable_cv(job))
    profile = profile_repo.get_profile()

    return render_template(
        'manage_jobs.html',
        jobs=jobs,
        current_folder=folder,
        search_query=search,
        current_sort=sort_by,
        sort_query=sort_query,
        job_sort_options=JOB_SORT_OPTIONS,
        folder_counts=folder_counts,
        job_statuses=JOB_WORKFLOW_STATUSES,
        status_labels=JOB_STATUS_LABELS,
        status_icons=JOB_STATUS_ICONS,
        status_badges=JOB_STATUS_BADGE_CLASSES,
        profile_has_matchable_skills=profile_has_matchable_skills(profile),
        profile_ready=profile_is_ready(profile),
        jobs_with_cv_count=jobs_with_cv_count,
        **_job_move_history_context(),
    )


def _jobs_for_manage_folder(folder: str, search: str) -> list[dict]:
    workflow_status = None if folder == 'all' else folder
    exclude_statuses = ['archived'] if folder == 'all' else None
    return job_repo.list_jobs(
        workflow_status=workflow_status,
        search=search or None,
        exclude_workflow_statuses=exclude_statuses,
    )


def _run_job_match_analyze_task(
    task_id: str,
    jobs: list[dict],
    profile: dict,
    min_match_score: float,
    return_folder: str,
    return_search: str,
    return_sort: str = '',
) -> None:
    with dev_task(task_id, "job_match_analyze"):
        total = len(jobs)

        def on_progress(index: int, _total: int, job: dict) -> None:
            percent = 5 + int(((index + 1) / max(total, 1)) * 90)
            update_task(
                task_id,
                status='running',
                step='analyzing',
                message=f"Analyzing job {index + 1} of {total}: {job.get('title', 'Untitled')}",
                percent=percent,
                meta={
                    'current_index': index + 1,
                    'total_jobs': total,
                    'current_job_title': job.get('title', ''),
                },
            )

        def should_continue() -> bool:
            try:
                task_control_checkpoint(task_id)
                return True
            except TaskStopped:
                return False

        try:
            update_task(
                task_id,
                status='running',
                step='starting',
                message='Starting AI profile match analysis…',
                percent=5,
                meta={'total_jobs': total, 'min_match_score': min_match_score},
            )
            result = analyze_jobs_with_threshold(
                jobs,
                profile,
                min_match_score,
                on_progress=on_progress,
                should_continue=should_continue,
            )

            status_changes: list[dict] = []
            for job, updated in zip(jobs, result['jobs']):
                job_id = job.get('id')
                if not job_id:
                    continue
                job_repo.update_job(
                    job_id,
                    {
                        'matched_categories': updated.get('matched_categories', {}),
                        'matched_skills': updated.get('matched_skills', job.get('matched_skills', [])),
                    },
                )
                previous_status = job.get('workflow_status') or DEFAULT_JOB_STATUS
                new_status = updated.get('workflow_status') or previous_status
                if new_status != previous_status:
                    status_changes.extend(job_repo.move_jobs_status([job_id], new_status))

            stats = result['stats']
            current_task = get_task(task_id)
            stopped = bool(current_task and current_task.get('control') == 'stop')
            if stopped:
                analyzed = stats.get('analyzed', 0)
                if analyzed == 0:
                    fail_task(task_id, 'Profile match analysis stopped before any jobs were analyzed.')
                    return
                complete_task(
                    task_id,
                    {
                        'stats': stats,
                        'status_changes': status_changes,
                        'return_folder': return_folder,
                        'return_search': return_search,
                        'return_sort': return_sort,
                        'stopped': True,
                    },
                    message=f'Profile match analysis stopped — analyzed {analyzed} of {total} jobs',
                )
                return

            complete_task(
                task_id,
                {
                    'stats': stats,
                    'status_changes': status_changes,
                    'return_folder': return_folder,
                    'return_search': return_search,
                    'return_sort': return_sort,
                },
            )
            update_task(
                task_id,
                step='complete',
                message='Profile match analysis complete',
                percent=100,
            )
        except Exception as exc:
            logger.error('Job match analysis failed: %s', exc)
            fail_task(task_id, str(exc))


@app.route('/jobs/manage/analyze-match', methods=['POST'])
def analyze_jobs_match():
    """Analyze jobs in the current folder against the profile and route low matches."""
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '').strip()
    return_sort = request.form.get('return_sort', '').strip()
    min_match_score = normalize_min_match_score(request.form.get('min_match_score'))

    profile = profile_repo.get_profile()
    if not profile_has_matchable_skills(profile):
        flash('Add technical skills or stacks on your profile before running match analysis.', 'warning')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    jobs = _jobs_for_manage_folder(return_folder, return_search)
    if not jobs:
        flash('No jobs to analyze in this folder.', 'warning')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    task_id = _enqueue_ai_task(
        'job_match_analyze',
        {
            'return_folder': return_folder,
            'return_search': return_search,
            'return_sort': return_sort,
            'min_match_score': min_match_score,
            'meta': {
                'total_jobs': len(jobs),
                'min_match_score': min_match_score,
                'return_folder': return_folder,
                'return_search': return_search,
                'return_sort': return_sort,
            },
        },
    )
    session['job_match_analyze_active'] = task_id
    return redirect(url_for('job_match_analyze_progress', task_id=task_id))


@app.route('/jobs/manage/analyze-match/<task_id>')
def job_match_analyze_progress(task_id):
    """Show progress while jobs are analyzed against the profile."""
    task = _resolve_task(task_id)
    if not task or task.get('task_type') != 'job_match_analyze':
        flash('Match analysis task not found', 'error')
        return redirect(url_for('manage_jobs'))

    meta = task.get('meta', {})
    return_folder = meta.get('return_folder') or 'all'
    return_search = meta.get('return_search') or ''
    return_sort = meta.get('return_sort') or ''

    return render_template(
        'job_match_progress.html',
        task_id=task_id,
        status_url=url_for('cv_task_status', task_id=task_id),
        complete_url=url_for('job_match_analyze_complete', task_id=task_id),
        control_url=url_for('control_background_task', task_id=task_id),
        back_url=url_for(
            'manage_jobs',
            **_manage_jobs_url_kwargs(return_folder, return_search, return_sort),
        ),
        back_label='Back to Manage Jobs',
        min_match_score=meta.get('min_match_score', 50),
        job_count=meta.get('total_jobs', 0),
    )


@app.route('/jobs/manage/analyze-match/complete/<task_id>')
def job_match_analyze_complete(task_id):
    """Finish match analysis and return to manage jobs."""
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('Match analysis result not found', 'error')
        return redirect(url_for('manage_jobs'))

    result = task['result']
    stats = result.get('stats', {})
    threshold = stats.get('min_match_score', 50)
    analyzed = stats.get('analyzed', 0)
    moved = stats.get('moved_to_not_match', 0)
    restored = stats.get('restored_to_new', 0)

    session.pop('job_match_analyze_active', None)

    status_changes = result.get('status_changes', [])
    if status_changes:
        moved = stats.get('moved_to_not_match', 0)
        restored = stats.get('restored_to_new', 0)
        label_parts = []
        if moved:
            label_parts.append(f'moved {moved} to Not Match')
        if restored:
            label_parts.append(f'restored {restored} to New')
        history_label = 'AI match analysis'
        if label_parts:
            history_label = f'AI match analysis ({", ".join(label_parts)})'
        _record_job_status_changes(status_changes, history_label=history_label)

    if result.get('stopped'):
        flash(
            task.get('message', f'Analysis stopped after {analyzed} job(s).'),
            'warning',
        )
    else:
        flash(
            f'Analyzed {analyzed} job(s) with a minimum match score of {threshold:g}%. '
            f'Moved {moved} to Not Match With You'
            + (f' and restored {restored} to New / Discovered.' if restored else '.'),
            'success',
        )

    return _manage_jobs_redirect(
        result.get('return_folder', 'all'),
        result.get('return_search', ''),
        result.get('return_sort') or 'match_desc',
    )


@app.route('/jobs/manage/batch-ats-friendly', methods=['POST'])
def batch_ats_friendly():
    """Run three ATS passes on every job in the folder that has a generated CV."""
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '').strip()
    return_sort = request.form.get('return_sort', '').strip()

    profile = profile_repo.get_profile()
    if not profile_is_ready(profile):
        flash('Please complete your CV profile before running batch ATS optimization.', 'warning')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    if _sync_session_background_task('batch_ats_friendly_active'):
        flash('Batch ATS optimization is already in progress.', 'warning')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    jobs = _jobs_for_manage_folder(return_folder, return_search)
    jobs_with_cv = _jobs_with_cv(jobs)
    if not jobs_with_cv:
        flash('No jobs with generated CVs found in this folder.', 'warning')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    task_id = _enqueue_ai_task(
        'batch_ats_friendly',
        {
            'return_folder': return_folder,
            'return_search': return_search,
            'return_sort': return_sort,
            'meta': {
                'total_jobs': len(jobs_with_cv),
                'passes': len(BATCH_ATS_FRIENDLY_PASSES),
                'return_folder': return_folder,
                'return_search': return_search,
                'return_sort': return_sort,
            },
        },
    )
    session['batch_ats_friendly_active'] = task_id
    return redirect(url_for('batch_ats_friendly_progress', task_id=task_id))


@app.route('/jobs/manage/batch-ats-friendly/<task_id>')
def batch_ats_friendly_progress(task_id):
    """Show progress while batch ATS optimization runs."""
    task = _resolve_task(task_id)
    if not task or task.get('task_type') != 'batch_ats_friendly':
        flash('Batch ATS optimization task not found', 'error')
        return redirect(url_for('manage_jobs'))

    meta = task.get('meta', {})
    return_folder = meta.get('return_folder') or 'all'
    return_search = meta.get('return_search') or ''
    return_sort = meta.get('return_sort') or ''

    return render_template(
        'batch_ats_friendly_progress.html',
        task_id=task_id,
        status_url=url_for('cv_task_status', task_id=task_id),
        complete_url=url_for('batch_ats_friendly_complete', task_id=task_id),
        control_url=url_for('control_background_task', task_id=task_id),
        back_url=url_for(
            'manage_jobs',
            **_manage_jobs_url_kwargs(return_folder, return_search, return_sort),
        ),
        back_label='Back to Manage Jobs',
        job_count=meta.get('total_jobs', 0),
        pass_count=meta.get('passes', len(BATCH_ATS_FRIENDLY_PASSES)),
    )


@app.route('/jobs/manage/batch-ats-friendly/complete/<task_id>')
def batch_ats_friendly_complete(task_id):
    """Finish batch ATS optimization and return to manage jobs."""
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('Batch ATS optimization result not found', 'error')
        return redirect(url_for('manage_jobs'))

    result = task['result']
    stats = result.get('stats', {})
    total_jobs = stats.get('total_jobs', 0)
    suggestions_applied = stats.get('suggestions_applied', 0)
    failed = stats.get('failed', 0)

    session.pop('batch_ats_friendly_active', None)

    if result.get('stopped'):
        flash(task.get('message', 'Batch ATS optimization stopped.'), 'warning')
    elif failed:
        flash(
            f'Batch ATS optimization finished for {total_jobs} job(s) with {failed} failure(s). '
            f'Applied {suggestions_applied} suggestion(s) in passes 1 and 2. '
            f'Pass 3 left suggestions for manual review.',
            'warning',
        )
    else:
        flash(
            f'Batch ATS optimization complete for {total_jobs} job(s). '
            f'Applied {suggestions_applied} suggestion(s) in passes 1 and 2. '
            f'Pass 3 left suggestions for manual review.',
            'success',
        )

    return _manage_jobs_redirect(
        result.get('return_folder', 'all'),
        result.get('return_search', ''),
        result.get('return_sort', ''),
    )


@app.route('/jobs/new/linkedin', methods=['GET', 'POST'])
def create_job_from_linkedin():
    """Import a job by scraping a LinkedIn job share link."""
    return_folder = (
        request.form.get('return_folder')
        or request.args.get('folder', 'all')
    )
    return_search = (
        request.form.get('return_search')
        or request.args.get('q', '')
    )

    if request.method == 'POST':
        raw_url = (request.form.get('linkedin_url') or '').strip()
        linkedin_url = parse_linkedin_job_url(raw_url)
        if not linkedin_url:
            flash('Please enter a valid LinkedIn job link.', 'error')
            return render_template(
                'job_import_linkedin.html',
                linkedin_url=raw_url,
                return_folder=return_folder,
                return_search=return_search,
            )

        task_id = _enqueue_urgent_task(
            'linkedin_job_import',
            {
                'linkedin_url': linkedin_url,
                'return_folder': return_folder,
                'return_search': return_search,
                'meta': {'linkedin_url': linkedin_url},
            },
        )
        flash(
            'LinkedIn import queued — the urgent worker will scrape it shortly '
            '(run: job-apply-ai urgent-worker)',
            'success',
        )
        return redirect(url_for('linkedin_job_import_progress', task_id=task_id))

    return render_template(
        'job_import_linkedin.html',
        linkedin_url='',
        return_folder=return_folder,
        return_search=return_search,
    )


@app.route('/jobs/import/linkedin/<task_id>')
def linkedin_job_import_progress(task_id):
    """Show progress while a LinkedIn job is being scraped."""
    task = _resolve_task(task_id)
    if not task:
        flash('Import task not found', 'error')
        return redirect(url_for('create_job_from_linkedin'))

    return render_template(
        'job_import_progress.html',
        task_id=task_id,
        status_url=url_for('cv_task_status', task_id=task_id),
        complete_url=url_for('linkedin_job_import_complete', task_id=task_id),
        back_url=url_for('create_job_from_linkedin'),
    )


@app.route('/jobs/import/linkedin/complete/<task_id>')
def linkedin_job_import_complete(task_id):
    """Save the scraped job and open it for review."""
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('LinkedIn import result not found', 'error')
        return redirect(url_for('create_job_from_linkedin'))

    result = task['result']
    job = result.get('job', {})
    if not job.get('title'):
        flash(
            'Imported job is missing a title. The listing may require sign-in — try again or add manually.',
            'error',
        )
        return redirect(url_for('create_job_from_linkedin'))

    try:
        job_id = job_repo.create_job(job)
    except ValueError as exc:
        flash(f'Could not save imported job: {exc}', 'error')
        return redirect(url_for('create_job_from_linkedin'))

    flash(
        f'Job #{job_id} imported from LinkedIn. Review the details and save if you change anything.',
        'success',
    )
    return redirect(
        url_for(
            'edit_job',
            job_id=job_id,
            folder=result.get('return_folder', 'all'),
            q=result.get('return_search') or None,
        )
    )


@app.route('/jobs/new', methods=['GET', 'POST'])
def create_job():
    """Create a job manually."""
    if request.method == 'POST':
        job_data = _job_form_data()
        if not job_data.get('title'):
            flash('Job title is required', 'error')
            return render_template(
                'job_form.html',
                job=job_data,
                job_statuses=JOB_WORKFLOW_STATUSES,
                status_labels=JOB_STATUS_LABELS,
                return_folder=request.form.get('return_folder', 'all'),
                return_search=request.form.get('return_search', ''),
            )

        job_id = job_repo.create_job(job_data)
        flash(f'Job #{job_id} created successfully', 'success')
        return _manage_jobs_redirect(
            request.form.get('return_folder', 'all'),
            request.form.get('return_search', ''),
        )

    return render_template(
        'job_form.html',
        job=None,
        job_statuses=JOB_WORKFLOW_STATUSES,
        status_labels=JOB_STATUS_LABELS,
        return_folder=request.args.get('folder', 'all'),
        return_search=request.args.get('q', ''),
    )


@app.route('/jobs/<int:job_id>/edit', methods=['GET', 'POST'])
def edit_job(job_id):
    """Update an existing job."""
    job = job_repo.get_job(job_id)
    if not job:
        flash('Job not found', 'error')
        return _manage_jobs_redirect()

    return_folder = request.args.get('folder') or request.form.get('return_folder', 'all')
    return_search = request.args.get('q') or request.form.get('return_search', '')
    return_sort = request.args.get('sort') or request.form.get('return_sort', '')

    if request.method == 'POST':
        job_data = _job_form_data()
        if not job_data.get('title'):
            flash('Job title is required', 'error')
            return render_template(
                'job_form.html',
                job={**job, **job_data},
                job_statuses=JOB_WORKFLOW_STATUSES,
                status_labels=JOB_STATUS_LABELS,
                return_folder=return_folder,
                return_search=return_search,
                return_sort=return_sort,
            )

        workflow_status = job_data.pop('workflow_status', DEFAULT_JOB_STATUS)
        job_repo.update_job(job_id, job_data)
        previous_status = job.get('workflow_status', DEFAULT_JOB_STATUS)
        if workflow_status != previous_status:
            _move_jobs_with_history(
                [job_id],
                workflow_status,
                history_label=f'Move to {job_status_label(workflow_status)}',
            )
        flash('Job updated successfully', 'success')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    return render_template(
        'job_form.html',
        job=job,
        job_statuses=JOB_WORKFLOW_STATUSES,
        status_labels=JOB_STATUS_LABELS,
        return_folder=return_folder,
        return_search=return_search,
        return_sort=return_sort,
    )


@app.route('/jobs/<int:job_id>/status', methods=['POST'])
def update_job_status(job_id):
    """Move a job to another workflow folder."""
    workflow_status = request.form.get('workflow_status', '')
    return_view = request.form.get('return_view', 'manage')
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '')
    return_sort = request.form.get('return_sort', '')
    return_from_manage = bool(request.form.get('return_from_manage'))

    if not is_valid_job_status(workflow_status):
        flash('Invalid job status', 'error')
        return _redirect_after_job_status_update(
            return_view,
            job_id,
            workflow_status,
            return_folder,
            return_search,
            return_sort,
            return_from_manage,
        )

    job = job_repo.get_job(job_id)
    if not job:
        flash('Job not found', 'error')
        return _redirect_after_job_status_update(
            return_view,
            job_id,
            workflow_status,
            return_folder,
            return_search,
            return_sort,
            return_from_manage,
        )

    if _move_jobs_with_history(
        [job_id],
        workflow_status,
        history_label=f'Move to {job_status_label(workflow_status)}',
    ):
        flash(f'Job moved to {job_status_label(workflow_status)}', 'success')
    else:
        flash(f'Job is already in {job_status_label(workflow_status)}', 'info')
    return _redirect_after_job_status_update(
        return_view,
        job_id,
        workflow_status,
        return_folder,
        return_search,
        return_sort,
        return_from_manage,
    )


@app.route('/jobs/<int:job_id>/clear_cv', methods=['POST'])
def clear_job_cv(job_id):
    """Remove a job's generated CV so it can be regenerated from the base profile."""
    job = job_repo.get_job(job_id)
    return_view = request.form.get('return_view', 'list')
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '')
    return_sort = request.form.get('return_sort', '')

    if not job:
        flash('Job not found', 'error')
        return _redirect_after_clear_cv(
            return_view,
            return_folder,
            return_search,
            return_sort,
        )

    if not job.get('cv_filename') and not job.get('cover_letter_filename'):
        flash('This job has no generated CV to clear', 'info')
        return _redirect_after_clear_cv(
            return_view,
            return_folder,
            return_search,
            return_sort,
        )

    if _clear_job_cv(job, job_id):
        flash(
            'CV cleared. Use Make AI CV to regenerate from your base profile.',
            'success',
        )
    else:
        flash('Could not clear CV for this job', 'error')

    return _redirect_after_clear_cv(
        return_view,
        return_folder,
        return_search,
        return_sort,
    )


@app.route('/jobs/batch/status', methods=['POST'])
def batch_update_job_status():
    """Move multiple selected jobs to another workflow folder."""
    raw_job_ids = request.form.getlist('job_ids')
    workflow_status = request.form.get('workflow_status', '')
    return_view = request.form.get('return_view', 'manage')
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '')
    return_sort = request.form.get('return_sort', '')

    if not raw_job_ids:
        flash('No jobs selected', 'warning')
        if return_view == 'list':
            return redirect(url_for('job_list', sort=return_sort or None))
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    if not is_valid_job_status(workflow_status):
        flash('Invalid job status', 'error')
        if return_view == 'list':
            return redirect(url_for('job_list', sort=return_sort or None))
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    try:
        job_ids = [int(job_id) for job_id in raw_job_ids]
    except ValueError:
        flash('Invalid job selection', 'error')
        if return_view == 'list':
            return redirect(url_for('job_list', sort=return_sort or None))
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    updated = _move_jobs_with_history(
        job_ids,
        workflow_status,
        history_label=f'Move {len(job_ids)} job(s) to {job_status_label(workflow_status)}',
    )
    if updated:
        flash(
            f'Moved {updated} job{"s" if updated != 1 else ""} to {job_status_label(workflow_status)}',
            'success',
        )
    else:
        flash('No jobs were updated', 'warning')

    if return_view == 'list':
        return redirect(url_for('job_list', sort=return_sort or None))
    return _manage_jobs_redirect(return_folder, return_search, return_sort)


@app.route('/jobs/manage/archive-folder', methods=['POST'])
def archive_folder_jobs():
    """Move every job in the current folder to Archived in one action."""
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '').strip()
    return_sort = request.form.get('return_sort', '').strip()

    if return_folder == 'archived':
        flash('Jobs are already in Archived.', 'info')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    jobs = _jobs_for_manage_folder(return_folder, return_search)
    job_ids = [int(job['id']) for job in jobs if job.get('id')]
    if not job_ids:
        flash('No jobs to archive in this folder.', 'warning')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    if return_folder == 'all':
        folder_label = 'All Jobs'
    else:
        folder_label = job_status_label(return_folder)

    updated = _move_jobs_with_history(
        job_ids,
        'archived',
        history_label=f'Archive all in {folder_label}',
    )
    flash(
        f'Archived {updated} job{"s" if updated != 1 else ""} from {folder_label}',
        'success',
    )
    return _manage_jobs_redirect(return_folder, return_search, return_sort)


@app.route('/jobs/manage/undo-move', methods=['POST'])
@app.route('/jobs/undo-move', methods=['POST'])
def undo_job_move():
    """Undo the most recent job folder move."""
    return_view = request.form.get('return_view', 'manage')
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '').strip()
    return_sort = request.form.get('return_sort', '').strip()
    return _undo_job_moves_redirect(return_view, return_folder, return_search, return_sort)


@app.route('/jobs/manage/redo-move', methods=['POST'])
@app.route('/jobs/redo-move', methods=['POST'])
def redo_job_move():
    """Redo the most recently undone job folder move."""
    return_view = request.form.get('return_view', 'manage')
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '').strip()
    return_sort = request.form.get('return_sort', '').strip()
    return _redo_job_moves_redirect(return_view, return_folder, return_search, return_sort)


@app.route('/export/<fmt>')
def export_jobs_route(fmt):
    """Export jobs as Excel, CSV, or PDF."""
    if fmt not in ('excel', 'csv', 'pdf'):
        flash('Unsupported export format', 'error')
        return redirect(url_for('job_list'))

    search_run_id = request.args.get('search_run_id', type=int) or session.get('search_run_id')
    jobs = _get_jobs_for_view(search_run_id)

    if not jobs:
        flash('No jobs available to export', 'warning')
        return redirect(url_for('index'))

    today_date = datetime.today().strftime("%Y-%m-%d")
    mimetypes = {
        'excel': ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', f'jobs_{today_date}.xlsx'),
        'csv': ('text/csv', f'jobs_{today_date}.csv'),
        'pdf': ('application/pdf', f'jobs_{today_date}.pdf'),
    }
    mimetype, filename = mimetypes[fmt]

    if fmt == 'excel':
        filepath = os.path.join(app.config['JOBS_OUTPUT_DIR'], filename)
        export_jobs(jobs, 'excel', filepath)
        return send_file(filepath, as_attachment=True, download_name=filename)

    if fmt == 'csv':
        buffer = export_jobs(jobs, 'csv')
        return send_file(
            buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename,
        )

    buffer = export_jobs(jobs, 'pdf')
    return send_file(
        buffer,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename,
    )


@app.route('/download_excel')
def download_excel():
    """Download the Excel file with job listings."""
    return redirect(url_for('export_jobs_route', fmt='excel'))

@app.route('/make_cv/<int:job_id>')
def make_cv(job_id):
    """Queue single-job AI CV generation and show progress."""
    job = job_repo.get_job(job_id)
    profile = profile_repo.get_profile()
    return_folder = request.args.get('folder', 'all')
    return_search = request.args.get('q', '')
    return_sort = request.args.get('sort', '')
    return_from_manage = 'folder' in request.args or bool(return_search)

    if not job:
        flash('Job not found', 'error')
        if return_from_manage:
            return _manage_jobs_redirect(return_folder, return_search, return_sort)
        return redirect(url_for('job_list', sort=return_sort or None))

    if not profile_is_ready(profile):
        flash('Please complete your CV profile first', 'error')
        return redirect(url_for('user_profile'))

    task_id = _enqueue_ai_task(
        'single_cv',
        {
            'job_id': job_id,
            'return_folder': return_folder,
            'return_search': return_search,
            'return_from_manage': return_from_manage,
            'return_sort': return_sort,
        },
        job_id=job_id,
    )
    session['cv_generation_active'] = task_id
    flash(
        'CV generation queued — track progress here or from the AI queue '
        '(run: job-apply-ai ai-worker)',
        'success',
    )
    return redirect(url_for('ai_cv_task_progress', task_id=task_id))


@app.route('/ai_tasks/cv/<task_id>')
def ai_cv_task_progress(task_id):
    """Show progress for a queued single or batch CV generation task."""
    task = _resolve_task(task_id)
    if not task or task.get('task_type') not in ('single_cv', 'batch_cv'):
        flash('CV generation task not found', 'error')
        return redirect(url_for('job_list'))

    meta = task.get('meta') or {}
    job_count = meta.get('total_jobs', 1)
    job = None
    if task.get('task_type') == 'single_cv':
        job_id = task.get('job_id') or meta.get('job_id')
        if job_id:
            job = job_repo.get_job(job_id)
            job_count = 1

    back_url, back_label = _ai_cv_back_context(task)
    return render_template(
        'batch_cv_progress.html',
        task_id=task_id,
        job=job,
        job_count=job_count,
        status_url=url_for('cv_task_status', task_id=task_id),
        complete_url=_ai_cv_complete_url(task),
        control_url=url_for('control_background_task', task_id=task_id),
        back_url=back_url,
        back_label=back_label,
        ai_queue_url=url_for('ai_queue_list'),
    )


@app.route('/api/cv_generation/release', methods=['POST', 'GET'])
def release_cv_generation_lock():
    """Dismiss the active CV task banner (does not stop the queue worker)."""
    session.pop('cv_generation_active', None)
    if request.method == 'GET':
        flash('Active CV task banner dismissed.', 'info')
        return redirect(request.referrer or url_for('manage_jobs'))
    return jsonify({'ok': True})


@app.route('/api/ats_friendly/release', methods=['POST', 'GET'])
def release_ats_friendly_lock():
    """Clear UI lock after failed, abandoned, or stale ATS analysis."""
    session.pop('ats_friendly_active', None)
    if request.method == 'GET':
        flash('ATS analysis lock cleared.', 'info')
        return redirect(request.referrer or url_for('manage_jobs'))
    return jsonify({'ok': True})


@app.route('/api/cv_tasks/<task_id>/status')
def cv_task_status(task_id):
    """Poll background CV generation progress."""
    task = _resolve_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


@app.route('/api/cv_tasks/<task_id>/control', methods=['POST'])
def control_background_task(task_id):
    """Pause, resume, or stop a controllable background task."""
    queue_job = batch_queue_repo.get_job_by_task_id(task_id)
    if queue_job:
        payload = request.get_json(silent=True) or {}
        action = (payload.get('action') or request.form.get('action', '')).strip().lower()
        handlers = {
            'pause': lambda: batch_queue_repo.pause_job(queue_job['id']),
            'resume': lambda: batch_queue_repo.resume_job(queue_job['id']),
            'stop': lambda: batch_queue_repo.request_stop(queue_job['id']),
        }
        handler = handlers.get(action)
        if not handler:
            return jsonify({'error': 'Invalid action'}), 400
        if not handler():
            return jsonify({'error': f'Cannot {action} task in its current state'}), 409
        updated = to_task_snapshot(batch_queue_repo.get_job_by_task_id(task_id))
        return jsonify(updated)

    ai_job = ai_task_queue_repo.get_job_by_task_id(task_id)
    if ai_job and ai_job['task_type'] in CONTROLLABLE_AI_TASK_TYPES:
        payload = request.get_json(silent=True) or {}
        action = (payload.get('action') or request.form.get('action', '')).strip().lower()
        handlers = {
            'pause': lambda: ai_task_queue_repo.pause_job(ai_job['id']),
            'resume': lambda: ai_task_queue_repo.resume_job(ai_job['id']),
            'stop': lambda: ai_task_queue_repo.request_stop(ai_job['id']),
        }
        handler = handlers.get(action)
        if not handler:
            return jsonify({'error': 'Invalid action'}), 400
        if not handler():
            return jsonify({'error': f'Cannot {action} task in its current state'}), 409
        updated = to_ai_task_snapshot(ai_task_queue_repo.get_job_by_task_id(task_id))
        return jsonify(updated)

    urgent_job = urgent_task_queue_repo.get_job_by_task_id(task_id)
    if urgent_job and urgent_job['task_type'] in CONTROLLABLE_URGENT_TASK_TYPES:
        payload = request.get_json(silent=True) or {}
        action = (payload.get('action') or request.form.get('action', '')).strip().lower()
        handlers = {
            'pause': lambda: urgent_task_queue_repo.pause_job(urgent_job['id']),
            'resume': lambda: urgent_task_queue_repo.resume_job(urgent_job['id']),
            'stop': lambda: urgent_task_queue_repo.request_stop(urgent_job['id']),
        }
        handler = handlers.get(action)
        if not handler:
            return jsonify({'error': 'Invalid action'}), 400
        if not handler():
            return jsonify({'error': f'Cannot {action} task in its current state'}), 409
        updated = to_urgent_task_snapshot(urgent_task_queue_repo.get_job_by_task_id(task_id))
        return jsonify(updated)

    task = get_task(task_id)
    if not task or task.get('task_type') not in CONTROLLABLE_TASK_TYPES:
        return jsonify({'error': 'Task not found'}), 404

    payload = request.get_json(silent=True) or {}
    action = (payload.get('action') or request.form.get('action', '')).strip().lower()
    handlers = {
        'pause': pause_task,
        'resume': resume_task,
        'stop': request_task_stop,
    }
    handler = handlers.get(action)
    if not handler:
        return jsonify({'error': 'Invalid action'}), 400
    if not handler(task_id):
        return jsonify({'error': f'Cannot {action} task in its current state'}), 409

    updated = get_task(task_id)
    return jsonify(updated)


@app.route('/make_cv/complete/<task_id>')
def make_cv_complete(task_id):
    """Render success page after background CV generation completes."""
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('CV generation result not found', 'error')
        return redirect(url_for('job_list'))

    result = task['result']
    session.pop('cv_generation_active', None)
    session['current_cv'] = result.get('output_path')
    session['current_cv_filename'] = result.get('cv_filename')

    flash('Professional CV and cover letter generated successfully with RAG + AI', 'success')
    return_from_manage = result.get('return_from_manage', False)
    return_folder = result.get('return_folder', 'all')
    return_search = result.get('return_search', '')
    return_sort = result.get('return_sort', '')

    cv_filename = result.get('cv_filename', '')
    if cv_filename:
        store = normalize_store(_load_job_cv_store(cv_filename) or {})
        if store.get('tailored_content'):
            try:
                _sync_job_cv_docx_from_preview(
                    cv_filename,
                    store,
                    profile_repo.get_profile(),
                )
            except Exception as exc:
                logger.warning('Failed to rebuild CV docx after generation for job preview: %s', exc)

    context = _cv_preview_context(
        result.get('job') or {},
        tailored_content=result.get('tailored_content', {}),
        matched_categories=result.get('matched_categories', {}),
        analysis=result.get('analysis', {}),
        generation_meta=result.get('generation_meta', {}),
        rag_chunk_count=result.get('rag_chunk_count', 0),
        show_success_banner=True,
        return_folder=return_folder,
        return_search=return_search,
        return_sort=return_sort,
        return_from_manage=return_from_manage,
    )
    return render_template('cv_success.html', **context)


@app.route('/jobs/<int:job_id>/cv/preview')
def preview_job_cv(job_id):
    """Preview and chat-edit a generated CV for a job."""
    job = job_repo.get_job(job_id)
    return_folder = request.args.get('folder', 'all')
    return_search = request.args.get('q', '')
    return_sort = request.args.get('sort', '')
    return_from_manage = 'folder' in request.args or bool(return_search)

    if not job:
        flash('Job not found', 'error')
        if return_from_manage:
            return _manage_jobs_redirect(return_folder, return_search, return_sort)
        return redirect(url_for('job_list', sort=return_sort or None))

    cv_filename = job.get('cv_filename', '')
    cv_path = os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)
    if not cv_filename or not os.path.exists(cv_path):
        flash('No CV has been generated for this job yet', 'error')
        if return_from_manage:
            return _manage_jobs_redirect(return_folder, return_search, return_sort)
        return redirect(url_for('job_list', sort=return_sort or None))

    store = normalize_store(_load_job_cv_store(cv_filename) or {})
    if not store or not store.get('tailored_content'):
        flash('CV preview data not found. Regenerate the CV to enable preview and chat editing.', 'warning')
        if return_from_manage:
            return _manage_jobs_redirect(return_folder, return_search, return_sort)
        return redirect(url_for('job_list', sort=return_sort or None))

    profile = profile_repo.get_profile()
    try:
        store = _sync_job_cv_docx_from_preview(cv_filename, store, profile, persist_store=True)
    except Exception as exc:
        logger.error('Failed to rebuild CV docx for job %s preview: %s', job_id, exc)
        flash('Could not rebuild the CV document for preview.', 'warning')

    context = _cv_preview_context(
        job,
        show_success_banner=False,
        return_folder=return_folder,
        return_search=return_search,
        return_sort=return_sort,
        return_from_manage=return_from_manage,
    )
    return render_template('cv_success.html', **context)


@app.route('/jobs/<int:job_id>/ats-friendly')
def ats_friendly_progress(job_id):
    """Show progress while ATS analysis runs, or existing results when available."""
    job = job_repo.get_job(job_id)
    return_folder = request.args.get('folder', 'all')
    return_search = request.args.get('q', '')
    return_sort = request.args.get('sort', '')
    return_from_manage = 'folder' in request.args or bool(return_search)
    force = request.args.get('force') == '1'

    if not job:
        flash('Job not found', 'error')
        if return_from_manage:
            return _manage_jobs_redirect(return_folder, return_search, return_sort)
        return redirect(url_for('job_list', sort=return_sort or None))

    cv_filename = job.get('cv_filename', '')
    cv_path = os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)
    if not cv_filename or not os.path.exists(cv_path):
        flash('Generate a CV for this job before running ATS analysis.', 'error')
        if return_from_manage:
            return _manage_jobs_redirect(return_folder, return_search, return_sort)
        return redirect(url_for('job_list', sort=return_sort or None))

    if force:
        stale_task_id = session.pop('ats_friendly_active', None)
        if stale_task_id:
            stale_task = get_task(stale_task_id)
            if stale_task and stale_task.get('status') in ('pending', 'running'):
                fail_task(stale_task_id, 'Superseded by a new ATS analysis.')

    if not force:
        existing = get_job_ats_analysis(job)
        if existing:
            context = _ats_friendly_results_context(
                job,
                existing,
                return_folder=return_folder,
                return_search=return_search,
                return_sort=return_sort,
                return_from_manage=return_from_manage,
            )
            return render_template('ats_friendly_results.html', **context)

    return render_template(
        'ats_friendly_progress.html',
        job=job,
        start_url=url_for(
            'start_ats_friendly',
            job_id=job_id,
            folder=return_folder,
            q=return_search or None,
            sort=return_sort or None,
        ),
        status_url_template=url_for('cv_task_status', task_id='TASK_ID'),
        complete_url_template=url_for('ats_friendly_complete', job_id=job_id, task_id='TASK_ID'),
        back_url=_cv_generation_back_url(
            return_from_manage,
            return_folder,
            return_search,
            return_sort,
        ),
    )


@app.route('/jobs/<int:job_id>/ats-friendly/start', methods=['POST'])
def start_ats_friendly(job_id):
    """Start background ATS-friendly analysis for one job."""
    job = job_repo.get_job(job_id)
    profile = profile_repo.get_profile()
    return_folder = request.args.get('folder', 'all')
    return_search = request.args.get('q', '')
    return_sort = request.args.get('sort', '')
    return_from_manage = 'folder' in request.args or bool(return_search)

    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not profile_is_ready(profile):
        return jsonify({'error': 'Please complete your CV profile first'}), 400

    cv_filename = job.get('cv_filename', '')
    if not cv_filename or not os.path.exists(os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)):
        return jsonify({'error': 'Generate a CV for this job first'}), 400

    if _sync_session_background_task('ats_friendly_active'):
        return jsonify({'error': 'Another ATS analysis is already in progress'}), 409

    task_id = _enqueue_ai_task(
        'ats_friendly',
        {
            'job_id': job_id,
            'cv_filename': cv_filename,
            'return_folder': return_folder,
            'return_search': return_search,
            'return_from_manage': return_from_manage,
            'return_sort': return_sort,
        },
        job_id=job_id,
    )
    session['ats_friendly_active'] = task_id
    return jsonify({'task_id': task_id})


@app.route('/jobs/<int:job_id>/ats-friendly/complete/<task_id>')
def ats_friendly_complete(job_id, task_id):
    """Render ATS analysis results after background processing completes."""
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('ATS analysis result not found', 'error')
        return redirect(url_for('ats_friendly_progress', job_id=job_id))

    result = task['result']
    session.pop('ats_friendly_active', None)
    job = result.get('job') or job_repo.get_job(job_id) or {}
    context = _ats_friendly_results_context(
        job,
        result.get('ats_analysis', {}),
        return_folder=result.get('return_folder', 'all'),
        return_search=result.get('return_search', ''),
        return_sort=result.get('return_sort', ''),
        return_from_manage=result.get('return_from_manage', False),
    )
    return render_template('ats_friendly_results.html', **context)


@app.route('/api/jobs/<int:job_id>/ats-friendly/suggestions', methods=['POST'])
def ats_suggestion_action(job_id):
    """Apply, deny, regenerate, or bulk-apply ATS improvement suggestions."""
    job = job_repo.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    cv_filename = job.get('cv_filename', '')
    cv_path = os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)
    if not cv_filename or not os.path.exists(cv_path):
        return jsonify({'error': 'No CV file found for this job'}), 404

    payload = request.get_json(silent=True) or {}
    action = str(payload.get('action', '')).strip().lower()
    suggestion_id = str(payload.get('suggestion_id', '')).strip()
    if action not in {'apply', 'deny', 'reapply', 'apply_all'}:
        return jsonify({'error': 'Invalid action'}), 400

    store = normalize_store(_load_job_cv_store(cv_filename) or {})
    analysis = normalize_ats_analysis(store.get('ats_analysis', {}))
    if not analysis.get('suggestions'):
        return jsonify({'error': 'No ATS suggestions found. Run ATS analysis first.'}), 404

    profile = profile_repo.get_profile()
    profile_name = profile.get('full_name', '')
    current_content = resolve_effective_tailored_content(
        store.get('tailored_content', {}),
        profile_name,
        stored_lines=store.get('cv_preview_lines'),
        customized=bool(store.get('cv_preview_customized')),
    )

    if action == 'apply_all':
        if not current_content:
            return jsonify({'error': 'CV content not available for editing'}), 404
        to_apply = pending_suggestions(analysis)
        if not to_apply:
            return jsonify({'error': 'No pending suggestions to apply'}), 400
        suggestion_ids = [item['id'] for item in to_apply]
        try:
            analyzer = ATSFriendlyAnalyzer()
            with dev_agent("ATSFriendlyAnalyzer", job_id=job_id):
                updated_content = analyzer.apply_all_suggestions(
                    job=job,
                    cv_content=current_content,
                    profile=profile,
                    suggestions=to_apply,
                )
            editor = CVChatEditor()
            editor.rebuild_document(cv_path, updated_content, profile)
            matched_categories = CVChatEditor.content_to_matched_categories(updated_content)
            job['matched_categories'] = matched_categories
            job_repo.update_job(job_id, job)

            analysis = update_suggestions_status(analysis, suggestion_ids, status='applied')
            store['ats_analysis'] = analysis
            _clear_cv_preview_customization(store, updated_content)
            _save_job_cv_content(cv_filename, updated_content, store=store)

            session['current_cv'] = cv_path
            session['current_cv_filename'] = cv_filename

            return jsonify({
                'ok': True,
                'ats_analysis': analysis,
                'content': updated_content,
                'cv_preview_lines': cv_content_to_preview_lines(
                    updated_content,
                    profile_name,
                ),
                'matched_categories': matched_categories,
            })
        except Exception as exc:
            logger.error('ATS apply_all failed for job %s: %s', job_id, exc)
            try:
                analysis = update_suggestions_status(
                    analysis,
                    suggestion_ids,
                    status='failed',
                    error=str(exc),
                )
                store['ats_analysis'] = analysis
                _save_job_cv_content(cv_filename, current_content, store=store)
            except KeyError:
                pass
            return jsonify({'error': str(exc), 'ats_analysis': analysis}), 500

    if not suggestion_id:
        return jsonify({'error': 'suggestion_id is required'}), 400
    if action in {'apply', 'reapply'} and not current_content:
        return jsonify({'error': 'CV content not available for editing'}), 404

    try:
        if action == 'deny':
            analysis = update_suggestion_status(analysis, suggestion_id, status='denied')
            store['ats_analysis'] = analysis
            _save_job_cv_content(cv_filename, current_content, store=store)
            return jsonify({'ok': True, 'ats_analysis': analysis})

        suggestion = get_suggestion(analysis, suggestion_id)

        if action == 'reapply':
            analyzer = ATSFriendlyAnalyzer()
            with dev_agent("ATSFriendlyAnalyzer", job_id=job_id):
                refreshed = analyzer.reapply_suggestion(
                    job=job,
                    cv_content=current_content,
                    profile=profile,
                    suggestion=suggestion,
                )
            analysis = replace_suggestion(analysis, suggestion_id, refreshed)
            store['ats_analysis'] = analysis
            _save_job_cv_content(cv_filename, current_content, store=store)
            return jsonify({'ok': True, 'ats_analysis': analysis, 'suggestion': refreshed})

        analyzer = ATSFriendlyAnalyzer()
        with dev_agent("ATSFriendlyAnalyzer", job_id=job_id):
            updated_content = analyzer.apply_suggestion(
                job=job,
                cv_content=current_content,
                profile=profile,
                suggestion=suggestion,
            )
        editor = CVChatEditor()
        editor.rebuild_document(cv_path, updated_content, profile)
        matched_categories = CVChatEditor.content_to_matched_categories(updated_content)
        job['matched_categories'] = matched_categories
        job_repo.update_job(job_id, job)

        analysis = update_suggestion_status(analysis, suggestion_id, status='applied')
        store['ats_analysis'] = analysis
        _clear_cv_preview_customization(store, updated_content)
        _save_job_cv_content(cv_filename, updated_content, store=store)

        session['current_cv'] = cv_path
        session['current_cv_filename'] = cv_filename

        return jsonify({
            'ok': True,
            'ats_analysis': analysis,
            'content': updated_content,
            'cv_preview_lines': cv_content_to_preview_lines(
                updated_content,
                profile_name,
            ),
            'matched_categories': matched_categories,
        })
    except Exception as exc:
        logger.error('ATS suggestion %s failed for job %s: %s', action, job_id, exc)
        try:
            analysis = update_suggestion_status(
                analysis,
                suggestion_id,
                status='failed',
                error=str(exc),
            )
            store['ats_analysis'] = analysis
            _save_job_cv_content(cv_filename, current_content, store=store)
        except KeyError:
            pass
        return jsonify({'error': str(exc), 'ats_analysis': analysis}), 500


@app.route('/api/jobs/<int:job_id>/documents/chat', methods=['POST'])
@app.route('/api/jobs/<int:job_id>/cv/chat', methods=['POST'])
def document_chat(job_id):
    """Apply a chat instruction to refine a generated CV or cover letter."""
    job = job_repo.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    cv_filename = job.get('cv_filename', '')
    cv_path = os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)
    if not cv_filename or not os.path.exists(cv_path):
        return jsonify({'error': 'No CV file found for this job'}), 404

    payload = request.get_json(silent=True) or {}
    user_message = str(payload.get('message', '')).strip()
    document_type = str(payload.get('document', 'cv')).strip().lower()
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    store = normalize_store(_load_job_cv_store(cv_filename) or {})
    profile = profile_repo.get_profile()

    try:
        endpoint = f"POST /api/jobs/{job_id}/documents/chat"
        if document_type == 'cover_letter':
            cl_filename = job.get('cover_letter_filename', '')
            cl_path = os.path.join(app.config['CV_OUTPUT_DIR'], cl_filename)
            if not cl_filename or not store.get('cover_letter'):
                return jsonify({'error': 'Cover letter not available for editing'}), 404

            chat_history = get_active_chat_messages(store, 'cover_letter')
            editor = CoverLetterChatEditor()
            with dev_llm_context(
                endpoint=endpoint,
                operation="cover_letter_chat",
                chat_history=chat_history,
                context={"user_message": user_message, "document_type": document_type},
            ), dev_agent("CoverLetterChatEditor", job_id=job_id):
                result = editor.modify(
                    current_content=store['cover_letter'],
                    user_message=user_message,
                    job=job,
                    profile=profile,
                    tailored_cv_content=store.get('tailored_content', {}),
                    chat_history=chat_history,
                )
            updated_content = result['content']
            reply = result['reply']
            editor.rebuild_document(cl_path, updated_content)

            append_active_chat_messages(
                store,
                'cover_letter',
                [
                    {'role': 'user', 'content': user_message},
                    {'role': 'assistant', 'content': reply},
                ],
            )
            _save_job_cv_content(
                cv_filename,
                store.get('tailored_content', {}),
                cover_letter=updated_content,
                store=store,
            )

            return jsonify(_document_chat_payload(
                store,
                'cover_letter',
                extra={
                    'reply': reply,
                    'cover_letter': updated_content,
                },
            ))

        if not store.get('tailored_content'):
            return jsonify({'error': 'CV content not available for editing'}), 404

        chat_history = get_active_chat_messages(store, 'cv')
        current_content = store['tailored_content']
        profile_name = profile.get('full_name', '')
        current_preview_lines = resolve_cv_preview_lines(
            current_content,
            profile_name,
            stored_lines=store.get('cv_preview_lines'),
            customized=bool(store.get('cv_preview_customized')),
        )

        editor = CVChatEditor()
        with dev_llm_context(
            endpoint=endpoint,
            operation="cv_chat",
            chat_history=chat_history,
            context={"user_message": user_message, "document_type": document_type},
        ), dev_agent("CVChatEditor", job_id=job_id):
            result = editor.modify(
                current_content=current_content,
                user_message=user_message,
                job=job,
                profile=profile,
                chat_history=chat_history,
                preview_lines=current_preview_lines,
                preview_customized=bool(store.get('cv_preview_customized')),
            )
        updated_content = result['content']
        reply = result['reply']

        editor.rebuild_document(cv_path, updated_content, profile)
        matched_categories = CVChatEditor.content_to_matched_categories(updated_content)
        job['matched_categories'] = matched_categories
        job_repo.update_job(job_id, job)

        append_active_chat_messages(
            store,
            'cv',
            [
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': reply},
            ],
        )
        store['cv_preview_lines'] = []
        store['cv_preview_customized'] = False
        _save_job_cv_content(
            cv_filename,
            updated_content,
            cover_letter=store.get('cover_letter', {}),
            store=store,
        )

        session['current_cv'] = cv_path
        session['current_cv_filename'] = cv_filename

        return jsonify(_document_chat_payload(
            store,
            'cv',
            extra={
                'reply': reply,
                'content': updated_content,
                'cv_preview_lines': cv_content_to_preview_lines(
                    updated_content,
                    profile.get('full_name', ''),
                ),
                'matched_categories': matched_categories,
            },
        ))
    except Exception as exc:
        logger.error('Document chat edit failed for job %s: %s', job_id, exc)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/jobs/<int:job_id>/cv/ask', methods=['POST'])
def cv_ask_chat(job_id):
    """Answer questions about the job and tailored CV without modifying documents."""
    job = job_repo.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    cv_filename = job.get('cv_filename', '')
    if not cv_filename:
        return jsonify({'error': 'No CV file found for this job'}), 404

    payload = request.get_json(silent=True) or {}
    user_message = str(payload.get('message', '')).strip()
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    store = normalize_store(_load_job_cv_store(cv_filename) or {})
    if not store.get('tailored_content'):
        return jsonify({'error': 'CV content not available'}), 404

    profile = profile_repo.get_profile()
    chat_history = get_active_chat_messages(store, 'cv_ask')
    current_content = store['tailored_content']
    profile_name = profile.get('full_name', '')
    current_preview_lines = resolve_cv_preview_lines(
        current_content,
        profile_name,
        stored_lines=store.get('cv_preview_lines'),
        customized=bool(store.get('cv_preview_customized')),
    )

    try:
        endpoint = f"POST /api/jobs/{job_id}/cv/ask"
        assistant = CVAskAssistant()
        with dev_llm_context(
            endpoint=endpoint,
            operation="cv_ask",
            chat_history=chat_history,
            context={"user_message": user_message, "document_type": "cv_ask"},
        ), dev_agent("CVAskAssistant", job_id=job_id):
            reply = assistant.ask(
                current_content=current_content,
                user_message=user_message,
                job=job,
                profile=profile,
                chat_history=chat_history,
                preview_lines=current_preview_lines,
                preview_customized=bool(store.get('cv_preview_customized')),
            )

        append_active_chat_messages(
            store,
            'cv_ask',
            [
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': reply},
            ],
        )
        _save_job_cv_content(
            cv_filename,
            current_content,
            cover_letter=store.get('cover_letter', {}),
            store=store,
        )

        return jsonify(_document_chat_payload(
            store,
            'cv_ask',
            extra={'reply': reply, 'document': 'cv_ask'},
        ))
    except Exception as exc:
        logger.error('CV ask chat failed for job %s: %s', job_id, exc)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/jobs/<int:job_id>/cv/preview-lines', methods=['POST'])
@app.route('/api/jobs/<int:job_id>/cv/preview-lines/reorder', methods=['POST'])
def cv_preview_lines_update(job_id):
    """Persist user-edited numbered CV preview lines."""
    job = job_repo.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    cv_filename = job.get('cv_filename', '')
    if not cv_filename:
        return jsonify({'error': 'No CV file found for this job'}), 404

    payload = request.get_json(silent=True) or {}
    lines = payload.get('lines')
    if not isinstance(lines, list):
        return jsonify({'error': 'lines must be an array'}), 400

    store = normalize_store(_load_job_cv_store(cv_filename) or {})
    content = store.get('tailored_content', {})
    if not content:
        return jsonify({'error': 'CV content not available for editing'}), 404

    normalized = normalize_preview_lines(lines)
    store['cv_preview_lines'] = normalized
    store['cv_preview_customized'] = True

    profile = profile_repo.get_profile()
    try:
        store = _sync_job_cv_docx_from_preview(
            cv_filename,
            store,
            profile,
            persist_store=True,
        )
        normalized = store.get('cv_preview_lines', normalized)
    except Exception as exc:
        logger.error('Failed to rebuild CV docx after preview edit for job %s: %s', job_id, exc)
        _save_job_cv_content(cv_filename, content, store=store)
        return jsonify({'error': f'Could not rebuild CV document: {exc}'}), 500

    return jsonify({
        'ok': True,
        'cv_preview_lines': normalized,
        'cv_preview_customized': True,
        'content': store.get('tailored_content', content),
    })


@app.route('/api/jobs/<int:job_id>/documents/chat/sessions', methods=['POST'])
def document_chat_sessions(job_id):
    """Create or switch persisted chat sessions for CV or cover letter editing."""
    job = job_repo.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    cv_filename = job.get('cv_filename', '')
    if not cv_filename:
        return jsonify({'error': 'No CV file found for this job'}), 404

    payload = request.get_json(silent=True) or {}
    document_type = str(payload.get('document', 'cv')).strip().lower()
    action = str(payload.get('action', 'new')).strip().lower()
    session_id = str(payload.get('session_id', '')).strip()

    store = normalize_store(_load_job_cv_store(cv_filename) or {})
    document = _resolve_chat_document(document_type)

    if action == 'switch':
        if not session_id:
            return jsonify({'error': 'session_id is required to switch sessions'}), 400
        if not set_active_chat_session(store, document, session_id):
            return jsonify({'error': 'Chat session not found'}), 404
    else:
        start_chat_session(store, document)

    _save_job_cv_content(
        cv_filename,
        store.get('tailored_content', {}),
        cover_letter=store.get('cover_letter', {}),
        store=store,
    )

    return jsonify(_document_chat_payload(store, document))


@app.route('/api/jobs/<int:job_id>/cover-letter/generate', methods=['POST'])
def generate_job_cover_letter(job_id):
    """Generate or regenerate a cover letter from the job CV and description."""
    job = job_repo.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    cv_filename = job.get('cv_filename', '')
    if not cv_filename:
        return jsonify({'error': 'Generate a CV first before creating a cover letter'}), 400

    store = _load_job_cv_store(cv_filename)
    tailored_content = (store or {}).get('tailored_content', {})
    if not tailored_content:
        return jsonify({'error': 'CV content not found. Regenerate the CV first.'}), 404

    profile = profile_repo.get_profile()
    try:
        cl_filename, _, cl_content = _generate_and_save_cover_letter(
            job, job_id, profile, tailored_content, cv_filename, reset_cover_letter_chat=True
        )
        return jsonify({
            'cover_letter_filename': cl_filename,
            'cover_letter': cl_content,
            'cover_letter_download_url': url_for('download_job_cover_letter', job_id=job_id),
            'cover_letter_download_pdf_url': url_for('download_job_cover_letter_pdf', job_id=job_id),
        })
    except Exception as exc:
        logger.error('Cover letter generation failed for job %s: %s', job_id, exc)
        return jsonify({'error': str(exc)}), 500


@app.route('/jobs/<int:job_id>/send-application', methods=['POST'])
@app.route('/api/jobs/<int:job_id>/send-application', methods=['POST'])
def send_job_application(job_id):
    """Email the generated CV and cover letter to the job contact address."""
    account_id = request.form.get('smtp_account_id') or (request.get_json(silent=True) or {}).get('smtp_account_id')
    payload, status = _send_application_for_job(job_id, account_id=account_id or None)
    wants_json = (
        request.path.startswith('/api/')
        or request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or request.accept_mimetypes.best == 'application/json'
    )
    if wants_json:
        return jsonify(payload), status

    if payload.get('ok'):
        flash(payload.get('message', 'Application email sent'), 'success')
    else:
        flash(payload.get('error', 'Failed to send application email'), 'error')

    return_folder = request.form.get('return_folder', request.args.get('folder', 'all'))
    return_search = request.form.get('return_search', request.args.get('q', ''))
    if request.form.get('return_from_manage') or request.args.get('folder') or return_search:
        return _manage_jobs_redirect(return_folder, return_search)
    if request.referrer and 'job_list' in request.referrer:
        return redirect(url_for('job_list'))
    return redirect(url_for('preview_job_cv', job_id=job_id, folder=return_folder, q=return_search or None))


@app.route('/download_cv')
def download_cv():
    """Download the current CV."""
    cv_path = session.get('current_cv')
    
    if not cv_path or not os.path.exists(cv_path):
        flash('CV file not found', 'error')
        return redirect(url_for('job_list'))
    
    return send_file(cv_path, as_attachment=True)


@app.route('/jobs/<int:job_id>/cv')
def download_job_cv(job_id):
    """Download the AI-generated CV stored for a job."""
    job = job_repo.get_job(job_id)
    cv_filename = (job or {}).get('cv_filename', '')
    if not job or not cv_filename:
        flash('No CV has been generated for this job yet', 'error')
        return redirect(url_for('manage_jobs'))

    cv_path = os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)
    if not os.path.exists(cv_path):
        flash('CV file not found on disk', 'error')
        return redirect(url_for('manage_jobs'))

    return send_file(cv_path, as_attachment=True, download_name=cv_filename)


@app.route('/jobs/<int:job_id>/cv/pdf')
def download_job_cv_pdf(job_id):
    """Download the AI-generated CV as PDF."""
    job = job_repo.get_job(job_id)
    cv_filename = (job or {}).get('cv_filename', '')
    if not job or not cv_filename:
        flash('No CV has been generated for this job yet', 'error')
        return redirect(url_for('manage_jobs'))

    cv_path = os.path.join(app.config['CV_OUTPUT_DIR'], cv_filename)
    pdf_path = pdf_path_for_docx(cv_path)
    if not os.path.exists(cv_path):
        flash('CV file not found on disk', 'error')
        return redirect(url_for('manage_jobs'))

    if not os.path.exists(pdf_path):
        profile = profile_repo.get_profile()
        store = normalize_store(_load_job_cv_store(cv_filename) or {})
        content = store.get('tailored_content') or {}
        preview_lines = resolve_cv_preview_lines(
            content,
            profile.get('full_name', ''),
            stored_lines=store.get('cv_preview_lines'),
            customized=bool(store.get('cv_preview_customized')),
        )
        if preview_lines:
            build_cv_pdf(cv_path, preview_lines, profile, content)
        else:
            flash('Could not build PDF export for this CV', 'error')
            return redirect(url_for('manage_jobs'))

    pdf_filename = os.path.splitext(cv_filename)[0] + '.pdf'
    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename, mimetype='application/pdf')


@app.route('/jobs/<int:job_id>/cover-letter')
def download_job_cover_letter(job_id):
    """Download the AI-generated cover letter stored for a job."""
    job = job_repo.get_job(job_id)
    cl_filename = (job or {}).get('cover_letter_filename', '')
    if not job or not cl_filename:
        flash('No cover letter has been generated for this job yet', 'error')
        return redirect(url_for('manage_jobs'))

    cl_path = os.path.join(app.config['CV_OUTPUT_DIR'], cl_filename)
    if not os.path.exists(cl_path):
        flash('Cover letter file not found on disk', 'error')
        return redirect(url_for('manage_jobs'))

    return send_file(cl_path, as_attachment=True, download_name=cl_filename)


@app.route('/jobs/<int:job_id>/cover-letter/pdf')
def download_job_cover_letter_pdf(job_id):
    """Download the AI-generated cover letter as PDF."""
    job = job_repo.get_job(job_id)
    cl_filename = (job or {}).get('cover_letter_filename', '')
    if not job or not cl_filename:
        flash('No cover letter has been generated for this job yet', 'error')
        return redirect(url_for('manage_jobs'))

    cl_path = os.path.join(app.config['CV_OUTPUT_DIR'], cl_filename)
    pdf_path = pdf_path_for_docx(cl_path)
    if not os.path.exists(cl_path):
        flash('Cover letter file not found on disk', 'error')
        return redirect(url_for('manage_jobs'))

    if not os.path.exists(pdf_path):
        cv_filename = job.get('cv_filename', '')
        store = normalize_store(_load_job_cv_store(cv_filename) or {}) if cv_filename else {}
        cover_letter = store.get('cover_letter') or {}
        if not cover_letter.get('body_paragraphs'):
            flash('Could not build PDF export for this cover letter', 'error')
            return redirect(url_for('manage_jobs'))
        build_cover_letter_pdf(cl_path, cover_letter)

    pdf_filename = os.path.splitext(cl_filename)[0] + '.pdf'
    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename, mimetype='application/pdf')


@app.route('/make_all_cvs')
def make_all_cvs():
    """Queue batch AI CV generation for the current job list view."""
    search_run_id = session.get('search_run_id')
    processed_jobs = _get_jobs_for_view(search_run_id)
    profile = profile_repo.get_profile()

    if not processed_jobs:
        flash('No jobs found. Please search for jobs first.', 'error')
        return redirect(url_for('index'))

    if not profile_is_ready(profile):
        flash('Please complete your CV profile first', 'error')
        return redirect(url_for('user_profile'))

    job_ids = [job['id'] for job in processed_jobs if job.get('id')]
    task_id = _enqueue_ai_task(
        'batch_cv',
        {
            'job_ids': job_ids,
            'meta': {'total_jobs': len(processed_jobs)},
        },
    )
    session['cv_generation_active'] = task_id
    session['batch_cv_back_url'] = url_for('job_list')
    session['batch_cv_back_label'] = 'Back to Job List'
    flash(
        f'Queued CV generation for {len(processed_jobs)} job(s). '
        'Run: job-apply-ai ai-worker',
        'success',
    )
    return redirect(url_for('ai_cv_task_progress', task_id=task_id))


@app.route('/make_all_cvs/complete/<task_id>')
def make_all_cvs_complete(task_id):
    """Render batch success page after background generation completes."""
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('Batch CV generation result not found', 'error')
        return redirect(url_for('job_list'))

    result = task['result']
    session.pop('cv_generation_active', None)
    session['generated_cvs'] = result.get('generated_cvs', [])
    session['successful_jobs'] = result.get('successful_jobs', [])
    session['failed_jobs'] = result.get('failed_jobs', [])

    flash(f"Successfully generated {result.get('generated_count', 0)} tailored CVs with RAG + AI", 'success')
    if result.get('failed_jobs'):
        flash(f"Failed to generate {len(result['failed_jobs'])} CVs", 'warning')

    return render_template(
        'all_cvs_success.html',
        successful_jobs=result.get('successful_jobs', []),
        failed_jobs=result.get('failed_jobs', []),
        back_url=url_for('job_list'),
    )


@app.route('/jobs/batch/make_cvs', methods=['POST'])
def batch_make_cvs():
    """Start batch CV generation for selected jobs."""
    raw_job_ids = request.form.getlist('job_ids')
    return_view = request.form.get('return_view', 'list')
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '')
    return_sort = request.form.get('return_sort', '')

    job_ids = _parse_bulk_job_ids(raw_job_ids)
    if not job_ids:
        flash('No jobs selected', 'warning')
        if return_view == 'list':
            return redirect(url_for('job_list', sort=return_sort or None))
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    jobs = _get_jobs_by_ids(job_ids)
    if not jobs:
        flash('Selected jobs were not found', 'error')
        if return_view == 'list':
            return redirect(url_for('job_list', sort=return_sort or None))
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    profile = profile_repo.get_profile()
    if not profile_is_ready(profile):
        flash('Please complete your CV profile first', 'error')
        return redirect(url_for('user_profile'))

    session['batch_cv_back_url'] = _batch_cv_back_url(
        return_view, return_folder, return_search, return_sort
    )
    session['batch_cv_back_label'] = (
        'Back to Manage Jobs' if return_view == 'manage' else 'Back to Job List'
    )

    task_id = _enqueue_ai_task(
        'batch_cv',
        {
            'job_ids': job_ids,
            'meta': {'total_jobs': len(jobs), 'selected': True},
        },
    )
    session['cv_generation_active'] = task_id
    flash(
        f'Queued CV generation for {len(jobs)} selected job(s). '
        'Run: job-apply-ai ai-worker',
        'success',
    )
    return redirect(url_for('ai_cv_task_progress', task_id=task_id))


@app.route('/jobs/batch/make_cvs/progress')
def batch_make_cvs_progress_legacy():
    """Legacy URL — redirect to active CV task progress if available."""
    task_id = session.get('cv_generation_active')
    if task_id and _resolve_task(task_id):
        return redirect(url_for('ai_cv_task_progress', task_id=task_id))
    flash('No active batch CV generation.', 'info')
    return redirect(url_for('job_list'))


@app.route('/jobs/batch/make_cvs/complete/<task_id>')
def batch_make_cvs_complete(task_id):
    """Render batch success page after selected-job generation completes."""
    back_url = session.get('batch_cv_back_url') or url_for('job_list')
    task = _resolve_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('Batch CV generation result not found', 'error')
        session.pop('batch_cv_back_url', None)
        session.pop('batch_cv_back_label', None)
        return redirect(back_url)

    result = task['result']
    back_label = session.pop('batch_cv_back_label', None) or 'Back to Job List'
    session.pop('cv_generation_active', None)
    session.pop('batch_cv_back_url', None)
    session['generated_cvs'] = result.get('generated_cvs', [])
    session['successful_jobs'] = result.get('successful_jobs', [])
    session['failed_jobs'] = result.get('failed_jobs', [])

    flash(f"Successfully generated {result.get('generated_count', 0)} tailored CVs with RAG + AI", 'success')
    if result.get('failed_jobs'):
        flash(f"Failed to generate {len(result['failed_jobs'])} CVs", 'warning')

    return render_template(
        'all_cvs_success.html',
        successful_jobs=result.get('successful_jobs', []),
        failed_jobs=result.get('failed_jobs', []),
        back_url=back_url,
        back_label=back_label,
    )


@app.route('/download_all_cvs')
def download_all_cvs():
    """Download all generated CVs as a zip file."""
    generated_cvs = session.get('generated_cvs', [])
    
    if not generated_cvs:
        flash('No generated CVs available', 'error')
        return redirect(url_for('job_list'))
    
    # Create a zip file in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for cv_path in generated_cvs:
            if os.path.exists(cv_path):
                zf.write(cv_path, os.path.basename(cv_path))
                pdf_path = pdf_path_for_docx(cv_path)
                if os.path.exists(pdf_path):
                    zf.write(pdf_path, os.path.basename(pdf_path))
    
    # Reset file pointer
    memory_file.seek(0)
    
    # Create a date-stamped filename for the zip
    today_date = datetime.today().strftime("%Y-%m-%d")
    zip_filename = f"All_CVs_{today_date}.zip"
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename
    )

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

def main():
    """Run the Flask application."""
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    ensure_directory_exists(templates_dir)
    
    # Create basic templates if they don't exist
    create_basic_templates(templates_dir)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5050)

def create_basic_templates(templates_dir):
    """Create basic HTML templates if they don't exist."""
    templates = {
        'index.html': '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Job Application AI Agent</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding-top: 20px; padding-bottom: 40px; }
                .jumbotron { padding: 2rem; background-color: #f8f9fa; border-radius: 0.3rem; margin-bottom: 2rem; }
                .card { margin-bottom: 1.5rem; }
                .btn-primary { background-color: #0d6efd; }
                .btn-success { background-color: #198754; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="jumbotron">
                    <h1 class="display-4">Job Application AI Agent</h1>
                    <p class="lead">Search for jobs and generate tailored CVs</p>
                    <hr class="my-4">
                    
                    {% with messages = get_flashed_messages(with_categories=true) %}
                        {% if messages %}
                            {% for category, message in messages %}
                                <div class="alert alert-{{ category }}">{{ message }}</div>
                            {% endfor %}
                        {% endif %}
                    {% endwith %}
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header bg-primary text-white">
                                    <h5 class="mb-0">Step 1: Upload CV Template</h5>
                                </div>
                                <div class="card-body">
                                    <p>First, upload your CV template (.docx format)</p>
                                    <a href="{{ url_for('upload_cv') }}" class="btn btn-primary">Upload CV Template</a>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header bg-primary text-white">
                                    <h5 class="mb-0">Step 2: Search for Jobs</h5>
                                </div>
                                <div class="card-body">
                                    <form action="{{ url_for('search_jobs') }}" method="post">
                                        <div class="mb-3">
                                            <label for="keyword" class="form-label">Job Title</label>
                                            <input type="text" class="form-control" id="keyword" name="keyword" placeholder="e.g., Software Engineer" required>
                                        </div>
                                        <div class="mb-3">
                                            <label for="location" class="form-label">Location</label>
                                            <input type="text" class="form-control" id="location" name="location" placeholder="e.g., Remote, Berlin" required>
                                        </div>
                                        <div class="mb-3">
                                            <label for="max_jobs" class="form-label">Number of Jobs</label>
                                            <input type="number" class="form-control" id="max_jobs" name="max_jobs" value="5" min="1" max="20">
                                        </div>
                                        <button type="submit" class="btn btn-primary">Search Jobs</button>
                                    </form>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        ''',
        
        'upload_cv.html': '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Upload CV Template</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding-top: 20px; padding-bottom: 40px; }
                .card { margin-top: 2rem; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Upload CV Template</h1>
                
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }}">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                <a href="{{ url_for('index') }}" class="btn btn-secondary mb-3">Back to Home</a>
                
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Upload Your CV Template (.docx)</h5>
                    </div>
                    <div class="card-body">
                        <form action="{{ url_for('upload_cv') }}" method="post" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="cv_file" class="form-label">Select CV Template File</label>
                                <input class="form-control" type="file" id="cv_file" name="cv_file" accept=".docx" required>
                                <div class="form-text">Please upload a Microsoft Word (.docx) file.</div>
                            </div>
                            <button type="submit" class="btn btn-primary">Upload</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        ''',
        
        'job_list.html': '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Job Listings</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding-top: 20px; padding-bottom: 40px; }
                .job-card { margin-bottom: 1.5rem; }
                .skills-badge { 
                    display: inline-block;
                    background-color: #e9ecef;
                    padding: 0.25rem 0.5rem;
                    border-radius: 0.25rem;
                    margin-right: 0.5rem;
                    margin-bottom: 0.5rem;
                    font-size: 0.875rem;
                }
                .action-buttons {
                    display: flex;
                    justify-content: space-between;
                    margin-top: 1rem;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Job Listings</h1>
                
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }}">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                <div class="action-buttons mb-4">
                    <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Home</a>
                    
                    <div>
                        {% if excel_file %}
                            <a href="{{ url_for('download_excel') }}" class="btn btn-success">Download Excel</a>
                        {% endif %}
                        
                        {% if jobs %}
                            <a href="{{ url_for('make_all_cvs') }}" class="btn btn-primary">Generate All CVs</a>
                        {% endif %}
                    </div>
                </div>
                
                <div class="alert alert-info">
                    <h4 class="alert-heading">Found {{ jobs|length }} Jobs</h4>
                    <p>Click "Make CV" to generate a tailored CV for a specific job.</p>
                </div>
                
                {% for job in jobs %}
                <div class="card job-card">
                    <div class="card-header">
                        <h5 class="mb-0">{{ job.title }}</h5>
                    </div>
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">{{ job.company }}</h6>
                        
                        {% if job.matched_skills %}
                            <div class="mt-3">
                                <h6>Matched Skills:</h6>
                                <div>
                                    {% for skill in job.matched_skills %}
                                        <span class="skills-badge">{{ skill }}</span>
                                    {% endfor %}
                                </div>
                            </div>
                        {% endif %}
                        
                        <div class="d-flex justify-content-between align-items-center mt-3">
                            <a href="{{ job.link }}" target="_blank" class="btn btn-outline-primary btn-sm">View on LinkedIn</a>
                            <a href="{{ url_for('make_cv', job_id=loop.index0) }}" class="btn btn-success">Make CV</a>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        ''',
        
        'cv_success.html': '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>CV Generated Successfully</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding-top: 20px; padding-bottom: 40px; }
                .category-card { margin-bottom: 1rem; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="alert alert-success">
                    <h4 class="alert-heading">Success!</h4>
                    <p>Your CV has been tailored for the position of <strong>{{ job.title }}</strong> at <strong>{{ job.company }}</strong>.</p>
                </div>
                
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }}">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Skills Added to Your CV</h5>
                    </div>
                    <div class="card-body">
                        {% if matched_categories %}
                            {% for category, skills in matched_categories.items() %}
                                <div class="card category-card">
                                    <div class="card-header">
                                        <h6 class="mb-0">{{ category }}</h6>
                                    </div>
                                    <div class="card-body">
                                        <p>{{ skills|join(', ') }}</p>
                                    </div>
                                </div>
                            {% endfor %}
                        {% else %}
                            <p class="text-muted">No specific skills matched.</p>
                        {% endif %}
                    </div>
                </div>
                
                <div class="d-flex justify-content-between">
                    <a href="{{ url_for('job_list') }}" class="btn btn-secondary">Back to Job List</a>
                    <a href="{{ url_for('download_cv') }}" class="btn btn-primary">Download CV</a>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        ''',
        
        'all_cvs_success.html': '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>All CVs Generated</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding-top: 20px; padding-bottom: 40px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="alert alert-success">
                    <h4 class="alert-heading">Success!</h4>
                    <p>Successfully generated {{ cv_count }} tailored CVs.</p>
                </div>
                
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }}">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Download Options</h5>
                    </div>
                    <div class="card-body">
                        <p>You can download all generated CVs as a ZIP file.</p>
                        <a href="{{ url_for('download_all_cvs') }}" class="btn btn-primary">Download All CVs (ZIP)</a>
                    </div>
                </div>
                
                <a href="{{ url_for('job_list') }}" class="btn btn-secondary">Back to Job List</a>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        ''',
        
        '404.html': '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Page Not Found</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5 text-center">
                <h1 class="display-1">404</h1>
                <h2>Page Not Found</h2>
                <p class="lead">The page you are looking for does not exist.</p>
                <a href="{{ url_for('index') }}" class="btn btn-primary">Go Home</a>
            </div>
        </body>
        </html>
        ''',
        
        '500.html': '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Server Error</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5 text-center">
                <h1 class="display-1">500</h1>
                <h2>Server Error</h2>
                <p class="lead">Something went wrong on our end. Please try again later.</p>
                <a href="{{ url_for('index') }}" class="btn btn-primary">Go Home</a>
            </div>
        </body>
        </html>
        '''
    }
    
    for filename, content in templates.items():
        filepath = os.path.join(templates_dir, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
            logger.info(f"Created template: {filename}")

# Add template filter to get basename from path
@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)

if __name__ == '__main__':
    main() 