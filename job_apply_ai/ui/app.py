"""
Web Interface for Job Application AI Agent

This module provides a Flask web application for the job application AI agent.
"""

import os
import logging
import secrets
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify
import zipfile
import io

from job_apply_ai.scraper.aggregator import search_jobs as aggregate_search_jobs
from job_apply_ai.batch_search import (
    build_search_queue,
    decode_uploaded_text,
    parse_lines,
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
from job_apply_ai.cv_modifier.cover_letter_builder import CoverLetterBuilder
from job_apply_ai.cv_modifier.cover_letter_chat_editor import CoverLetterChatEditor
from job_apply_ai.cv_modifier.cover_letter_generator import CoverLetterGenerator
from job_apply_ai.cv_modifier.cv_chat_editor import CVChatEditor
from job_apply_ai.cv_modifier.cv_content_store import load_cv_content, save_cv_content
from job_apply_ai.cv_modifier.cv_generator import RAGCVGenerator
from job_apply_ai.cv_modifier.profile_importer import ProfileImporter
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
from job_apply_ai.utils.helpers import ensure_directory_exists, sanitize_filename
from job_apply_ai.storage.database import init_db
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.storage.user_profile import (
    UserProfileRepository,
    get_default_cv_template_path,
    import_has_changes,
    merge_profiles,
    profile_from_form,
    profile_is_ready,
    profile_to_form_fields,
    remove_smtp_account,
    set_default_smtp_account,
    summarize_import_changes,
    update_smtp_account_tokens,
    upsert_oauth_smtp_account,
)
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
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'job_apply_ai')
ensure_directory_exists(app.config['UPLOAD_FOLDER'])

# Create output directories
app.config['CV_OUTPUT_DIR'] = os.path.join(app.config['UPLOAD_FOLDER'], 'cvs')
app.config['JOBS_OUTPUT_DIR'] = os.path.join(app.config['UPLOAD_FOLDER'], 'jobs')
ensure_directory_exists(app.config['CV_OUTPUT_DIR'])
ensure_directory_exists(app.config['JOBS_OUTPUT_DIR'])

# Initialize SQLite database
init_db()
job_repo = JobRepository()
profile_repo = UserProfileRepository()

# Clear abandoned background tasks after this many seconds without progress
BACKGROUND_TASK_STALE_SECONDS = 900
BACKGROUND_TASK_SESSION_KEYS = (
    'cv_generation_active',
    'batch_search_active',
    'job_match_analyze_active',
)

# Ensure session data is saved
app.config['SESSION_TYPE'] = 'filesystem'


def _sync_session_background_task(session_key: str) -> str | None:
    """Drop stale session task pointers and return the task id only if still active."""
    task_id = session.get(session_key)
    if not task_id:
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
        'batch_search': 'Batch job search',
        'job_match_analyze': 'Analyzing Profile Match',
        'single_cv': 'Generating AI CV',
        'batch_cv': 'Generating AI CVs',
    }
    return labels.get(task.get('task_type', ''), 'Background task')


def _background_task_progress_url(task_id: str, task: dict) -> str | None:
    task_type = task.get('task_type')
    if task_type == 'batch_search':
        return url_for('batch_search_progress', task_id=task_id)
    if task_type == 'job_match_analyze':
        return url_for('job_match_analyze_progress', task_id=task_id)
    if task_type == 'single_cv':
        job_id = task.get('job_id')
        if job_id:
            return url_for('make_cv', job_id=job_id)
    if task_type == 'batch_cv':
        return url_for('make_all_cvs')
    return None


def _active_background_tasks() -> list[dict]:
    """Build UI entries for in-progress background tasks the user can return to."""
    active_tasks: list[dict] = []
    for session_key in BACKGROUND_TASK_SESSION_KEYS:
        task_id = session.get(session_key)
        if not task_id:
            continue
        task = get_task(task_id)
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
        if session_key == 'cv_generation_active':
            entry['release_url'] = url_for('release_cv_generation_lock')
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
        'active_background_tasks': _active_background_tasks(),
        'smtp_configured': bool(accounts),
        'smtp_accounts': accounts,
        'google_oauth_configured': google_oauth_configured(),
        'microsoft_oauth_configured': microsoft_oauth_configured(),
        'job_sort_options': JOB_SORT_OPTIONS,
        'default_job_sort': DEFAULT_JOB_SORT,
    }


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


def _load_job_cv_store(cv_filename: str) -> dict | None:
    return load_cv_content(app.config['CV_OUTPUT_DIR'], cv_filename)


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

    job_repo.update_job_status(job_id, 'cv_sent')
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
    cl_content = generator.generate(job, profile, tailored_content)
    CoverLetterBuilder().build(cl_path, cl_content)

    if job_id:
        job['cover_letter_filename'] = cl_filename
        job_repo.update_job(job_id, job)

    store = _load_job_cv_store(cv_filename) or {}
    _save_job_cv_content(
        cv_filename,
        store.get('tailored_content', tailored_content),
        chat_history=store.get('chat_history', []),
        cover_letter=cl_content,
        cover_letter_chat_history=(
            [] if reset_cover_letter_chat else store.get('cover_letter_chat_history', [])
        ),
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
    store = _load_job_cv_store(cv_filename) if cv_filename else None
    profile = profile_repo.get_profile()
    content = tailored_content or (store or {}).get('tailored_content', {})
    chat_history = (store or {}).get('chat_history', [])
    cover_letter = (store or {}).get('cover_letter', {})
    cover_letter_chat_history = (store or {}).get('cover_letter_chat_history', [])
    categories = matched_categories or CVChatEditor.content_to_matched_categories(content)
    job_id = job.get('id')
    cover_letter_filename = job.get('cover_letter_filename', '')

    return {
        'job': job,
        'job_id': job_id,
        'profile_name': profile.get('full_name', ''),
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
        'cover_letter_download_url': (
            url_for('download_job_cover_letter', job_id=job_id)
            if job_id and cover_letter_filename
            else None
        ),
        'chat_api_url': url_for('document_chat', job_id=job_id) if job_id else None,
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

    def on_progress(step: str, message: str, percent: int) -> None:
        if task_id:
            update_task(
                task_id,
                status='running',
                step=step,
                message=message,
                percent=percent,
            )

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


CONTROLLABLE_TASK_TYPES = frozenset({'batch_search', 'job_match_analyze'})


def _run_batch_search_task(
    task_id: str,
    queue: list[tuple[str, str]],
    max_jobs: int,
    sources: str,
    source_list: list[str],
    mode: str,
    profile: dict,
) -> None:
    total = len(queue)
    unique_titles = len({keyword for keyword, _ in queue})
    unique_locations = len({location for _, location in queue})
    search_run_id = job_repo.create_search_run(
        f'batch: {unique_titles} title(s)',
        f'batch: {unique_locations} location(s)',
        sources,
        mode,
    )

    total_jobs_saved = 0
    failed_searches: list[dict] = []
    stopped = False
    searches_completed = 0

    update_task(
        task_id,
        status='running',
        step='searching',
        message=f'Starting batch search ({total} combinations)…',
        percent=1,
        meta={'total_searches': total},
    )

    for index, (keyword, location) in enumerate(queue, start=1):
        try:
            task_control_checkpoint(task_id)
        except TaskStopped:
            stopped = True
            break

        percent = max(1, min(99, int(((index - 1) / total) * 100)))
        update_task(
            task_id,
            status='running',
            step='searching',
            message=f'Searching {index} of {total}',
            percent=percent,
            meta={
                'current_index': index,
                'total_searches': total,
                'keyword': keyword,
                'location': location,
            },
        )

        try:
            jobs = aggregate_search_jobs(
                keyword,
                location,
                max_jobs=max_jobs,
                sources=source_list,
                mode=mode,
                enrich_details=True,
            )
            if jobs:
                processed_jobs = _enrich_jobs_with_skills(jobs)
                processed_jobs = classify_jobs_by_profile_match(processed_jobs, profile)
                job_repo.upsert_jobs(processed_jobs, search_run_id=search_run_id)
                total_jobs_saved += len(processed_jobs)
            searches_completed += 1
        except Exception as exc:
            logger.error(
                'Batch search failed for %r in %r: %s',
                keyword,
                location,
                exc,
            )
            failed_searches.append(
                {
                    'keyword': keyword,
                    'location': location,
                    'error': str(exc),
                }
            )
            searches_completed += 1

    if stopped:
        if total_jobs_saved == 0:
            fail_task(task_id, 'Batch search stopped before saving any jobs.')
            return

        message = (
            f'Batch search stopped — saved {total_jobs_saved} jobs '
            f'after {searches_completed} of {total} searches'
        )
        if failed_searches:
            message += f' ({len(failed_searches)} searches failed)'
        complete_task(
            task_id,
            {
                'search_run_id': search_run_id,
                'total_jobs': total_jobs_saved,
                'searches_run': searches_completed,
                'failed_searches': failed_searches,
                'stopped': True,
            },
            message=message,
        )
        return

    if total_jobs_saved == 0:
        detail = (
            f'{len(failed_searches)} of {total} searches failed.'
            if failed_searches
            else 'No jobs matched any title/location combination.'
        )
        fail_task(task_id, f'Batch search found no jobs. {detail}')
        return

    message = f'Batch search complete — saved {total_jobs_saved} jobs'
    if failed_searches:
        message += f' ({len(failed_searches)} searches failed)'

    complete_task(
        task_id,
        {
            'search_run_id': search_run_id,
            'total_jobs': total_jobs_saved,
            'searches_run': total,
            'failed_searches': failed_searches,
        },
        message=message,
    )


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

    on_progress('validating_ollama', 'Checking Ollama and installed models…', 5)
    if not generator.ollama.is_available():
        raise RuntimeError('Ollama is not reachable.')

    generator.ollama.validate_models()
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
    """Handle job search form and display results."""
    if request.method == 'POST':
        keyword = request.form.get('keyword', '')
        location = request.form.get('location', '')
        max_jobs = int(request.form.get('max_jobs', 10))
        sources = request.form.get('sources', 'linkedin,adzuna,reed,indeed')
        mode = request.form.get('mode', 'both')
        
        if not keyword or not location:
            flash('Please enter both job title and location', 'error')
            return redirect(url_for('index'))
        
        try:
            source_list = [source.strip() for source in sources.split(',') if source.strip()]
            jobs = aggregate_search_jobs(
                keyword,
                location,
                max_jobs=max_jobs,
                sources=source_list,
                mode=mode,
                enrich_details=True,
            )
            
            if not jobs:
                flash('No jobs found. Try different search terms.', 'warning')
                return redirect(url_for('index'))

            search_run_id = job_repo.create_search_run(
                keyword,
                location,
                sources,
                mode,
            )
            processed_jobs = _enrich_jobs_with_skills(jobs)
            profile = profile_repo.get_profile()
            processed_jobs = classify_jobs_by_profile_match(processed_jobs, profile)
            job_repo.upsert_jobs(processed_jobs, search_run_id=search_run_id)
            processed_jobs = sort_jobs(
                job_repo.list_jobs(search_run_id=search_run_id),
                'match_desc',
            )

            session['search_run_id'] = search_run_id

            return render_template(
                'job_list.html',
                jobs=processed_jobs,
                search_run_id=search_run_id,
                current_sort='match_desc',
                job_sort_options=JOB_SORT_OPTIONS,
            )
            
        except Exception as e:
            logger.error(f"Error during job search: {str(e)}")
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/search/batch', methods=['POST'])
def batch_search_jobs():
    """Queue a batch search: every job title × every location."""
    titles = _read_batch_lines_from_request('titles_file', 'titles_text')
    locations = _read_batch_lines_from_request('locations_file', 'locations_text')
    queue = build_search_queue(titles, locations)
    queue_error = validate_batch_queue(queue)

    if queue_error:
        flash(queue_error, 'error')
        return redirect(url_for('index'))

    max_jobs = int(request.form.get('max_jobs', 5))
    sources = request.form.get('sources', 'linkedin,adzuna,reed,indeed')
    mode = request.form.get('mode', 'both')
    source_list = [source.strip() for source in sources.split(',') if source.strip()]
    profile = profile_repo.get_profile()

    task_id = create_task(
        'batch_search',
        meta={
            'total_searches': len(queue),
            'titles': len(titles),
            'locations': len(locations),
        },
    )
    session['batch_search_active'] = task_id
    start_background_task(
        task_id,
        lambda: _run_batch_search_task(
            task_id,
            queue,
            max_jobs,
            sources,
            source_list,
            mode,
            profile,
        ),
    )
    return redirect(url_for('batch_search_progress', task_id=task_id))


@app.route('/search/batch/<task_id>')
def batch_search_progress(task_id):
    """Show progress while batch job search runs."""
    task = get_task(task_id)
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
    task = get_task(task_id)
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


def _clear_profile_import_session() -> None:
    session.pop('profile_draft', None)
    session.pop('profile_import_summary', None)


def _run_profile_import_task(task_id: str, cv_path: str, current_profile: dict) -> None:
    try:
        update_task(task_id, status='running', step='extracting', message='Reading CV document…', percent=10)
        importer = ProfileImporter()
        extracted = importer.extract_from_docx(cv_path)

        update_task(
            task_id,
            step='merging',
            message='Merging new details into your profile…',
            percent=70,
        )
        merged_profile, changes = merge_profiles(current_profile, extracted)
        summary_lines = summarize_import_changes(changes)

        complete_task(
            task_id,
            {
                'form': profile_to_form_fields(merged_profile),
                'import_summary': summary_lines,
                'has_changes': import_has_changes(changes),
                'merged_profile': merged_profile,
            },
        )
    finally:
        if os.path.exists(cv_path):
            os.remove(cv_path)


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

    task_id = create_task('profile_import')
    current_profile = profile_repo.get_profile()
    start_background_task(
        task_id,
        lambda: _run_profile_import_task(task_id, temp_path, current_profile),
    )
    return redirect(url_for('profile_import_progress', task_id=task_id))


@app.route('/profile/import/<task_id>')
def profile_import_progress(task_id):
    """Show progress while a CV import is being parsed."""
    task = get_task(task_id)
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
    task = get_task(task_id)
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
        profile_has_matchable_skills=profile_has_matchable_skills(profile_repo.get_profile()),
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
                job_repo.update_job_status(job_id, new_status)

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
        flash('Add technical skills, minor skills, or stacks on your profile before running match analysis.', 'warning')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    jobs = _jobs_for_manage_folder(return_folder, return_search)
    if not jobs:
        flash('No jobs to analyze in this folder.', 'warning')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    task_id = create_task(
        'job_match_analyze',
        meta={
            'total_jobs': len(jobs),
            'min_match_score': min_match_score,
            'return_folder': return_folder,
            'return_search': return_search,
            'return_sort': return_sort,
        },
    )
    session['job_match_analyze_active'] = task_id
    start_background_task(
        task_id,
        lambda: _run_job_match_analyze_task(
            task_id,
            jobs,
            profile,
            min_match_score,
            return_folder,
            return_search,
            return_sort,
        ),
    )
    return redirect(url_for('job_match_analyze_progress', task_id=task_id))


@app.route('/jobs/manage/analyze-match/<task_id>')
def job_match_analyze_progress(task_id):
    """Show progress while jobs are analyzed against the profile."""
    task = get_task(task_id)
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
    task = get_task(task_id)
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
        if workflow_status != job.get('workflow_status', DEFAULT_JOB_STATUS):
            job_repo.update_job_status(job_id, workflow_status)
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
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '')
    return_sort = request.form.get('return_sort', '')

    if not is_valid_job_status(workflow_status):
        flash('Invalid job status', 'error')
        return _manage_jobs_redirect(return_folder, return_search, return_sort)

    if job_repo.update_job_status(job_id, workflow_status):
        flash(f'Job moved to {job_status_label(workflow_status)}', 'success')
    else:
        flash('Job not found', 'error')
    return _manage_jobs_redirect(return_folder, return_search, return_sort)


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
    """Show progress UI while AI CV generation runs in the background."""
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

    return render_template(
        'cv_progress.html',
        job=job,
        batch=False,
        start_url=url_for(
            'start_make_cv',
            job_id=job_id,
            folder=return_folder,
            q=return_search or None,
            sort=return_sort or None,
        ),
        status_url_template=url_for('cv_task_status', task_id='TASK_ID'),
        complete_url_template=url_for('make_cv_complete', task_id='TASK_ID'),
        back_url=_cv_generation_back_url(
            return_from_manage,
            return_folder,
            return_search,
            return_sort,
        ),
    )


@app.route('/make_cv/<int:job_id>/start', methods=['POST'])
def start_make_cv(job_id):
    """Start background AI CV generation for one job."""
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

    if _sync_cv_generation_lock():
        return jsonify({'error': 'Another CV generation is already in progress'}), 409

    task_id = create_task('single_cv', job_id=job_id)
    session['cv_generation_active'] = task_id
    start_background_task(
        task_id,
        lambda: _run_single_cv_task(
            task_id,
            profile,
            job,
            job_id,
            return_folder,
            return_search,
            return_from_manage,
            return_sort,
        ),
    )
    return jsonify({'task_id': task_id})


@app.route('/api/cv_generation/release', methods=['POST', 'GET'])
def release_cv_generation_lock():
    """Clear UI lock after failed, abandoned, or stale CV generation."""
    session.pop('cv_generation_active', None)
    if request.method == 'GET':
        flash('CV generation lock cleared. You can start a new CV.', 'info')
        return redirect(request.referrer or url_for('manage_jobs'))
    return jsonify({'ok': True})


@app.route('/api/cv_tasks/<task_id>/status')
def cv_task_status(task_id):
    """Poll background CV generation progress."""
    task = get_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


@app.route('/api/cv_tasks/<task_id>/control', methods=['POST'])
def control_background_task(task_id):
    """Pause, resume, or stop a controllable background task."""
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
    task = get_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('CV generation result not found', 'error')
        return redirect(url_for('job_list'))

    result = task['result']
    session.pop('cv_generation_active', None)
    session['current_cv'] = result.get('output_path')
    session['current_cv_filename'] = result.get('cv_filename')

    flash('Professional CV and cover letter generated successfully with RAG + Ollama', 'success')
    return_from_manage = result.get('return_from_manage', False)
    return_folder = result.get('return_folder', 'all')
    return_search = result.get('return_search', '')
    return_sort = result.get('return_sort', '')
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

    store = _load_job_cv_store(cv_filename)
    if not store or not store.get('tailored_content'):
        flash('CV preview data not found. Regenerate the CV to enable preview and chat editing.', 'warning')
        if return_from_manage:
            return _manage_jobs_redirect(return_folder, return_search, return_sort)
        return redirect(url_for('job_list', sort=return_sort or None))

    context = _cv_preview_context(
        job,
        show_success_banner=False,
        return_folder=return_folder,
        return_search=return_search,
        return_sort=return_sort,
        return_from_manage=return_from_manage,
    )
    return render_template('cv_success.html', **context)


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

    store = _load_job_cv_store(cv_filename) or {}
    profile = profile_repo.get_profile()

    try:
        if document_type == 'cover_letter':
            cl_filename = job.get('cover_letter_filename', '')
            cl_path = os.path.join(app.config['CV_OUTPUT_DIR'], cl_filename)
            if not cl_filename or not store.get('cover_letter'):
                return jsonify({'error': 'Cover letter not available for editing'}), 404

            chat_history = store.get('cover_letter_chat_history', [])
            editor = CoverLetterChatEditor()
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

            chat_history = chat_history + [
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': reply},
            ]
            _save_job_cv_content(
                cv_filename,
                store.get('tailored_content', {}),
                chat_history=store.get('chat_history', []),
                cover_letter=updated_content,
                cover_letter_chat_history=chat_history,
            )

            return jsonify({
                'reply': reply,
                'document': 'cover_letter',
                'cover_letter': updated_content,
                'cover_letter_chat_history': chat_history,
            })

        if not store.get('tailored_content'):
            return jsonify({'error': 'CV content not available for editing'}), 404

        chat_history = store.get('chat_history', [])
        current_content = store['tailored_content']
        editor = CVChatEditor()
        result = editor.modify(
            current_content=current_content,
            user_message=user_message,
            job=job,
            profile=profile,
            chat_history=chat_history,
        )
        updated_content = result['content']
        reply = result['reply']

        editor.rebuild_document(cv_path, updated_content, profile)
        matched_categories = CVChatEditor.content_to_matched_categories(updated_content)
        job['matched_categories'] = matched_categories
        job_repo.update_job(job_id, job)

        chat_history = chat_history + [
            {'role': 'user', 'content': user_message},
            {'role': 'assistant', 'content': reply},
        ]
        _save_job_cv_content(
            cv_filename,
            updated_content,
            chat_history=chat_history,
            cover_letter=store.get('cover_letter', {}),
            cover_letter_chat_history=store.get('cover_letter_chat_history', []),
        )

        session['current_cv'] = cv_path
        session['current_cv_filename'] = cv_filename

        return jsonify({
            'reply': reply,
            'document': 'cv',
            'content': updated_content,
            'matched_categories': matched_categories,
            'chat_history': chat_history,
        })
    except Exception as exc:
        logger.error('Document chat edit failed for job %s: %s', job_id, exc)
        return jsonify({'error': str(exc)}), 500


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


@app.route('/make_all_cvs')
def make_all_cvs():
    """Show progress UI while batch AI CV generation runs."""
    search_run_id = session.get('search_run_id')
    processed_jobs = _get_jobs_for_view(search_run_id)
    profile = profile_repo.get_profile()

    if not processed_jobs:
        flash('No jobs found. Please search for jobs first.', 'error')
        return redirect(url_for('index'))

    if not profile_is_ready(profile):
        flash('Please complete your CV profile first', 'error')
        return redirect(url_for('user_profile'))

    return render_template(
        'cv_progress.html',
        job=None,
        batch=True,
        job_count=len(processed_jobs),
        start_url=url_for('start_make_all_cvs'),
        status_url_template=url_for('cv_task_status', task_id='TASK_ID'),
        complete_url_template=url_for('make_all_cvs_complete', task_id='TASK_ID'),
        back_url=url_for('job_list'),
    )


@app.route('/make_all_cvs/start', methods=['POST'])
def start_make_all_cvs():
    """Start background batch AI CV generation."""
    search_run_id = session.get('search_run_id')
    processed_jobs = _get_jobs_for_view(search_run_id)
    profile = profile_repo.get_profile()

    if not processed_jobs:
        return jsonify({'error': 'No jobs found'}), 404
    if not profile_is_ready(profile):
        return jsonify({'error': 'Please complete your CV profile first'}), 400

    if _sync_cv_generation_lock():
        return jsonify({'error': 'Another CV generation is already in progress'}), 409

    task_id = create_task('batch_cv', meta={'total_jobs': len(processed_jobs)})
    session['cv_generation_active'] = task_id
    start_background_task(
        task_id,
        lambda: _run_batch_cv_task(task_id, profile, processed_jobs),
    )
    return jsonify({'task_id': task_id})


@app.route('/make_all_cvs/complete/<task_id>')
def make_all_cvs_complete(task_id):
    """Render batch success page after background generation completes."""
    task = get_task(task_id)
    if not task or task.get('status') != 'complete' or not task.get('result'):
        flash('Batch CV generation result not found', 'error')
        return redirect(url_for('job_list'))

    result = task['result']
    session.pop('cv_generation_active', None)
    session['generated_cvs'] = result.get('generated_cvs', [])
    session['successful_jobs'] = result.get('successful_jobs', [])
    session['failed_jobs'] = result.get('failed_jobs', [])

    flash(f"Successfully generated {result.get('generated_count', 0)} tailored CVs with RAG + Ollama", 'success')
    if result.get('failed_jobs'):
        flash(f"Failed to generate {len(result['failed_jobs'])} CVs", 'warning')

    return render_template(
        'all_cvs_success.html',
        successful_jobs=result.get('successful_jobs', []),
        failed_jobs=result.get('failed_jobs', []),
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
                # Add file to zip with just the filename (not the full path)
                zf.write(cv_path, os.path.basename(cv_path))
    
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