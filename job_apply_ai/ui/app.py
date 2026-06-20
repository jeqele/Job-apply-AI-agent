"""
Web Interface for Job Application AI Agent

This module provides a Flask web application for the job application AI agent.
"""

import os
import logging
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify
import zipfile
import io

from job_apply_ai.scraper.aggregator import search_jobs as aggregate_search_jobs
from job_apply_ai.job_schema import JOB_COLUMNS
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
from job_apply_ai.cv_modifier.cv_generator import RAGCVGenerator
from job_apply_ai.ui.cv_tasks import (
    complete_task,
    create_task,
    get_task,
    start_background_task,
    update_task,
)
from job_apply_ai.utils.helpers import ensure_directory_exists, sanitize_filename
from job_apply_ai.storage.database import init_db
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.storage.exports import export_jobs

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

# Ensure session data is saved
app.config['SESSION_TYPE'] = 'filesystem'

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


def _generate_rag_cv(
    cv_template: str,
    job: dict,
    *,
    reindex: bool = True,
    task_id: str | None = None,
) -> dict:
    """Generate a tailored CV using RAG + Ollama."""
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
        cv_template,
        job,
        output_path,
        reindex=reindex,
        on_progress=on_progress,
    )
    result['output_filename'] = output_filename
    return result


def _cv_generation_back_url(return_from_manage: bool, return_folder: str, return_search: str):
    if return_from_manage:
        kwargs = {}
        if return_search:
            kwargs['q'] = return_search
        if return_folder and return_folder != 'all':
            kwargs['folder'] = return_folder
        return url_for('manage_jobs', **kwargs)
    return url_for('job_list')


def _run_single_cv_task(
    task_id: str,
    cv_template: str,
    job: dict,
    job_id: int,
    return_folder: str,
    return_search: str,
    return_from_manage: bool,
) -> None:
    result = _generate_rag_cv(cv_template, job, task_id=task_id)
    output_path = result['output_path']
    output_filename = result['output_filename']
    tailored_content = result.get('tailored_content', {})

    matched_categories = {'Key Skills': tailored_content.get('key_skills', [])}
    job['matched_categories'] = matched_categories
    job['cv_filename'] = output_filename
    job_repo.update_job(job_id, job)

    complete_task(
        task_id,
        {
            'job': job,
            'cv_filename': output_filename,
            'output_path': output_path,
            'matched_categories': matched_categories,
            'tailored_content': tailored_content,
            'analysis': result.get('analysis', {}),
            'generation_meta': result.get('models', {}),
            'rag_chunk_count': result.get('chunk_count', 0),
            'return_folder': return_folder,
            'return_search': return_search,
            'return_from_manage': return_from_manage,
        },
    )


def _run_batch_cv_task(task_id: str, cv_template: str, jobs: list[dict]) -> None:
    generator = RAGCVGenerator()

    def on_progress(step: str, message: str, percent: int) -> None:
        update_task(task_id, status='running', step=step, message=message, percent=percent)

    on_progress('validating_ollama', 'Checking Ollama and installed models…', 5)
    if not generator.ollama.is_available():
        raise RuntimeError('Ollama is not reachable.')

    generator.ollama.validate_models()
    on_progress('indexing_cv', 'Indexing your CV with RAG…', 12)
    generator.prepare_cv_index(cv_template)

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
                cv_template,
                job,
                output_path,
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
                job['matched_categories'] = {'Key Skills': tailored_content.get('key_skills', [])}
                job['cv_filename'] = output_filename
                job_repo.update_job(job_id, job)
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


def _get_jobs_for_view(search_run_id: int | None = None) -> list[dict]:
    """Load jobs from SQLite for the current view."""
    if search_run_id is not None:
        jobs = job_repo.list_jobs(search_run_id=search_run_id)
        if jobs:
            return jobs
    return job_repo.list_jobs()


def _job_form_data() -> dict:
    """Read job fields from the current request form."""
    data = {column: request.form.get(column, '') for column in JOB_COLUMNS}
    workflow_status = request.form.get('workflow_status', DEFAULT_JOB_STATUS)
    data['workflow_status'] = (
        workflow_status if is_valid_job_status(workflow_status) else DEFAULT_JOB_STATUS
    )
    return data


def _manage_jobs_redirect(folder: str = 'all', search: str = ''):
    """Redirect back to the manage jobs view preserving folder context."""
    kwargs = {}
    if search:
        kwargs['q'] = search
    if folder and folder != 'all':
        kwargs['folder'] = folder
    return redirect(url_for('manage_jobs', **kwargs))


@app.template_filter('job_status_label')
def _job_status_label_filter(status):
    return job_status_label(status)


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

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
            job_repo.upsert_jobs(processed_jobs, search_run_id=search_run_id)
            processed_jobs = job_repo.list_jobs(search_run_id=search_run_id)

            session['search_run_id'] = search_run_id

            return render_template(
                'job_list.html',
                jobs=processed_jobs,
                search_run_id=search_run_id,
            )
            
        except Exception as e:
            logger.error(f"Error during job search: {str(e)}")
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/upload_cv', methods=['GET', 'POST'])
def upload_cv():
    """Handle CV template upload."""
    if request.method == 'POST':
        if 'cv_file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['cv_file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.docx'):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'cv_template.docx')
            file.save(filename)
            session['cv_template'] = filename
            flash(
                'CV uploaded successfully. It will be indexed with RAG when you generate tailored CVs.',
                'success',
            )
            
            # If we have jobs, redirect to job list
            if session.get('search_run_id') or job_repo.count_jobs():
                return redirect(url_for('job_list'))
            return redirect(url_for('index'))
        else:
            flash('Please upload a .docx file', 'error')
            return redirect(request.url)
    
    return render_template('upload_cv.html')

@app.route('/job_list')
def job_list():
    """Display the list of jobs with Make CV buttons."""
    search_run_id = session.get('search_run_id')
    jobs = _get_jobs_for_view(search_run_id)

    if not jobs:
        flash('No jobs found. Please search for jobs first.', 'warning')
        return redirect(url_for('index'))

    return render_template(
        'job_list.html',
        jobs=jobs,
        search_run_id=search_run_id,
    )


@app.route('/jobs/manage')
@app.route('/jobs/manage/<folder>')
def manage_jobs(folder='all'):
    """Display jobs grouped by workflow status with folder navigation."""
    search = request.args.get('q', '').strip()

    if folder != 'all' and not is_valid_job_status(folder):
        flash('Unknown job folder', 'warning')
        return redirect(url_for('manage_jobs'))

    workflow_status = None if folder == 'all' else folder
    jobs = job_repo.list_jobs(workflow_status=workflow_status, search=search or None)
    status_counts = job_repo.count_jobs_by_status()
    total_count = sum(status_counts.values())

    folder_counts = {'all': total_count}
    for status in JOB_WORKFLOW_STATUSES:
        folder_counts[status] = status_counts.get(status, 0)

    return render_template(
        'manage_jobs.html',
        jobs=jobs,
        current_folder=folder,
        search_query=search,
        folder_counts=folder_counts,
        job_statuses=JOB_WORKFLOW_STATUSES,
        status_labels=JOB_STATUS_LABELS,
        status_icons=JOB_STATUS_ICONS,
        status_badges=JOB_STATUS_BADGE_CLASSES,
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
            )

        workflow_status = job_data.pop('workflow_status', DEFAULT_JOB_STATUS)
        job_repo.update_job(job_id, job_data)
        if workflow_status != job.get('workflow_status', DEFAULT_JOB_STATUS):
            job_repo.update_job_status(job_id, workflow_status)
        flash('Job updated successfully', 'success')
        return _manage_jobs_redirect(return_folder, return_search)

    return render_template(
        'job_form.html',
        job=job,
        job_statuses=JOB_WORKFLOW_STATUSES,
        status_labels=JOB_STATUS_LABELS,
        return_folder=return_folder,
        return_search=return_search,
    )


@app.route('/jobs/<int:job_id>/status', methods=['POST'])
def update_job_status(job_id):
    """Move a job to another workflow folder."""
    workflow_status = request.form.get('workflow_status', '')
    return_folder = request.form.get('return_folder', 'all')
    return_search = request.form.get('return_search', '')

    if not is_valid_job_status(workflow_status):
        flash('Invalid job status', 'error')
        return _manage_jobs_redirect(return_folder, return_search)

    if job_repo.update_job_status(job_id, workflow_status):
        flash(f'Job moved to {job_status_label(workflow_status)}', 'success')
    else:
        flash('Job not found', 'error')
    return _manage_jobs_redirect(return_folder, return_search)


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
    cv_template = session.get('cv_template')
    return_folder = request.args.get('folder', 'all')
    return_search = request.args.get('q', '')
    return_from_manage = 'folder' in request.args or bool(return_search)

    if not job:
        flash('Job not found', 'error')
        if return_from_manage:
            return _manage_jobs_redirect(return_folder, return_search)
        return redirect(url_for('job_list'))

    if not cv_template:
        flash('Please upload your CV first', 'error')
        return redirect(url_for('upload_cv'))

    return render_template(
        'cv_progress.html',
        job=job,
        batch=False,
        start_url=url_for('start_make_cv', job_id=job_id, folder=return_folder, q=return_search or None),
        status_url_template=url_for('cv_task_status', task_id='TASK_ID'),
        complete_url_template=url_for('make_cv_complete', task_id='TASK_ID'),
        back_url=_cv_generation_back_url(return_from_manage, return_folder, return_search),
    )


@app.route('/make_cv/<int:job_id>/start', methods=['POST'])
def start_make_cv(job_id):
    """Start background AI CV generation for one job."""
    job = job_repo.get_job(job_id)
    cv_template = session.get('cv_template')
    return_folder = request.args.get('folder', 'all')
    return_search = request.args.get('q', '')
    return_from_manage = 'folder' in request.args or bool(return_search)

    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not cv_template:
        return jsonify({'error': 'Please upload your CV first'}), 400

    task_id = create_task('single_cv', job_id=job_id)
    session['cv_generation_active'] = task_id
    start_background_task(
        task_id,
        lambda: _run_single_cv_task(
            task_id,
            cv_template,
            job,
            job_id,
            return_folder,
            return_search,
            return_from_manage,
        ),
    )
    return jsonify({'task_id': task_id})


@app.route('/api/cv_generation/release', methods=['POST'])
def release_cv_generation_lock():
    """Clear UI lock after failed or abandoned CV generation."""
    session.pop('cv_generation_active', None)
    return jsonify({'ok': True})


@app.route('/api/cv_tasks/<task_id>/status')
def cv_task_status(task_id):
    """Poll background CV generation progress."""
    task = get_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


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

    flash('Professional CV generated successfully with RAG + Ollama', 'success')
    return_from_manage = result.get('return_from_manage', False)
    return_folder = result.get('return_folder', 'all')
    return_search = result.get('return_search', '')
    return render_template(
        'cv_success.html',
        job=result.get('job'),
        cv_filename=result.get('cv_filename'),
        matched_categories=result.get('matched_categories', {}),
        tailored_content=result.get('tailored_content', {}),
        analysis=result.get('analysis', {}),
        generation_meta=result.get('generation_meta', {}),
        rag_chunk_count=result.get('rag_chunk_count', 0),
        return_folder=return_folder,
        return_search=return_search,
        return_from_manage=return_from_manage,
        back_url=_cv_generation_back_url(return_from_manage, return_folder, return_search),
        back_label='Back to Jobs' if return_from_manage else 'Back to Job List',
    )

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


@app.route('/make_all_cvs')
def make_all_cvs():
    """Show progress UI while batch AI CV generation runs."""
    search_run_id = session.get('search_run_id')
    processed_jobs = _get_jobs_for_view(search_run_id)
    cv_template = session.get('cv_template')

    if not processed_jobs:
        flash('No jobs found. Please search for jobs first.', 'error')
        return redirect(url_for('index'))

    if not cv_template:
        flash('Please upload your CV first', 'error')
        return redirect(url_for('upload_cv'))

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
    cv_template = session.get('cv_template')

    if not processed_jobs:
        return jsonify({'error': 'No jobs found'}), 404
    if not cv_template:
        return jsonify({'error': 'Please upload your CV first'}), 400

    task_id = create_task('batch_cv', meta={'total_jobs': len(processed_jobs)})
    session['cv_generation_active'] = task_id
    start_background_task(
        task_id,
        lambda: _run_batch_cv_task(task_id, cv_template, processed_jobs),
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