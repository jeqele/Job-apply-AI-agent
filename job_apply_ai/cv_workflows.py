"""Shared CV / ATS / profile AI workflows for web UI and queue workers."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable, Protocol

from job_apply_ai.cv_modifier.ats_friendly_analyzer import (
    ATSFriendlyAnalyzer,
    pending_suggestions,
    update_suggestions_status,
)
from job_apply_ai.cv_modifier.chat_context import (
    resolve_effective_tailored_content,
)
from job_apply_ai.cv_modifier.cover_letter_builder import CoverLetterBuilder
from job_apply_ai.cv_modifier.cover_letter_generator import CoverLetterGenerator
from job_apply_ai.cv_modifier.cv_chat_editor import CVChatEditor
from job_apply_ai.cv_modifier.cv_content_store import (
    load_cv_content,
    normalize_store,
    save_cv_content,
    start_chat_session,
)
from job_apply_ai.cv_modifier.cv_generator import RAGCVGenerator
from job_apply_ai.cv_modifier.job_match_analyzer import analyze_jobs_with_threshold
from job_apply_ai.cv_modifier.pdf_builder import build_cover_letter_pdf
from job_apply_ai.cv_modifier.profile_importer import ProfileImporter
from job_apply_ai.dev_logging import dev_agent, dev_task
from job_apply_ai.job_status import DEFAULT_JOB_STATUS
from job_apply_ai.paths import get_cv_output_dir
from job_apply_ai.storage.job_repository import JobRepository
from job_apply_ai.storage.user_profile import (
    UserProfileRepository,
    get_default_cv_template_path,
    import_has_changes,
    merge_profiles,
    profile_to_form_fields,
    summarize_import_changes,
)
from job_apply_ai.utils.helpers import sanitize_filename

logger = logging.getLogger(__name__)

BATCH_ATS_FRIENDLY_PASSES = (
    {"label": "Analyze & accept all", "apply_all": True},
    {"label": "Analyze & accept all", "apply_all": True},
    {"label": "Analyze only", "apply_all": False},
)


class TaskProgress(Protocol):
    """Progress and lifecycle hooks used by queue and in-process runners."""

    task_id: str

    def update(
        self,
        *,
        status: str | None = None,
        step: str | None = None,
        message: str | None = None,
        percent: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None: ...

    def complete(self, result: dict[str, Any], *, message: str = "") -> None: ...

    def fail(self, error: str, *, result: dict | None = None) -> None: ...

    def checkpoint(self) -> None: ...

    def is_stop_requested(self) -> bool: ...


def cv_output_path(job: dict, cv_dir: str | None = None) -> tuple[str, str]:
    today_date = datetime.today().strftime("%Y-%m-%d")
    safe_company = sanitize_filename(job.get("company", "Company"))
    safe_title = sanitize_filename(job.get("title", "Role"))
    output_filename = f"CV_{today_date}_{safe_company}_{safe_title}.docx"
    root = cv_dir or get_cv_output_dir()
    return output_filename, os.path.join(root, output_filename)


def cover_letter_output_path(job: dict, cv_dir: str | None = None) -> tuple[str, str]:
    today_date = datetime.today().strftime("%Y-%m-%d")
    safe_company = sanitize_filename(job.get("company", "Company"))
    safe_title = sanitize_filename(job.get("title", "Role"))
    output_filename = f"CoverLetter_{today_date}_{safe_company}_{safe_title}.docx"
    root = cv_dir or get_cv_output_dir()
    return output_filename, os.path.join(root, output_filename)


def load_job_cv_store(cv_filename: str, cv_dir: str | None = None) -> dict | None:
    return load_cv_content(cv_dir or get_cv_output_dir(), cv_filename)


def save_job_cv_content(
    cv_filename: str,
    tailored_content: dict,
    cv_dir: str | None = None,
    **kwargs,
) -> None:
    save_cv_content(cv_dir or get_cv_output_dir(), cv_filename, tailored_content, **kwargs)


def job_has_sendable_cv(job: dict, cv_dir: str | None = None) -> bool:
    cv_filename = job.get("cv_filename", "")
    if not cv_filename:
        return False
    return os.path.exists(os.path.join(cv_dir or get_cv_output_dir(), cv_filename))


def jobs_with_cv(jobs: list[dict], cv_dir: str | None = None) -> list[dict]:
    return [job for job in jobs if job_has_sendable_cv(job, cv_dir)]


def clear_cv_preview_customization(store: dict, content: dict) -> None:
    store["tailored_content"] = content
    store["cv_preview_lines"] = []
    store["cv_preview_customized"] = False


def generate_and_save_cover_letter(
    job: dict,
    job_id: int | None,
    profile: dict,
    tailored_content: dict,
    cv_filename: str,
    *,
    job_repo: JobRepository | None = None,
    cv_dir: str | None = None,
    reset_cover_letter_chat: bool = False,
) -> tuple[str, str, dict]:
    repository = job_repo or JobRepository()
    root = cv_dir or get_cv_output_dir()
    cl_filename, cl_path = cover_letter_output_path(job, root)
    generator = CoverLetterGenerator()
    with dev_agent("CoverLetterGenerator", job_id=job_id):
        cl_content = generator.generate(job, profile, tailored_content)
    CoverLetterBuilder().build(cl_path, cl_content)
    build_cover_letter_pdf(cl_path, cl_content)

    if job_id:
        job["cover_letter_filename"] = cl_filename
        repository.update_job(job_id, job)

    store = normalize_store(load_job_cv_store(cv_filename, root) or {})
    if reset_cover_letter_chat:
        start_chat_session(store, "cover_letter")
    save_job_cv_content(
        cv_filename,
        store.get("tailored_content", tailored_content),
        cv_dir=root,
        chat_history=store.get("chat_history", []),
        cover_letter=cl_content,
        cover_letter_chat_history=store.get("cover_letter_chat_history", []),
        store=store,
    )
    return cl_filename, cl_path, cl_content


def generate_rag_cv(
    job: dict,
    profile: dict,
    *,
    cv_dir: str | None = None,
    reindex: bool = True,
    task_id: str | None = None,
    on_progress: Callable[[str, str, int], None] | None = None,
) -> dict:
    root = cv_dir or get_cv_output_dir()
    output_filename, output_path = cv_output_path(job, root)
    generator = RAGCVGenerator()
    job_id = job.get("id") if isinstance(job.get("id"), int) else None

    def progress(step: str, message: str, percent: int) -> None:
        if on_progress:
            on_progress(step, message, percent)

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
            on_progress=progress if on_progress else None,
        )
    result["output_filename"] = output_filename
    return result


def run_single_cv_workflow(
    progress: TaskProgress,
    *,
    profile: dict,
    job: dict,
    job_id: int,
    return_folder: str,
    return_search: str,
    return_from_manage: bool,
    return_sort: str = "",
    job_repo: JobRepository | None = None,
    cv_dir: str | None = None,
) -> None:
    repository = job_repo or JobRepository()
    root = cv_dir or get_cv_output_dir()

    def on_progress(step: str, message: str, percent: int) -> None:
        progress.update(status="running", step=step, message=message, percent=percent)

    with dev_task(progress.task_id, "cv_generation", job_id=job_id):
        result = generate_rag_cv(
            job,
            profile,
            cv_dir=root,
            task_id=progress.task_id,
            on_progress=on_progress,
        )
        output_path = result["output_path"]
        output_filename = result["output_filename"]
        tailored_content = result.get("tailored_content", {})

        matched_categories = {
            "Skills Matching Job Description": tailored_content.get("job_matched_skills", []),
            "Job Skills Not In CV": tailored_content.get("job_skills_not_in_cv", []),
            "Technical Skills": tailored_content.get(
                "technical_skills", tailored_content.get("key_skills", [])
            ),
            "Tools & Platforms": tailored_content.get("tools_platforms", []),
        }
        job["matched_categories"] = matched_categories
        job["cv_filename"] = output_filename
        repository.update_job(job_id, job)
        save_job_cv_content(output_filename, tailored_content, cv_dir=root, chat_history=[])

        try:
            cl_filename, _, cl_content = generate_and_save_cover_letter(
                job,
                job_id,
                profile,
                tailored_content,
                output_filename,
                job_repo=repository,
                cv_dir=root,
            )
        except Exception as cl_error:
            logger.error("Cover letter generation failed for job %s: %s", job_id, cl_error)
            cl_filename = ""
            cl_content = {}

        progress.complete(
            {
                "job": job,
                "cv_filename": output_filename,
                "cover_letter_filename": cl_filename,
                "cover_letter": cl_content,
                "output_path": output_path,
                "matched_categories": matched_categories,
                "tailored_content": tailored_content,
                "analysis": result.get("analysis", {}),
                "generation_meta": result.get("models", {}),
                "rag_chunk_count": result.get("chunk_count", 0),
                "return_folder": return_folder,
                "return_search": return_search,
                "return_sort": return_sort,
                "return_from_manage": return_from_manage,
            },
            message="CV generated successfully",
        )


def run_ats_friendly_workflow(
    progress: TaskProgress,
    *,
    job: dict,
    job_id: int,
    profile: dict,
    cv_filename: str,
    return_folder: str,
    return_search: str,
    return_from_manage: bool,
    return_sort: str = "",
    cv_dir: str | None = None,
) -> None:
    root = cv_dir or get_cv_output_dir()
    progress.update(
        status="running",
        step="loading_cv",
        message="Loading your CV content…",
        percent=10,
    )
    store = normalize_store(load_job_cv_store(cv_filename, root) or {})
    profile_name = profile.get("full_name", "")
    cv_content = resolve_effective_tailored_content(
        store.get("tailored_content", {}),
        profile_name,
        stored_lines=store.get("cv_preview_lines"),
        customized=bool(store.get("cv_preview_customized")),
    )
    if not cv_content:
        progress.fail("CV content not found. Regenerate the CV first.")
        return

    progress.update(
        status="running",
        step="analyzing_ats",
        message="Comparing your CV against ATS rules and the job description…",
        percent=35,
    )
    analyzer = ATSFriendlyAnalyzer()
    try:
        with dev_agent("ATSFriendlyAnalyzer", task_id=progress.task_id, job_id=job_id):
            analysis = analyzer.analyze(job=job, cv_content=cv_content, profile=profile)
    except Exception as exc:
        logger.error("ATS analysis failed for job %s: %s", job_id, exc)
        progress.fail(str(exc))
        return

    analysis["analyzed_at"] = datetime.utcnow().isoformat(timespec="seconds")
    store["ats_analysis"] = analysis
    progress.update(
        status="running",
        step="saving",
        message="Saving ATS report…",
        percent=90,
    )
    save_job_cv_content(cv_filename, cv_content, cv_dir=root, store=store)

    progress.complete(
        {
            "job": job,
            "job_id": job_id,
            "cv_filename": cv_filename,
            "ats_analysis": analysis,
            "return_folder": return_folder,
            "return_search": return_search,
            "return_sort": return_sort,
            "return_from_manage": return_from_manage,
        },
        message="ATS analysis complete",
    )


def run_ats_pass_for_job(
    *,
    job: dict,
    job_id: int,
    profile: dict,
    cv_filename: str,
    analyzer: ATSFriendlyAnalyzer,
    apply_all: bool,
    job_repo: JobRepository | None = None,
    cv_dir: str | None = None,
) -> dict[str, Any]:
    repository = job_repo or JobRepository()
    root = cv_dir or get_cv_output_dir()
    cv_path = os.path.join(root, cv_filename)
    store = normalize_store(load_job_cv_store(cv_filename, root) or {})
    profile_name = profile.get("full_name", "")
    cv_content = resolve_effective_tailored_content(
        store.get("tailored_content", {}),
        profile_name,
        stored_lines=store.get("cv_preview_lines"),
        customized=bool(store.get("cv_preview_customized")),
    )
    if not cv_content:
        return {"status": "skipped", "reason": "CV content not found"}

    with dev_agent("ATSFriendlyAnalyzer", job_id=job_id):
        analysis = analyzer.analyze(job=job, cv_content=cv_content, profile=profile)
    analysis["analyzed_at"] = datetime.utcnow().isoformat(timespec="seconds")

    result: dict[str, Any] = {
        "status": "ok",
        "analyzed": True,
        "applied": 0,
        "suggestion_count": len(analysis.get("suggestions", [])),
    }

    if apply_all:
        to_apply = pending_suggestions(analysis)
        if not to_apply:
            result["apply_skipped"] = True
        else:
            suggestion_ids = [item["id"] for item in to_apply]
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
                job["matched_categories"] = matched_categories
                repository.update_job(job_id, job)

                analysis = update_suggestions_status(analysis, suggestion_ids, status="applied")
                cv_content = updated_content
                clear_cv_preview_customization(store, updated_content)
                result["applied"] = len(suggestion_ids)
            except Exception as exc:
                logger.error("Batch ATS apply_all failed for job %s: %s", job_id, exc)
                try:
                    analysis = update_suggestions_status(
                        analysis,
                        suggestion_ids,
                        status="failed",
                        error=str(exc),
                    )
                except KeyError:
                    pass
                store["ats_analysis"] = analysis
                save_job_cv_content(cv_filename, cv_content, cv_dir=root, store=store)
                result["status"] = "apply_failed"
                result["error"] = str(exc)
                return result

    store["ats_analysis"] = analysis
    save_job_cv_content(cv_filename, cv_content, cv_dir=root, store=store)
    return result


def run_batch_ats_friendly_workflow(
    progress: TaskProgress,
    *,
    jobs: list[dict],
    profile: dict,
    return_folder: str,
    return_search: str,
    return_sort: str = "",
    job_repo: JobRepository | None = None,
    cv_dir: str | None = None,
) -> None:
    root = cv_dir or get_cv_output_dir()
    with dev_task(progress.task_id, "batch_ats_friendly"):
        jobs_with_cv_list = jobs_with_cv(jobs, root)
        if not jobs_with_cv_list:
            progress.fail("No jobs with generated CVs found in this folder.")
            return

        analyzer = ATSFriendlyAnalyzer()
        if not analyzer.llm.is_available():
            progress.fail(
                f"{analyzer.llm.provider_label} is not reachable. Check your LLM settings."
            )
            return

        try:
            analyzer.llm.validate_models()
        except Exception as exc:
            progress.fail(str(exc))
            return

        total_passes = len(BATCH_ATS_FRIENDLY_PASSES)
        total_steps = len(jobs_with_cv_list) * total_passes
        stats: dict[str, Any] = {
            "total_jobs": len(jobs_with_cv_list),
            "passes": total_passes,
            "analyzed": 0,
            "apply_passes": 0,
            "suggestions_applied": 0,
            "apply_skipped_no_suggestions": 0,
            "skipped": 0,
            "failed": 0,
            "failed_jobs": [],
        }

        step = 0
        for pass_index, pass_config in enumerate(BATCH_ATS_FRIENDLY_PASSES, start=1):
            apply_all = bool(pass_config["apply_all"])

            for job_index, job in enumerate(jobs_with_cv_list, start=1):
                try:
                    progress.checkpoint()
                except Exception:
                    stopped = progress.is_stop_requested()
                    if stats["analyzed"] == 0:
                        progress.fail("Batch ATS optimization stopped before any jobs were processed.")
                        return
                    progress.complete(
                        {
                            "stats": stats,
                            "return_folder": return_folder,
                            "return_search": return_search,
                            "return_sort": return_sort,
                            "stopped": stopped,
                        },
                        message=(
                            f'Batch ATS optimization stopped — processed {stats["analyzed"]} '
                            f'analysis pass(es) across {stats["total_jobs"]} job(s)'
                        ),
                    )
                    return

                step += 1
                job_id = job.get("id")
                title = job.get("title", "Untitled")
                cv_filename = job.get("cv_filename", "")
                percent = 5 + int((step / max(total_steps, 1)) * 90)
                action_label = "accepting suggestions" if apply_all else "analyzing"
                progress.update(
                    status="running",
                    step="batch_ats",
                    message=(
                        f"Pass {pass_index} of {total_passes} — job {job_index} of "
                        f"{len(jobs_with_cv_list)}: {title} — {action_label}…"
                    ),
                    percent=percent,
                    meta={
                        "current_pass": pass_index,
                        "total_passes": total_passes,
                        "current_index": job_index,
                        "total_jobs": len(jobs_with_cv_list),
                        "current_job_title": title,
                        "apply_all": apply_all,
                    },
                )

                if not job_id or not cv_filename:
                    stats["skipped"] += 1
                    continue

                try:
                    pass_result = run_ats_pass_for_job(
                        job=job,
                        job_id=job_id,
                        profile=profile,
                        cv_filename=cv_filename,
                        analyzer=analyzer,
                        apply_all=apply_all,
                        job_repo=job_repo,
                        cv_dir=root,
                    )
                except Exception as exc:
                    logger.error("Batch ATS pass failed for job %s: %s", job_id, exc)
                    stats["failed"] += 1
                    stats["failed_jobs"].append({"job_id": job_id, "title": title, "error": str(exc)})
                    continue

                if pass_result.get("status") == "skipped":
                    stats["skipped"] += 1
                    continue
                if pass_result.get("status") == "apply_failed":
                    stats["failed"] += 1
                    stats["failed_jobs"].append(
                        {
                            "job_id": job_id,
                            "title": title,
                            "error": pass_result.get("error", "Apply all failed"),
                        }
                    )
                    stats["analyzed"] += 1
                    continue

                stats["analyzed"] += 1
                if apply_all:
                    stats["apply_passes"] += 1
                    if pass_result.get("apply_skipped"):
                        stats["apply_skipped_no_suggestions"] += 1
                    else:
                        stats["suggestions_applied"] += pass_result.get("applied", 0)

        progress.complete(
            {
                "stats": stats,
                "return_folder": return_folder,
                "return_search": return_search,
                "return_sort": return_sort,
            },
            message=(
                f'Batch ATS optimization complete — {stats["analyzed"]} analysis pass(es) on '
                f'{stats["total_jobs"]} job(s), {stats["suggestions_applied"]} suggestion(s) applied'
            ),
        )


def run_batch_cv_workflow(
    progress: TaskProgress,
    *,
    profile: dict,
    jobs: list[dict],
    job_repo: JobRepository | None = None,
    cv_dir: str | None = None,
) -> None:
    repository = job_repo or JobRepository()
    root = cv_dir or get_cv_output_dir()
    generator = RAGCVGenerator()

    progress.update(
        status="running",
        step="validating_ollama",
        message=f"Checking {generator.llm.provider_label} and models…",
        percent=5,
    )
    if not generator.llm.is_available():
        raise RuntimeError(f"{generator.llm.provider_label} is not reachable.")

    generator.llm.validate_models()
    progress.update(status="running", step="indexing_cv", message="Indexing your profile with RAG…", percent=12)
    generator.prepare_profile_index(profile)

    successful_jobs = []
    failed_jobs = []
    generated_cvs = []
    total = len(jobs)

    for index, job in enumerate(jobs, start=1):
        try:
            progress.checkpoint()
        except Exception:
            if not generated_cvs:
                progress.fail("CV generation stopped before any CVs were created.")
                return
            progress.complete(
                {
                    "successful_jobs": successful_jobs,
                    "failed_jobs": failed_jobs,
                    "generated_cvs": generated_cvs,
                    "generated_count": len(generated_cvs),
                    "stopped": True,
                },
                message=f"CV generation stopped — created {len(generated_cvs)} of {total} CV(s)",
            )
            return

        job_id = job.get("id")
        title = job.get("title", "Job")
        base_percent = 15 + int(((index - 1) / max(total, 1)) * 80)
        progress.update(
            status="running",
            message=f"Generating CV {index} of {total}: {title}",
            percent=base_percent,
            meta={
                "current_index": index,
                "total_jobs": total,
                "current_job_title": title,
            },
        )

        try:
            output_filename, output_path = cv_output_path(job, root)
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
                    on_progress=lambda step, message, percent, bp=base_percent: progress.update(
                        step=step,
                        message=message,
                        percent=min(95, bp + percent // 10),
                        meta={
                            "current_index": index,
                            "total_jobs": total,
                            "current_job_title": title,
                        },
                    ),
                )
            generated_cvs.append(result["output_path"])
            successful_jobs.append(job)
            if job_id:
                tailored_content = result.get("tailored_content", {})
                job["matched_categories"] = {
                    "Skills Matching Job Description": tailored_content.get("job_matched_skills", []),
                    "Job Skills Not In CV": tailored_content.get("job_skills_not_in_cv", []),
                    "Technical Skills": tailored_content.get(
                        "technical_skills", tailored_content.get("key_skills", [])
                    ),
                    "Tools & Platforms": tailored_content.get("tools_platforms", []),
                }
                job["cv_filename"] = output_filename
                repository.update_job(job_id, job)
                save_job_cv_content(output_filename, tailored_content, cv_dir=root, chat_history=[])
                try:
                    generate_and_save_cover_letter(
                        job,
                        job_id,
                        profile,
                        tailored_content,
                        output_filename,
                        job_repo=repository,
                        cv_dir=root,
                    )
                except Exception as cl_error:
                    logger.error("Batch cover letter failed for %s: %s", title, cl_error)
        except Exception as job_error:
            logger.error("Batch CV failed for %s: %s", title, job_error)
            failed_jobs.append(job)

    if not generated_cvs:
        raise RuntimeError("No CVs were generated.")

    progress.complete(
        {
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "generated_cvs": generated_cvs,
            "generated_count": len(generated_cvs),
        },
        message=f"Generated {len(generated_cvs)} CV(s)",
    )


def run_job_match_analyze_workflow(
    progress: TaskProgress,
    *,
    jobs: list[dict],
    profile: dict,
    min_match_score: float,
    return_folder: str,
    return_search: str,
    return_sort: str = "",
    job_repo: JobRepository | None = None,
) -> None:
    repository = job_repo or JobRepository()
    with dev_task(progress.task_id, "job_match_analyze"):
        total = len(jobs)

        def on_progress(index: int, _total: int, job: dict) -> None:
            percent = 5 + int(((index + 1) / max(total, 1)) * 90)
            progress.update(
                status="running",
                step="analyzing",
                message=f"Analyzing job {index + 1} of {total}: {job.get('title', 'Untitled')}",
                percent=percent,
                meta={
                    "current_index": index + 1,
                    "total_jobs": total,
                    "current_job_title": job.get("title", ""),
                },
            )

        def should_continue() -> bool:
            try:
                progress.checkpoint()
                return True
            except Exception:
                return False

        progress.update(
            status="running",
            step="starting",
            message="Starting AI profile match analysis…",
            percent=5,
            meta={"total_jobs": total, "min_match_score": min_match_score},
        )
        result = analyze_jobs_with_threshold(
            jobs,
            profile,
            min_match_score,
            on_progress=on_progress,
            should_continue=should_continue,
        )

        for job, updated in zip(jobs, result["jobs"]):
            job_id = job.get("id")
            if not job_id:
                continue
            repository.update_job(
                job_id,
                {
                    "matched_categories": updated.get("matched_categories", {}),
                    "matched_skills": updated.get("matched_skills", job.get("matched_skills", [])),
                },
            )
            previous_status = job.get("workflow_status") or DEFAULT_JOB_STATUS
            new_status = updated.get("workflow_status") or previous_status
            if new_status != previous_status:
                repository.update_job_status(job_id, new_status)

        stats = result["stats"]
        stopped = progress.is_stop_requested()
        if stopped:
            analyzed = stats.get("analyzed", 0)
            if analyzed == 0:
                progress.fail("Profile match analysis stopped before any jobs were analyzed.")
                return
            progress.complete(
                {
                    "stats": stats,
                    "return_folder": return_folder,
                    "return_search": return_search,
                    "return_sort": return_sort,
                    "stopped": True,
                },
                message=f"Analysis stopped after {analyzed} job(s).",
            )
            return

        progress.complete(
            {
                "stats": stats,
                "return_folder": return_folder,
                "return_search": return_search,
                "return_sort": return_sort,
            },
            message="Profile match analysis complete",
        )


def run_profile_import_workflow(
    progress: TaskProgress,
    *,
    cv_path: str,
    current_profile: dict,
) -> None:
    try:
        with dev_task(progress.task_id, "profile_import"):
            progress.update(
                status="running",
                step="extracting",
                message="Reading CV document…",
                percent=10,
            )
            importer = ProfileImporter()
            with dev_agent("ProfileImporter"):
                extracted = importer.extract_from_docx(cv_path)

            progress.update(
                step="merging",
                message="Merging new details into your profile…",
                percent=70,
            )
            merged_profile, changes = merge_profiles(current_profile, extracted)
            summary_lines = summarize_import_changes(changes)

            progress.complete(
                {
                    "form": profile_to_form_fields(merged_profile),
                    "import_summary": summary_lines,
                    "has_changes": import_has_changes(changes),
                    "merged_profile": merged_profile,
                },
                message="Profile import complete",
            )
    finally:
        if os.path.exists(cv_path):
            os.remove(cv_path)
