"""Developer-mode logging with agent/task context and LLM conversation capture."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Iterator

from job_apply_ai.storage.dev_log import DevLogRepository

_agent: ContextVar[str] = ContextVar("dev_agent", default="")
_task_id: ContextVar[str] = ContextVar("dev_task_id", default="")
_job_id: ContextVar[int | None] = ContextVar("dev_job_id", default=None)
_endpoint: ContextVar[str] = ContextVar("dev_endpoint", default="")
_operation: ContextVar[str] = ContextVar("dev_operation", default="")
_chat_history: ContextVar[list[dict[str, str]] | None] = ContextVar("dev_chat_history", default=None)
_extra_context: ContextVar[dict[str, Any] | None] = ContextVar("dev_extra_context", default=None)
_dev_mode_cache: tuple[float, bool] | None = None
_CACHE_TTL_SECONDS = 2.0


def is_dev_mode() -> bool:
    """Return whether developer logging is enabled (cached briefly)."""
    global _dev_mode_cache
    now = time.monotonic()
    if _dev_mode_cache and (now - _dev_mode_cache[0]) < _CACHE_TTL_SECONDS:
        return _dev_mode_cache[1]
    try:
        from job_apply_ai.storage.app_settings import AppSettingsRepository

        enabled = bool(AppSettingsRepository().get_dev_mode())
    except Exception:
        enabled = False
    _dev_mode_cache = (now, enabled)
    return enabled


def invalidate_dev_mode_cache() -> None:
    global _dev_mode_cache
    _dev_mode_cache = None


def get_dev_context() -> dict[str, Any]:
    return {
        "agent": _agent.get(),
        "task_id": _task_id.get(),
        "job_id": _job_id.get(),
        "endpoint": _endpoint.get(),
        "operation": _operation.get(),
        "chat_history": deepcopy(_chat_history.get()) if _chat_history.get() else [],
        "extra_context": deepcopy(_extra_context.get()) if _extra_context.get() else {},
    }


def dev_log(
    category: str,
    event: str,
    message: str = "",
    *,
    data: dict[str, Any] | None = None,
    agent: str | None = None,
    task_id: str | None = None,
    job_id: int | None = None,
) -> int | None:
    """Write a developer log entry when dev mode is enabled."""
    if not is_dev_mode():
        return None
    return DevLogRepository().add_log(
        category=category,
        event=event,
        message=message,
        agent=agent if agent is not None else _agent.get(),
        data=data,
        task_id=task_id if task_id is not None else _task_id.get(),
        job_id=job_id if job_id is not None else _job_id.get(),
    )


def log_llm_conversation(
    *,
    call_type: str,
    provider: str,
    model: str,
    system: str | None,
    prompt: str,
    raw_response: str | None = None,
    parsed_response: dict[str, Any] | None = None,
    temperature: float | None = None,
    schema: dict[str, Any] | None = None,
    json_format: bool = False,
    attempt: int = 1,
    max_attempts: int = 1,
) -> int | None:
    """Log a full AI conversation turn with structured message history."""
    if not is_dev_mode():
        return None

    ctx = get_dev_context()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    for msg in ctx.get("chat_history") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "")
        if role and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt})

    if raw_response is not None:
        messages.append({"role": "assistant", "content": raw_response})

    data: dict[str, Any] = {
        "call_type": call_type,
        "provider": provider,
        "model": model,
        "endpoint": ctx.get("endpoint") or "",
        "operation": ctx.get("operation") or "",
        "messages": messages,
        "request": {
            "system": system or "",
            "prompt": prompt,
            "temperature": temperature,
            "json_format": json_format,
            "schema": schema,
        },
        "extra_context": ctx.get("extra_context") or {},
    }
    if raw_response is not None:
        data["response"] = {"raw": raw_response}
        if parsed_response is not None:
            data["response"]["parsed"] = parsed_response
    if max_attempts > 1:
        data["attempt"] = attempt
        data["max_attempts"] = max_attempts

    summary = f"{provider} → {model}"
    if ctx.get("operation"):
        summary = f"{ctx['operation']}: {summary}"
    if attempt > 1:
        summary = f"{summary} (attempt {attempt}/{max_attempts})"

    return dev_log(
        "llm",
        "conversation",
        summary,
        data=data,
        agent=ctx.get("agent") or None,
        task_id=ctx.get("task_id") or None,
        job_id=ctx.get("job_id"),
    )


@contextmanager
def dev_llm_context(
    *,
    endpoint: str = "",
    operation: str = "",
    chat_history: list[dict[str, str]] | None = None,
    context: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Attach endpoint, operation, chat history, and extra context to LLM logs."""
    tokens: list[tuple[str, Any]] = []
    if endpoint:
        tokens.append(("endpoint", _endpoint.set(endpoint)))
    if operation:
        tokens.append(("operation", _operation.set(operation)))
    if chat_history is not None:
        tokens.append(("chat_history", _chat_history.set(deepcopy(chat_history))))
    if context is not None:
        merged = deepcopy(_extra_context.get() or {})
        merged.update(context)
        tokens.append(("extra_context", _extra_context.set(merged)))
    try:
        yield
    finally:
        for name, token in reversed(tokens):
            if name == "endpoint":
                _endpoint.reset(token)
            elif name == "operation":
                _operation.reset(token)
            elif name == "chat_history":
                _chat_history.reset(token)
            elif name == "extra_context":
                _extra_context.reset(token)


@contextmanager
def dev_agent(
    agent_name: str,
    *,
    task_id: str | None = None,
    job_id: int | None = None,
    context: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Set the active agent for nested LLM and agent logs."""
    tokens: list[Any] = []
    tokens.append(_agent.set(agent_name))
    if task_id is not None:
        tokens.append(("task_id", _task_id.set(task_id)))
    if job_id is not None:
        tokens.append(("job_id", _job_id.set(job_id)))
    if context is not None:
        merged = deepcopy(_extra_context.get() or {})
        merged.update(context)
        tokens.append(("extra_context", _extra_context.set(merged)))
    dev_log("agent", "agent_start", f"{agent_name} started", data=context or None)
    try:
        yield
    finally:
        dev_log("agent", "agent_end", f"{agent_name} finished")
        for token in reversed(tokens):
            if isinstance(token, tuple):
                name, tok = token
                if name == "task_id":
                    _task_id.reset(tok)
                elif name == "job_id":
                    _job_id.reset(tok)
                else:
                    _extra_context.reset(tok)
            else:
                _agent.reset(token)


@contextmanager
def dev_task(
    task_id: str,
    task_type: str = "",
    *,
    job_id: int | None = None,
) -> Iterator[None]:
    """Set task context for logs emitted during a background job."""
    token_task = _task_id.set(task_id)
    token_job = _job_id.set(job_id) if job_id is not None else None
    dev_log(
        "task",
        "task_start",
        task_type or "background_task",
        data={"task_type": task_type, "task_id": task_id},
        task_id=task_id,
        job_id=job_id,
    )
    try:
        yield
    finally:
        if token_job is not None:
            _job_id.reset(token_job)
        _task_id.reset(token_task)
