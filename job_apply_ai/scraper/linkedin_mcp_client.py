"""HTTP client for the local linkedin-mcp-server sidecar."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MCP_URL = "http://127.0.0.1:8080/mcp"


class LinkedInMcpError(RuntimeError):
    """Raised when the LinkedIn MCP sidecar is unavailable or a tool fails."""


def mcp_url() -> str:
    return (os.environ.get("LINKEDIN_MCP_URL") or DEFAULT_MCP_URL).strip()


def mcp_enabled() -> bool:
    raw = os.environ.get("LINKEDIN_MCP_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _http_timeout() -> float:
    return float(os.environ.get("LINKEDIN_MCP_HTTP_TIMEOUT", "60"))


def _sse_timeout() -> float:
    return float(os.environ.get("LINKEDIN_MCP_SSE_TIMEOUT", "300"))


def _parse_tool_result(result: Any) -> dict[str, Any]:
    if getattr(result, "isError", False):
        message = ""
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                message = text
                break
        raise LinkedInMcpError(message or "LinkedIn MCP tool failed")

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
        if isinstance(payload, dict):
            return payload
    return {}


async def _call_tool_async(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        mcp_url(),
        timeout=timedelta(seconds=_http_timeout()),
        sse_read_timeout=timedelta(seconds=_sse_timeout()),
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return _parse_tool_result(result)


def call_linkedin_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a LinkedIn MCP tool via the local HTTP sidecar."""
    if not mcp_enabled():
        raise LinkedInMcpError("LinkedIn MCP is disabled (LINKEDIN_MCP_ENABLED=false).")
    try:
        return asyncio.run(_call_tool_async(tool_name, arguments))
    except LinkedInMcpError:
        raise
    except Exception as exc:
        raise LinkedInMcpError(
            f"LinkedIn MCP unreachable at {mcp_url()}. "
            "Start the sidecar: scripts/start-linkedin-mcp.ps1 "
            f"(or uvx mcp-server-linkedin@latest --transport streamable-http). ({exc})"
        ) from exc


def check_linkedin_mcp_health() -> bool:
    """Return True when the MCP sidecar responds to initialize/list_tools."""
    if not mcp_enabled():
        return False
    try:
        asyncio.run(_health_check_async())
        return True
    except Exception as exc:
        logger.debug("LinkedIn MCP health check failed: %s", exc)
        return False


async def _health_check_async() -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        mcp_url(),
        timeout=timedelta(seconds=_http_timeout()),
        sse_read_timeout=timedelta(seconds=30),
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await session.list_tools()
