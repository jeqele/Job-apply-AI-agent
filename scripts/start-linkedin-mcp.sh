#!/usr/bin/env bash
# Start the LinkedIn MCP HTTP sidecar for HermesHire job search.
# Prerequisites: uv installed — https://docs.astral.sh/uv/getting-started/installation/
# One-time login: uvx mcp-server-linkedin@latest --login

set -euo pipefail

HOST="${LINKEDIN_MCP_HOST:-127.0.0.1}"
PORT="${LINKEDIN_MCP_PORT:-8080}"
PATH_SUFFIX="${LINKEDIN_MCP_PATH:-/mcp}"

echo "Starting LinkedIn MCP sidecar at http://${HOST}:${PORT}${PATH_SUFFIX}"
echo "Press Ctrl+C to stop."

exec uvx mcp-server-linkedin@latest \
  --transport streamable-http \
  --host "$HOST" \
  --port "$PORT" \
  --path "$PATH_SUFFIX"
