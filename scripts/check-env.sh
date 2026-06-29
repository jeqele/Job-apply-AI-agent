#!/usr/bin/env bash
# Quick environment check for HermesHire workers (web, batch-worker, ai-worker).
# Verifies SQLite database path, disk space, and optional LLM providers.
#
# Usage:
#   ./scripts/check-env.sh
#   ./scripts/check-env.sh --fix   # create DB parent directory if missing

set -euo pipefail

FIX=0
if [[ "${1:-}" == "--fix" ]]; then
  FIX=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

PASS=0
WARN=0
FAIL=0

pass() { echo "  OK   $*"; PASS=$((PASS + 1)); }
warn() { echo "  WARN $*"; WARN=$((WARN + 1)); }
fail() { echo "  FAIL $*"; FAIL=$((FAIL + 1)); }

resolve_db_path() {
  if [[ -n "${JOB_APPLY_AI_DB:-}" ]]; then
  python3 - <<'PY' "${JOB_APPLY_AI_DB}"
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
  else
    echo "$PROJECT_ROOT/job_apply_ai/outputs/data/jobs.db"
  fi
}

echo "HermesHire environment check"
echo "Project root: $PROJECT_ROOT"
echo

# --- Disk space ---
echo "[Disk]"
for target in "$PROJECT_ROOT" "$(dirname "$(resolve_db_path)")"; do
  if [[ -d "$target" ]]; then
    free_mb="$(df -Pm "$target" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -n "$free_mb" ]]; then
      if [[ "$free_mb" -lt 64 ]]; then
        fail "$target — only ${free_mb} MB free (SQLite needs journal space)"
      elif [[ "$free_mb" -lt 256 ]]; then
        warn "$target — ${free_mb} MB free (low; consider freeing space)"
      else
        pass "$target — ${free_mb} MB free"
      fi
    else
      warn "Could not read disk usage for $target"
    fi
  else
    warn "Path does not exist yet: $target"
  fi
done
echo

# --- SQLite database ---
echo "[SQLite]"
DB_PATH="$(resolve_db_path)"
DB_DIR="$(dirname "$DB_PATH")"
echo "  DB path: $DB_PATH"

if [[ ! -d "$DB_DIR" ]]; then
  if [[ "$FIX" -eq 1 ]]; then
    mkdir -p "$DB_DIR"
    pass "Created database directory: $DB_DIR"
  else
    fail "Database directory missing: $DB_DIR (run with --fix to create)"
  fi
elif [[ ! -w "$DB_DIR" ]]; then
  fail "Database directory is not writable: $DB_DIR"
else
  pass "Database directory exists and is writable: $DB_DIR"
fi

if [[ -f "$DB_PATH" ]]; then
  if [[ ! -r "$DB_PATH" || ! -w "$DB_PATH" ]]; then
    fail "Database file exists but is not readable/writable: $DB_PATH"
  else
    size_kb="$(du -k "$DB_PATH" | awk '{print $1}')"
    pass "Database file exists (${size_kb} KB)"
  fi
else
  warn "Database file does not exist yet (will be created on first run)"
fi

if python3 - <<PY
import os
import sqlite3
import sys

path = os.path.abspath(${DB_PATH@Q})
parent = os.path.dirname(path) or "."
os.makedirs(parent, exist_ok=True)
conn = sqlite3.connect(path, timeout=5)
conn.execute("SELECT 1")
conn.close()
PY
then
  pass "SQLite open + query succeeded"
else
  fail "SQLite could not open or query database (see error above)"
fi
echo

# --- FreeLLMAPI ---
echo "[FreeLLMAPI]"
FREELLMAPI_BASE_URL="${FREELLMAPI_BASE_URL:-http://localhost:3001/v1}"
FREELLMAPI_BASE_URL="${FREELLMAPI_BASE_URL%/}"
echo "  Base URL: $FREELLMAPI_BASE_URL"

if [[ -z "${FREELLMAPI_API_KEY:-}" ]]; then
  warn "FREELLMAPI_API_KEY is not set (skip reachability test unless configured in Settings)"
else
  http_code="$(curl -sS -o /tmp/hermeshire-freellmapi-models.json -w "%{http_code}" \
    --connect-timeout 5 --max-time 15 \
    -H "Authorization: Bearer ${FREELLMAPI_API_KEY}" \
    "${FREELLMAPI_BASE_URL}/models" 2>/dev/null || echo "000")"

  if [[ "$http_code" == "200" ]]; then
    model_count="$(python3 - <<'PY' /tmp/hermeshire-freellmapi-models.json
import json, sys
try:
    data = json.load(open(sys.argv[1]))
    print(len(data.get("data") or []))
except Exception:
    print(0)
PY
)"
    if [[ "$model_count" -gt 0 ]]; then
      pass "Reachable — $model_count model(s) listed"
    else
      warn "Reachable but returned no models (check API key and server logs)"
    fi
  elif [[ "$http_code" == "401" ]]; then
    fail "Reachable but API key rejected (401)"
  elif [[ "$http_code" == "000" ]]; then
    fail "Not reachable at $FREELLMAPI_BASE_URL (is FreeLLMAPI running?)"
  else
    fail "Unexpected HTTP $http_code from $FREELLMAPI_BASE_URL/models"
  fi
  rm -f /tmp/hermeshire-freellmapi-models.json
fi
echo

# --- Ollama (optional) ---
echo "[Ollama]"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL%/}"
echo "  Base URL: $OLLAMA_BASE_URL"

ollama_code="$(curl -sS -o /dev/null -w "%{http_code}" \
  --connect-timeout 3 --max-time 10 \
  "${OLLAMA_BASE_URL}/api/tags" 2>/dev/null || echo "000")"

if [[ "$ollama_code" == "200" ]]; then
  pass "Reachable"
elif [[ "$ollama_code" == "000" ]]; then
  warn "Not reachable (fine if you use Alibaba or FreeLLMAPI instead)"
else
  warn "Unexpected HTTP $ollama_code from Ollama"
fi
echo

# --- Summary ---
echo "Summary: $PASS passed, $WARN warnings, $FAIL failed"
if [[ "$FAIL" -gt 0 ]]; then
  echo
  echo "Fix tips:"
  echo "  - Set JOB_APPLY_AI_DB to an absolute path shared by web and all workers"
  echo "  - Ensure the database parent directory exists and has free disk space"
  echo "  - Start FreeLLMAPI or switch LLM provider in Settings if CV tasks fail"
  exit 1
fi

if [[ "$WARN" -gt 0 ]]; then
  exit 2
fi

exit 0
