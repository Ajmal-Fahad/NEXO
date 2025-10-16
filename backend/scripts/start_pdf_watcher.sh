#!/usr/bin/env bash
# scripts/start_pdf_watcher.sh
# Usage: from backend folder -> ./scripts/start_pdf_watcher.sh
# This script activates venv, loads .env (if present), ensures OPENAI_API_KEY is set,
# and only then starts the pdf watcher to avoid processing before the key is available.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Activate venv (adjust path if your venv lives elsewhere)
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "ERROR: venv not found at .venv/bin/activate. Create or adjust path." >&2
  exit 1
fi

# Load .env if present (do not print; keep secure)
if [ -f ".env" ]; then
  # read each non-empty non-comment line KEY=VALUE and export it
  set -o allexport
  # shellcheck disable=SC1090
  source .env
  set +o allexport
fi

# Prefer OPENAI_API_KEY from environment now (after venv activation / .env)
# If not present, give the user a friendly retry/exit behavior
MAX_RETRIES=6
SLEEP_SEC=5
i=0
while [ -z "${OPENAI_API_KEY:-}" ]; do
  i=$((i+1))
  if [ "$i" -gt "$MAX_RETRIES" ]; then
    echo "ERROR: OPENAI_API_KEY not set after $MAX_RETRIES attempts. Please export OPENAI_API_KEY or create .env with it." >&2
    exit 2
  fi
  echo "OPENAI_API_KEY not set. Waiting for it to be set... (attempt $i/$MAX_RETRIES). Sleeping ${SLEEP_SEC}s"
  sleep "$SLEEP_SEC"
done

echo "OPENAI_API_KEY detected. Starting pdf_watcher..."
# run the watcher (stdout/stderr will show logs)
python3 services/pdf_watcher.py
