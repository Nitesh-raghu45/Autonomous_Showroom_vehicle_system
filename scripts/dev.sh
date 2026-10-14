#!/usr/bin/env bash
# Runs the API on :8000 and the Vite dev server on :5173, together.
# Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d frontend/node_modules ]; then
  echo "installing frontend deps..."
  (cd frontend && npm install)
fi

uvicorn app.main:app --reload --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

(cd frontend && npm run dev)
