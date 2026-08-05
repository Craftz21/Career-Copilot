#!/usr/bin/env bash
# API startup script for Render.
# Runs migrations (idempotent) then starts uvicorn.
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting API server..."
exec uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8000}"
