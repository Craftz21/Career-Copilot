.PHONY: dev worker seed migrate test lint format check clean help audit audit-fast audit-pure

# ── Local development ──────────────────────────────────────────────────────────

dev:
	docker compose -f docker/docker-compose.yml up --build

dev-bg:
	docker compose -f docker/docker-compose.yml up --build -d

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f

# ── Database ───────────────────────────────────────────────────────────────────

migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(msg)"

migrate-down:
	alembic downgrade -1

seed:
	python scripts/seed_db.py

reset-db:
	alembic downgrade base && alembic upgrade head && python scripts/seed_db.py

# ── Worker (run locally without Docker) ───────────────────────────────────────

worker:
	celery -A src.worker worker --loglevel=info --concurrency=2

worker-beat:
	celery -A src.worker beat --loglevel=info

# ── API (run locally without Docker) ──────────────────────────────────────────

api:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# ── Testing ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/unit/ -v

# ── Audit suite ────────────────────────────────────────────────────────────────
# Requires a seeded PostgreSQL database:
#   export TEST_DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname
#   make seed   # seed that DB with roles, skills, and profiles
#   make audit

audit:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  CareerCopilot — Full Audit Suite (12 phases)"
	@echo "  DB tests require: export TEST_DATABASE_URL=... && make seed"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	pytest tests/evaluation/ -v --tb=short -p no:warnings --color=yes
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

audit-fast:
	@echo "Running audit — stop on first failure..."
	pytest tests/evaluation/ -v --tb=short -p no:warnings -x

audit-pure:
	@echo "Running pure (no-DB) audit tests only..."
	pytest tests/evaluation/ -v --tb=short -p no:warnings -m "not requires_db and not requires_seed"

# ── Code quality ───────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

check: lint
	ruff check src/ tests/ scripts/ --fix

# ── Utilities ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache .mypy_cache

health:
	curl -s http://localhost:8000/health | python -m json.tool

install:
	pip install -e ".[dev]"

help:
	@echo ""
	@echo "CareerCopilot — Available Commands"
	@echo "──────────────────────────────────"
	@echo "  make dev          Start full stack with Docker Compose"
	@echo "  make api          Run API server only (needs local DB)"
	@echo "  make worker       Run Celery worker only"
	@echo "  make seed         Seed database with skills and jobs"
	@echo "  make migrate      Apply pending database migrations"
	@echo "  make test         Run test suite"
	@echo "  make audit        Run full 12-phase evaluation + regression suite"
	@echo "  make audit-fast   Run audit, stop at first failure"
	@echo "  make audit-pure   Run only pure (no-DB) audit tests"
	@echo "  make lint         Run ruff linter"
	@echo "  make clean        Remove compiled Python files"
	@echo ""
