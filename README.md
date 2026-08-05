🚀 Career Copilot is an AI-powered career guidance tool that analyzes resumes, identifies skill gaps, and generates personalized learning roadmaps.

## ✨ Features

- 📄 Resume parsing & skill extraction
- 🤖 AI-driven skill gap analysis
- 🎯 Personalized learning roadmap
- 🗄️ MySQL integration for user data

## 🛠️ Tech Stack

- FastAPI
- MySQL
- LangChain + Groq LLaMA3
- HuggingFace Embeddings
- FAISS Vector Store

=======

# CareerCopilot

CareerCopilot is an AI-powered career intelligence platform that analyzes a resume against a target role or a pasted job description, scores role readiness, identifies skill gaps, and generates a structured learning roadmap.

The current implementation is a production-oriented v2 stack: a FastAPI backend, Celery workers, PostgreSQL with pgvector, Redis, Groq-powered roadmap generation, and a separate Next.js frontend.

## Features

- Resume upload for PDF, DOCX, and DOC files up to the configured upload limit.
- Target-role analysis against seeded canonical role profiles.
- Job-description analysis that extracts skills from the JD and compares them directly with the uploaded resume.
- Two-pass skill extraction using exact alias matching plus sentence-transformer embeddings.
- Readiness scoring with matched, missing, and bonus skills.
- Structured roadmap generation through Groq, with schema validation, retry, cache lookup, and template fallback.
- Interactive results dashboard with readiness charts, skill balance, roadmap timeline, and skill lists.
- Async processing with Celery progress polling.
- Session-based access with expiring result links and no user account requirement.
- Resume text privacy wipe after analysis metadata has been cached.

## Architecture

```text
Browser
  |
  | Next.js app (frontend/)
  | - /upload
  | - /processing/[sessionId]
  | - /results/[sessionId]
  v
FastAPI API (src/main.py)
  |
  | POST /v1/resume/upload
  | POST /v1/jd/analyze
  | GET  /v1/tasks/{session_id}
  | GET  /v1/results/{session_id}
  | GET  /health
  v
Celery worker (src/worker.py)
  |
  | parse resume
  | extract resume/JD skills
  | compute readiness and gaps
  | generate roadmap
  | schedule privacy wipe
  v
PostgreSQL + pgvector       Redis
skills, roles, sessions,    Celery broker/result backend
roadmaps, JD analyses
```

## Tech Stack

| Layer      | Technology                                                                           |
| ---------- | ------------------------------------------------------------------------------------ |
| Frontend   | Next.js 16, React 19, TypeScript, Tailwind CSS 4, TanStack Query, Recharts           |
| API        | FastAPI, Pydantic Settings, SQLAlchemy                                               |
| Worker     | Celery with Redis broker/result backend                                              |
| Database   | PostgreSQL 16 with pgvector                                                          |
| AI/ML      | sentence-transformers, Groq LLM API                                                  |
| Parsing    | PyMuPDF, python-docx                                                                 |
| Migrations | Alembic                                                                              |
| Quality    | pytest, ruff, mypy configuration                                                     |
| Deployment | Render blueprint for API/worker; frontend can deploy separately on Vercel or similar |

## Repository Layout

```text
projectAI/
  src/
    main.py                 FastAPI application entrypoint
    config.py               Centralized environment settings
    worker.py               Celery app and task discovery
    api/                    API routes and frontend redirects
    models/                 SQLAlchemy ORM models
    services/               Resume parsing, extraction, scoring, roadmap logic
    tasks/                  Celery task pipelines
    templates/              Legacy/server-rendered fallback templates
  frontend/                 Next.js frontend application
  data/                     Seed CSVs for skills, roles, jobs, resources
  migrations/               Alembic migrations
  prompts/                  Roadmap prompt templates
  scripts/                  Seeding, benchmark, corpus validation utilities
  tests/                    Unit, API, corpus, benchmark, and evaluation tests
  docker/                   Dockerfile and local docker-compose stack
  render.yaml               Render API and worker blueprint
```

`src/main.py` is the current backend entrypoint. The older `api/app.py` file is a legacy prototype and is not used by the v2 application.

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker and Docker Compose
- `make` for the documented shortcuts
- A Groq API key for LLM roadmap generation

PostgreSQL and Redis can run locally through Docker Compose. The application expects PostgreSQL with the `vector` extension enabled; the bundled compose file uses `pgvector/pgvector:pg16`.

## Environment Variables

Copy the sample environment file and edit it for your machine:

```bash
cp .env.example .env
```

For local Docker development, these values are the usual defaults:

```env
APP_ENV=development
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/career_copilot
REDIS_URL=redis://localhost:6379
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
MAX_UPLOAD_SIZE_MB=5
SESSION_TTL_DAYS=7
FRONTEND_BASE_URL=http://127.0.0.1:3000
```

The frontend reads `NEXT_PUBLIC_API_URL`; for local development create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Local Development

### 1. Start backend services

From the repository root:

```bash
make dev-bg
```

This starts PostgreSQL, Redis, the FastAPI API, and a Celery worker through `docker/docker-compose.yml`.

### 2. Apply migrations and seed data

Run migrations and seed the skill/role/resource data inside the API container:

```bash
docker compose -f docker/docker-compose.yml exec api alembic upgrade head
docker compose -f docker/docker-compose.yml exec api python scripts/seed_db.py
```

Seeding generates embeddings for skills and roles, so the first run can take a few minutes.

### 3. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

- Frontend: http://127.0.0.1:3000
- Backend health: http://127.0.0.1:8000/health
- API docs in development: http://127.0.0.1:8000/api/docs

## Common Commands

| Command        | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `make dev`     | Start the Docker Compose backend stack in the foreground        |
| `make dev-bg`  | Start the Docker Compose backend stack in the background        |
| `make down`    | Stop the Docker Compose stack                                   |
| `make logs`    | Follow backend container logs                                   |
| `make api`     | Run FastAPI locally without Docker-managed API container        |
| `make worker`  | Run a local Celery worker                                       |
| `make migrate` | Apply Alembic migrations using the local Python environment     |
| `make seed`    | Seed the configured database using the local Python environment |
| `make test`    | Run the pytest suite                                            |
| `make audit`   | Run the evaluation/regression suite                             |
| `make lint`    | Run ruff checks                                                 |

Frontend commands are run from `frontend/`:

```bash
npm run dev
npm run build
npm run start
npm run lint
```

## API Overview

| Method | Path                       | Purpose                                                                             |
| ------ | -------------------------- | ----------------------------------------------------------------------------------- |
| `GET`  | `/health`                  | Check database and Redis availability                                               |
| `GET`  | `/v1/roles`                | Return role suggestions for the upload UI                                           |
| `POST` | `/v1/resume/upload`        | Upload a resume with `target_role` and `duration`; returns `202` and a `session_id` |
| `POST` | `/v1/jd/analyze`           | Upload a resume plus JD text; returns `202` and a `session_id`                      |
| `GET`  | `/v1/tasks/{session_id}`   | Poll async task status and progress                                                 |
| `GET`  | `/v1/results/{session_id}` | Fetch completed analysis, roadmap, recruiter summary, evidence, and role-fit data   |

Page routes such as `/`, `/processing/{session_id}`, and `/results/{session_id}` redirect to the configured frontend base URL.

## Analysis Flow

1. The user uploads a resume and either selects a target role or submits job-description text.
2. The API validates file type, size, duration, and request rate limits.
3. A session and task record are created, then the relevant Celery task is queued.
4. The worker parses resume text and sections from PDF/DOCX content.
5. Skills are extracted using alias matching, section weighting, and embedding similarity.
6. Role mode normalizes the target role to a seeded role profile; JD mode builds a temporary skill profile from the pasted job description.
7. Gap analysis computes readiness, matched skills, missing skills, bonus skills, category breakdowns, and score contributors.
8. Groq generates a structured roadmap from the gaps. If LLM output fails validation twice, a template roadmap is built from seeded learning resources.
9. Results are stored and exposed through `/v1/results/{session_id}`.
10. A privacy task wipes persisted raw resume text after the configured delay while preserving metadata needed by the results page.

## Privacy and Security Notes

- The app uses session IDs as shareable result links; there is no user authentication in the current implementation.
- Uploaded file bytes are processed asynchronously and are not saved as files by the application.
- Parsed resume text is persisted only long enough to support analysis, then wiped by `tasks.wipe_resume_text`.
- Results expire based on `SESSION_TTL_DAYS`.
- Upload rate limiting is in-process and suitable for a small single-instance deployment; use a shared limiter for multi-instance production.
- Keep `.env` out of version control and rotate any exposed API keys.

## Testing

Run the main test suite:

```bash
make test
```

Run the evaluation suite:

```bash
make audit
```

Some evaluation tests require a reachable PostgreSQL database with migrations applied and seed data loaded. Pure tests can be run without database-backed cases:

```bash
make audit-pure
```

For frontend quality checks:

```bash
cd frontend
npm run lint
```

## Deployment

### Backend on Render

`render.yaml` defines two services:

- `careerpilot-api`: FastAPI web service
- `careerpilot-worker`: Celery worker

Required production environment variables:

```env
APP_ENV=production
APP_SECRET_KEY=generated-or-secure-random-value
DATABASE_URL=postgresql+psycopg2://...
REDIS_URL=rediss://...
GROQ_API_KEY=...
LLM_MODEL=llama-3.3-70b-versatile
FRONTEND_BASE_URL=https://your-frontend-domain.example
```

The API startup script runs `alembic upgrade head` before starting Uvicorn. After the first deployment, seed the database once:

```bash
python scripts/seed_db.py
```

### Frontend

Deploy `frontend/` as a standalone Next.js app. Set:

```env
NEXT_PUBLIC_API_URL=https://your-api-domain.example
```

The backend should also know the deployed frontend URL through `FRONTEND_BASE_URL` so server page routes redirect correctly.

## Operational Notes

- The worker pre-warms sentence-transformer models on startup to reduce first-task latency.
- Celery uses late acknowledgements and rejects tasks on worker loss so interrupted work can be retried.
- Roadmap generation caches by role, duration, and top missing skills.
- `APP_ENV=development` enables FastAPI docs at `/api/docs` and creates tables on startup for convenience; use Alembic migrations in production.

## License

No license file is currently included. Add a license before accepting external contributions or reuse.
