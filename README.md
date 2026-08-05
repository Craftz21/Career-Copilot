# 🚀 CareerCopilot

> AI-powered career intelligence platform that analyzes resumes and job descriptions, identifies skill gaps, and generates personalized learning roadmaps.

CareerCopilot helps job seekers understand how well they match a target role, discover missing skills, and receive an actionable week-by-week learning plan using semantic search and LLM-powered recommendations.

---

## ✨ Features

- 📄 Resume upload (PDF, DOCX)
- 💼 Target Role Analysis
- 📑 Job Description Analysis
- 🧠 Semantic Skill Extraction
- 📊 AI Readiness Score
- 🎯 Skill Gap Identification
- 🗺️ Personalized Learning Roadmaps
- 📈 Interactive Analytics Dashboard
- ⚡ Asynchronous Resume Processing
- 🔗 Shareable Results

---

# Demo

> **Live Demo:** *Coming Soon*

### Landing Page
*(Add screenshot here)*

### Resume Upload
*(Add screenshot here)*

### Results Dashboard
*(Add screenshot here)*

### Learning Roadmap
*(Add screenshot here)*

---

# Architecture

```
                    Next.js Frontend
                           │
                           │
                  REST API Requests
                           │
                           ▼
                    FastAPI Backend
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
 Resume Parsing     Skill Extraction    Job Analysis
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                  Gap Analysis Engine
                           │
                           ▼
                 Groq LLM Roadmap Generator
                           │
                           ▼
                  PostgreSQL + pgvector
                           │
                           ▼
                     Celery + Redis
```

---

# Tech Stack

| Category | Technology |
|-----------|------------|
| Frontend | Next.js, React, TypeScript, Tailwind CSS |
| State Management | TanStack Query |
| Charts | Recharts |
| Backend | FastAPI |
| Async Processing | Celery |
| Database | PostgreSQL + pgvector |
| Cache / Broker | Redis |
| Embeddings | sentence-transformers (MiniLM-L6-v2) |
| LLM | Groq (Llama 3.3 70B) |
| ORM | SQLAlchemy |
| Migrations | Alembic |
| Testing | Pytest |
| Deployment | Docker, Render |

---

# How It Works

1. Upload a resume.
2. Choose a target role or provide a job description.
3. Resume skills are extracted using semantic search.
4. Skills are matched against role requirements.
5. A readiness score is calculated.
6. Missing skills are identified.
7. An AI-generated learning roadmap is created.
8. Interactive analytics are displayed in the dashboard.

---

# Key Features

## Semantic Skill Extraction

- Exact skill alias matching
- Embedding-based semantic similarity
- Section-aware resume parsing

---

## Role Readiness Analysis

CareerCopilot computes a readiness score by comparing extracted skills against curated role profiles, highlighting:

- Matched Skills
- Missing Skills
- Bonus Skills
- Category-wise breakdown
- Overall readiness percentage

---

## Personalized Roadmap

The platform generates structured learning plans including:

- Weekly milestones
- Skills to learn
- Estimated effort
- Curated learning resources

---

## Interactive Dashboard

The dashboard provides:

- Readiness Gauge
- Radar Chart
- Skill Distribution
- Timeline Roadmap
- Skill Breakdown

---

# Project Structure

```
CareerCopilot/
│
├── frontend/          # Next.js application
├── src/               # FastAPI backend
├── scripts/           # Utilities
├── tests/             # Test suite
├── migrations/        # Alembic migrations
├── docker/            # Docker configuration
├── prompts/           # LLM prompts
├── data/              # Seed datasets
│
├── README.md
├── pyproject.toml
├── Makefile
└── render.yaml
```

---

# Local Development

## Clone

```bash
git clone https://github.com/<username>/CareerCopilot.git
cd CareerCopilot
```

---

## Backend

```bash
cp .env.example .env

make up
make migrate
make seed

make dev
```

Run the worker in another terminal:

```bash
make worker
```

Backend:

```
http://localhost:8000
```

API Docs:

```
http://localhost:8000/api/docs
```

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

Frontend:

```
http://localhost:3000
```

---

# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/v1/resume/upload` | Upload resume |
| POST | `/v1/jd/analyze` | Analyze resume against job description |
| GET | `/v1/tasks/{session_id}` | Processing status |
| GET | `/v1/results/{session_id}` | Analysis results |
| GET | `/v1/roles` | Available role suggestions |
| GET | `/health` | Health check |

---

# Testing

Run backend tests:

```bash
make test
```

Frontend:

```bash
cd frontend

npm run build
npm run lint
```

---

# Deployment

The application is designed for deployment using:

- Render (Backend)
- PostgreSQL + pgvector
- Redis
- Vercel (Frontend)

---

# Future Improvements

- User authentication
- Resume version history
- ATS score analysis
- Interview preparation assistant
- Company-specific role recommendations
- Learning progress tracking
- Multi-language resume support

---

# Why CareerCopilot?

Unlike traditional resume checkers, CareerCopilot combines semantic search, vector embeddings, and large language models to provide intelligent career guidance rather than simple keyword matching.

---

# License

This project is licensed under the MIT License.

---

## ⭐ If you found this project useful, consider giving it a star!
