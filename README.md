# AI Software Company

**Turn a plain-English sentence into a fully working web application - automatically.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Sandboxed-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The Idea

You type one sentence:

> _"A recipe sharing app with user profiles, ratings, and comments"_

The system **plans, codes, tests, and deploys** the entire app for you - running 100% on your own machine, using free-tier AI models.

---

## What You Get

| Step | What Happens |
|------|--------------|
| 1 | **7 AI agents** plan and generate the code |
| 2 | **Docker sandbox** tests the app in isolation |
| 3 | **Human approval** gate - you decide |
| 4 | **Live app** at `http://127.0.0.1:8001` - ready to use |

---

## The 7 Agents

| Agent | Role |
|-------|------|
| **PM** | Reads your requirement, produces 3-5 features |
| **Architect** | Designs file structure, DB schema, API routes |
| **Designer** | Picks colors, fonts, responsive layout |
| **Coder** | Writes every file (RAG-augmented) |
| **Tester** | Runs security + syntax + HTTP tests in Docker |
| **Improver** | Audits performance with Lighthouse |
| **Deploy** | Pushes to GitHub and serves the app locally |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.12 |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML5 + Tailwind CSS + Jinja2 |
| **Orchestration** | LangGraph (multi-agent workflow) |
| **AI Models** | Claude Sonnet 4.6 (primary) + Groq / Gemini / OpenRouter (fallbacks) |
| **Vector DB** | ChromaDB + all-MiniLM-L6-v2 |
| **Sandbox** | Docker (isolated testing) |
| **Queue** | ARQ + Redis |
| **Database** | SQLite (dev) + PostgreSQL (prod-ready) |
| **Security** | bcrypt, CSRF, HttpOnly cookies, prompt-injection guard |

---

## Key Features

- **Multi-LLM fallback** - 4 AI providers, auto-switches on rate limits
- **Frozen boilerplate** - auth, DB, and layout templates are never AI-generated
- **Auto-repair loop** - detects and fixes missing imports, aliases, and syntax errors
- **Docker sandbox** - every generated app runs isolated (256 MB RAM, 0.5 CPU)
- **Human approval gate** - nothing deploys without your explicit yes
- **Full audit trail** - every action logged with trace IDs
- **RAG-augmented Coder** - 5 knowledge collections (FastAPI, Jinja2, Tailwind, SQLAlchemy, Pydantic)

---

## Quick Start

**Prerequisites:** Python 3.12, Docker Desktop, one free LLM API key

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/ai-company.git
cd ai-company

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env           # then edit .env with your API keys

# 5. Start infrastructure (Redis + ChromaDB)
docker run -d --name redis -p 127.0.0.1:6379:6379 redis --appendonly yes
docker run -d --name chroma -p 127.0.0.1:8888:8000 chromadb/chroma:0.5.23

# 6. Start the platform (two terminals)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000     # Terminal 1: API
python -m arq workers.build_worker.WorkerSettings                # Terminal 2: Worker