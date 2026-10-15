# AI Software Company

> Local-first multi-agent AI system that turns a plain-English requirement into a working web application.

## What It Does

Give it a sentence like "A recipe sharing app with ratings and comments" and it returns a fully generated FastAPI web application, tested in a Docker sandbox, with a human-approval gate before deployment.

## The 7 Agents

| # | Agent | Role |
|---|-------|------|
| 1 | PM | Requirement -> features |
| 2 | Architect | Files, DB schema, routes |
| 3 | Designer | Palettes, fonts, layout |
| 4 | Coder | File-by-file generation |
| 5 | Tester | Docker sandbox tests |
| 6 | Improver | Lighthouse audit |
| 7 | Deploy | GitHub push + local serve |

## Tech Stack

- Python 3.12, FastAPI, Uvicorn
- LangGraph, ARQ + Redis
- Claude Sonnet 4.6 (primary), Groq / Gemini / OpenRouter (fallback)
- ChromaDB, Docker, SQLite

## Quick Start

1. `python -m venv venv` and activate
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add API keys
4. Start Redis + ChromaDB via Docker
5. `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`
6. `python -m arq workers.build_worker.WorkerSettings`

Open http://127.0.0.1:8000/login

## License

MIT
