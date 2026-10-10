"""
AI Software Company — FastAPI backend (Step 2 + 3 + 15 + 17)
Routes:
  /health, /                            — platform health
  /login, /logout, /me                  — auth JSON API (Step 3)
  /approve/{job_id}, /approval/{job_id} — approval JSON API (Step 15)
  /status/{job_id}, /ready              — status API (Step 17)
  /login, /submit, /jobs, /jobs/{id}    — platform UI pages (Step 17)
  /ui/login, /ui/logout, /ui/build      — UI form handlers (Step 17)
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.routes_approval import router as approval_router
from api.routes_auth import router as auth_router
from api.routes_status import limiter as status_limiter
from api.routes_status import router as status_router
from api.routes_ui import router as ui_router
from db.database import init_db

# Load .env file (secrets yahan se aayenge)
load_dotenv()

APP_ENV = os.getenv("APP_ENV", "dev")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"[startup] APP_ENV={APP_ENV} DEBUG={DEBUG}")
    init_db()
    print("[startup] database ready")
    yield
    # Shutdown
    print("[shutdown] platform stopped")


app = FastAPI(
    title="AI Software Company",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Rate limiting (Section 25) ---
app.state.limiter = status_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# --- Routers ---
app.include_router(auth_router)          # /login /logout /me (JSON)
app.include_router(approval_router)      # /approve /approval (JSON)
app.include_router(status_router)        # /status /ready
app.include_router(ui_router)            # /login /submit /jobs /ui/* (HTML)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Software Company platform is running."}