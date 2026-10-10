"""
AI Software Company — FastAPI backend (Step 2 + Step 3 + Step 15)
Routes:
  /health, /                          — platform health
  /login, /logout, /me                — auth (Step 3)
  /approve/{job_id}, /approval/{job_id} — approval (Step 15)
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes_auth import router as auth_router
from api.routes_approval import router as approval_router
from db.database import init_db

# Load .env file (secrets yahan se aayenge)
load_dotenv()

APP_ENV = os.getenv("APP_ENV", "dev")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"[startup] APP_ENV={APP_ENV} DEBUG={DEBUG}")
    init_db()  # create tables if they don't exist
    print("[startup] database ready")
    yield
    # Shutdown
    print("[shutdown] platform stopped")


app = FastAPI(
    title="AI Software Company",
    version="0.1.0",
    lifespan=lifespan,
)

# Auth routes (login / logout / me)
app.include_router(auth_router)

# Approval routes (approve / approval status)
app.include_router(approval_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Software Company platform is running."}