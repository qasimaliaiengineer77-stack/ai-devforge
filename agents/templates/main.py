"""
Frozen main.py skeleton (template). Marker {{APP_ROUTES}} gets AI routes.
"""

import os
import secrets
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy.orm import Session
from sqlalchemy import select

from database import SessionLocal, engine, Base, backup_db
import models, schemas, auth

# Ensure tables exist at import time (TestClient may not fire startup)
try:
    Base.metadata.create_all(bind=engine)
except Exception:
    pass


app = FastAPI(title="App")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
from jinja2 import ChainableUndefined
templates.env.undefined = ChainableUndefined


def render_first_available(request, candidates: list, context: dict, status_code: int = 200):
    """Try each template in order; return the first that exists. Fallback to first candidate."""
    import os as _os
    for _c in candidates:
        _p = _os.path.join("templates", _c)
        if _os.path.isfile(_p):
            return templates.TemplateResponse(request, _c, context, status_code=status_code)
    return templates.TemplateResponse(request, candidates[0], context, status_code=status_code)


# Safe url_for: returns '/' if route name missing (AI route names vary)
def url_for_safe(name, **params):
    try:
        return app.url_path_for(name, **params)
    except Exception:
        return '/' + name if not name.startswith('/') else name
templates.env.globals['url_for'] = url_for_safe


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


csrf_store: dict = {}


def generate_csrf_token(session_id: str) -> str:
    token = secrets.token_urlsafe(32)
    csrf_store[session_id] = token
    return token


def validate_csrf_token(session_id: str, token: str) -> bool:
    stored = csrf_store.get(session_id)
    return stored is not None and secrets.compare_digest(stored, token)


def get_session_id(request: Request) -> str:
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = secrets.token_urlsafe(16)
    return session_id


def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[models.User]:
    user_id = request.cookies.get("user_id")
    if not user_id:
        return None
    try:
        stmt = select(models.User).where(models.User.id == int(user_id))
        return db.execute(stmt).scalar_one_or_none()
    except Exception:
        return None




@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    session_id = get_session_id(request)
    csrf_token = generate_csrf_token(session_id)
    response = templates.TemplateResponse(request, "login.html", {"csrf_token": csrf_token})
    response.set_cookie(key="session_id", value=session_id, httponly=True, samesite="lax")
    return response


@app.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    session_id = get_session_id(request)
    if not validate_csrf_token(session_id, csrf_token):
        raise HTTPException(status_code=400, detail="Invalid CSRF token")
    stmt = select(models.User).where(models.User.email == email)
    user: Optional[models.User] = db.execute(stmt).scalar_one_or_none()
    if not user or not auth.verify_password(password, user.hashed_password):
        new_token = generate_csrf_token(session_id)
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Invalid credentials", "csrf_token": new_token},
            status_code=401,
        )
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="user_id", value=str(user.id), httponly=True, samesite="lax")
    response.set_cookie(key="session_id", value=session_id, httponly=True, samesite="lax")
    return response


@app.post("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="user_id")
    response.delete_cookie(key="session_id")
    return response


@app.get("/health", response_class=HTMLResponse)
async def health_check():
    return "OK"


@app.on_event("startup")
async def on_startup():
    Base.metadata.create_all(bind=engine)
    try:
        backup_db()
    except Exception:
        pass


@app.on_event("shutdown")
async def on_shutdown():
    try:
        backup_db()
    except Exception:
        pass


# ============================================================================
# AUTO-DISCOVER ROUTES from routes/*.py
# ============================================================================
import importlib as _imp
from pathlib import Path as _Path

_routes_dir = _Path(__file__).parent / "routes"
if _routes_dir.exists() and _routes_dir.is_dir():
    for _py in sorted(_routes_dir.glob("*.py")):
        if _py.name.startswith("_") or _py.name == "__init__.py":
            continue
        _mod_name = f"routes.{_py.stem}"
        try:
            _mod = _imp.import_module(_mod_name)
            if hasattr(_mod, "router"):
                app.include_router(_mod.router)
                print(f"[routes] loaded {_mod_name}")
            else:
                raise RuntimeError(f"{_mod_name} has no router object")
        except Exception as _e:
            print(f"[routes] FAILED {_mod_name}: {_e}")
            raise

