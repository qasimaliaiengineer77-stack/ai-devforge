# PROJECT STATUS — AI Software Company

Last updated: 2026-10-10
Current step: Step 17 (API + Platform UI + Cleanup Job) — waiting for user approval
Last completed: Step 16 (LangGraph Workflow) — COMPLETE
## Architecture
AI Software Company — Final Production Architecture v7.0
7 agents, 18 steps, 37 fixes. Source of truth: master prompt file.

## Steps 1-12 (COMPLETE)
Step 1: Accounts + Tools
Step 2: Backend setup (FastAPI)
Step 3: Auth + Roles + seed_admin + sessions + logout + CSRF
Step 4: AI setup (Groq, migrated to openai/gpt-oss-120b + openai/gpt-oss-20b)
Step 5: Security layer (17/17 PASS)
Step 6: Observability (3 PASS)
Step 7: PM Agent (3 PASS)
Step 8: Architect Agent (10 files, 2 tables, 9 routes PASS)
Step 9: Designer Agent (5 palettes, WCAG AA PASS)
Step 10: RAG + Coder Agent (10/10 files PASS, chroma Docker container)
Step 11: Dependency Check + SECURITY UPGRADES:
        fastapi 0.115.6 -> 0.142.2
        starlette 0.41.3 -> 1.6.0 (5 HIGH CVEs patched)
        jinja2 3.1.5 -> 3.1.6 (sandbox breakout CVE)
        python-multipart 0.0.20 -> 0.0.32 (6 CVEs)
        pip-audit==2.7.3 added
Step 12: Tester Agent (Docker sandbox, 2-phase)

## Step 13 — Improver Agent (Lighthouse) (COMPLETE)

### Files created:
- docker/lighthouse/Dockerfile
    Node 22-slim + Chromium + Xvfb + Chromium deps
    (kept in repo for reference; not used at runtime after pivot below)
- sandbox/app_runner.py:
    - build_app_image(): writes generated files to workspace and
      `docker build` the app image (Section 37 Dockerfile).
    - run_app_container(): runs container on ai_company_network with
      --memory 512m, --cpus 1.0, --pids-limit 200, --cap-drop ALL,
      --security-opt no-new-privileges, -e PYTHONUNBUFFERED=1,
      -e DATABASE_URL=sqlite:///./app.db.
      Host ports only if host_port given (localhost-only, F5 / Sec 37).
    - stop_app_container / remove_app_image / cleanup_app
    - wait_until_ready(): polls docker logs for "Uvicorn running on"
      or "Application startup complete".
- agents/improver_agent.py:
    - run_improver_agent(files, job_id, user_id, trace_id, output_root,
                         run_lighthouse=True) -> ImproverResult
    - _run_lighthouse(url): runs Lighthouse container on shared network
    - _extract_categories(stdout): parses JSON from stdout
    - _scores_from_report(data): {category: 0..100}
    - _summarize_lighthouse_issues(stdout): top-N failing audits
    - Always cleans up app container + image in finally

### CRITICAL PIVOT (Step 13)
Custom Lighthouse Dockerfile was replaced with the community-maintained
image `femtopixel/google-lighthouse:latest`.

Reason:
1. Custom image hit the chrome-launcher stale-devtools-port bug
   (GitHub issue #535) — "waiting for dynamic debugging port in
   chrome-err.log". chrome-launcher reads an obsolete port from the
   persistent profile's stderr log instead of the freshly launched
   browser's port.
2. Google's own Lighthouse CI docs explicitly say Docker + Chrome
   needs --cap-add=SYS_ADMIN, plus --shm-size=2g (Docker default
   64MB /dev/shm causes Chrome to crash silently).
3. femtopixel handles Chrome launch, profile rotation, and env-var
   based CHROME_FLAGS. Verified via manual test (see below).

### VERIFIED Lighthouse invocation (Step 13)
Image     : femtopixel/google-lighthouse:latest
Caps      : --cap-add SYS_ADMIN
Shm       : --shm-size=2g
Env       : CHROME_FLAGS=--headless=new --no-sandbox
                          --disable-dev-shm-usage
                          --disable-setuid-sandbox
                          --disable-gpu
Output    : --output=json  (to stdout, NOT --output-path)
No        : custom --remote-debugging-port (image manages its own)
Network   : ai_company_network (shared)
URL       : http://app-{job_id}:8000 (service name, F22/F23)
Timeout   : LIGHTHOUSE_TIMEOUT_S = 150

### MINIMAL CORRECTIONS (Step 13)
1. sandbox/app_runner.py: added -e PYTHONUNBUFFERED=1 to the app
   container so uvicorn logs are flushed immediately and
   wait_until_ready() can detect startup in detached mode.
2. BUILD_TIMEOUT_S raised 180 -> 300 (slow pip on this network).
3. Improver no longer passes --chrome-flags or --remote-debugging-*.

### Step 13 test (manual verification PASS)
Manual run against app-verify-001 (mock compliant app) with the
verified command above:
    categories: performance=49, accessibility=82, best-practices=74, seo=90
    JSON output = 594 KB, valid, saveable, parseable.

### Step 13 test (agent integration PASS)
tests_step13.py with the same mock app:
    ok: True
    app_build: 80s (rc=0)
    app_run: 8s, wait_until_ready OK
    lighthouse: 54s, returncode 0
    before_scores: {performance: 62, accessibility: 82,
                    best-practices: 74, seo: 90}
    after_scores:  (same as before; real improvement is Step 16)
    issues_count: 5 (is-on-https, redirects-http, total-blocking-time,
                     max-potential-fid, errors-in-console)
    duration_ms: ~159000
    cleanup: OK

## Files created so far (full list)
- PROJECT_STATUS.md, .gitignore, .env (git-ignored), requirements.txt
- api/main.py, api/routes_auth.py
- db/: database.py, models.py, audit.py, __init__.py
- auth/: passwords.py, sessions.py, deps.py, __init__.py
- scripts/: seed_admin.py, reset_admin_password.py, __init__.py
- config/: groq_config.py, __init__.py
- ai/: groq_client.py, state.py, __init__.py
- security/: secret_scan.py, path_check.py, prompt_guard.py,
             dependency_check.py, __init__.py
- observability/: logger.py, cost_tracker.py, metrics.py, __init__.py
- agents/: pm_agent.py, architect_agent.py, designer_agent.py,
           coder_agent.py, tester_agent.py, improver_agent.py, __init__.py
- rag/: chroma_client.py, embedding_fn.py, chunker.py, retriever.py, __init__.py
- sandbox/: docker_runner.py, test_runner.py, app_runner.py, __init__.py
- docker/lighthouse/Dockerfile (reference only)
- ai_company.db (SQLite, git-ignored), venv/ (git-ignored)

## Docker images used at runtime
- redis (latest)                     — port 127.0.0.1:6379
- chromadb/chroma:0.5.23             — port 127.0.0.1:8888
- femtopixel/google-lighthouse:latest — Lighthouse runner
- ai-company-app-{job_id}:latest      — ephemeral, removed after run
- ai-company-sandbox-{job_id}:latest  — ephemeral, removed after run

## Failed tests
- None (in final state).

## Known bugs / open decisions
- Docker Desktop memory 1.812 GiB — may need increase later.
- Git repo NOT yet initialized. User deferred. Flag for Step 18.
- ai/groq_client.py line ~124 has cosmetic indent mismatch (functionally fine).
- disk_usage_mb reports TOTAL disk usage of C: (deferred).
- Groq TPM 8000 (openai/gpt-oss-120b).
- Starlette deprecation warning about httpx2 (informational only).
- Improver "issues" list is informational; no repair loop yet.
  Real repair/regeneration happens in Step 16 (LangGraph routes back
  to Coder if after_scores < before_scores or below threshold).

## SECURITY INCIDENTS (need follow-up)
- Admin password "78612345" appeared in chat multiple times earlier. Reset.
- Session cookie + CSRF token values appeared in chat twice. Deleted.
- ACTION: Never paste passwords, cookies, CSRF tokens, API keys into chat.

## Next single action
Wait for user approval to start Step 14 (Deploy Agent — GitHub push,
Port Manager, local serve).

## Backup / Git commit status
- Git repo initialized at Step 14 start.
- First commit done: e7eac9a "Step 14: Initial commit — steps 1-13 complete"
- 47 files committed, 6389 insertions.
- .env, venv/, ai_company.db, output/, logs/ excluded by .gitignore (VERIFIED).
- GitHub remote NOT added yet — Step 14 later substeps.
- Repo will be created as PRIVATE on GitHub (user decision).

## Security notes
- Redis bound to 127.0.0.1 only.
- Chroma bound to 127.0.0.1 only.
- .env in .gitignore.
- bcrypt==4.0.1 pinned per F7.
- Groq client + structlog both redact secrets before emission.
- Session cookie: HttpOnly, SameSite=Lax.
- CSRF token: sessions table, constant-time compare.
- Security layer: 3-level prompt defense + path traversal block + secret scan.
- Architect validates every file path; Designer validates WCAG AA contrast.
- Coder validates every generated file (path + syntax + imports + secrets +
  dangerous patterns).
- Dependency check: PyPI + version + pip-audit CVE scan, banned pkgs blocked.
- Tester runs in Docker sandbox with full hardening; 3-level check.
- Improver uses SYS_ADMIN + shm=2g for Lighthouse (trusted tool, ephemeral),
  while generated app containers keep --cap-drop ALL. 

## Step 15 — Human Approval Gate (server-side) (COMPLETE)

### Files created/updated:
- db/models.py: added `ApprovalRequest` table
    id, job_id (UNIQUE), requested_by FK, status (pending/approved/rejected),
    decision (yes/no/NULL), decided_by FK, decided_at, created_at
- db/audit.py: added ACTION_APPROVAL_REQUESTED / GRANTED / REJECTED
- security/approval.py:
    request_approval(db, job_id, requested_by) -> ApprovalRequest (idempotent)
    decide_approval(db, job_id, decision, decider_user_id)
    get_approval(db, job_id), is_approved(db, job_id)
    State transitions: (none)->pending->approved|rejected
    Once decided, cannot change (ApprovalAlreadyDecidedError)
    Audit log on each state change; commit on success, rollback on exception
- api/routes_approval.py:
    POST /approve/{job_id}  body {"decision":"yes"|"no"}
        - role check: user.role == "approver" (403 otherwise)
        - CSRF check via X-CSRF-Token (403 otherwise)
        - 404 if not found, 409 if already decided, 400 if invalid decision
    GET  /approval/{job_id}
        - any logged-in user; returns status or "none"
- api/main.py:
    - Fixed: auth_router was accidentally removed in Step 15 (restored)
    - added approval_router include
    - current routes: /, /health, /login, /logout, /me,
                      /approve/{job_id}, /approval/{job_id}

### Step 15 tests (12/12 PASS)
T1  login both users (user1, approver1)
T2  GET /approval before request -> status "none"
T3  request created -> status "pending"
T4  GET -> "pending"
T5  user role POST /approve -> 403 "Approval Authority role required"
T6  approver without CSRF -> 403 "CSRF check failed"
T7  approver with CSRF -> 200 status "approved", decided_by set
T8  GET -> "approved"
T9  duplicate decision -> 409 "already 'approved'"
T10 reject flow -> 200 status "rejected", decision "no"
T11 GET rejected job -> "rejected"
T12 invalid decision ("maybe") -> 400 "decision must be 'yes' or 'no'"

### Security features verified
- Role-based access (Section 7)
- CSRF (Section 8) — POST requires X-CSRF-Token
- Idempotency: once decided, further POST -> 409
- Input validation: decision must be "yes"|"no"
- Audit log actions: approval_requested, approval_granted, approval_rejected
- Atomic writes: commit on success, rollback on exception

### Git commit (Step 15)
949e055  Step 15: Human Approval Gate (server-side, /approve + /approval)

### Note on test users
- approver1 (role=approver) and user1 (role=user) created by the test.
  These remain in the SQLite DB. Fine for further testing; the platform
  seed script only creates an admin (Step 3).  

## Step 16 — LangGraph Workflow (COMPLETE)

### Files created:
- graph/__init__.py
- graph/state.py:
    WorkflowState (TypedDict, total=False) — extends Section 16 AppState
    with LangGraph routing fields: current_node, next_route,
    coder_attempts, tester_attempts, improver_attempts, approval_status,
    errors, started_at, finished_at.
    Constants: MAX_CODER_ATTEMPTS=3, MAX_IMPROVER_ATTEMPTS=2.
    Helpers: new_workflow_state, add_error, bump, is_approved, mark_finished.
- graph/nodes.py:
    9 async node functions, each: state -> partial dict
        sanitize_node, pm_node, architect_node, designer_node,
        coder_node, tester_node, improver_node, approval_node, deploy_node
    Nodes never raise on agent failure; they record to state["errors"]
    and set next_route (either next agent or "abort").
    Per-node logging via observability.logger with job binding.
- graph/workflow.py:
    build_graph() -> CompiledStateGraph
    START -> sanitize -> pm -> architect -> designer -> coder -> tester
    Tester pass -> improver; Tester fail -> coder (if attempts < 3)
    Improver ok -> approval; Improver fail with retries -> coder
    approval -> END (deploy is a separate resume step)
    Any *node returns next_route="abort" -> abort -> END
    run_to_approval(requirement, job_id, user_id, trace_id) -> WorkflowState
    resume_deploy(prev_state) -> WorkflowState (runs deploy_node directly)

### Step 16 tests (7/7 PASS)
T1 new_workflow_state defaults (0 errors, started_at set)
T2 sanitize_node clean input -> next_route "pm"
T3 sanitize_node injection -> next_route "abort" + error recorded
T4 pm_node (real Groq) -> SimpleTodo, 4 features, 4 pages
T5 architect_node (real Groq) -> 10 files, 2 tables, 9 routes
T6 designer_node (real Groq) -> Creative palette, design_brief complete
T7 state integrity after 4 nodes -> 0 errors, next_route "coder"

Graph compile check:
    python -c "from graph.workflow import build_graph; build_graph()"
    -> "graph compiled OK", type=CompiledStateGraph

### Git commit (Step 16)
46303b6  Step 16: LangGraph workflow (9 nodes + conditional edges + abort path)

### Notes
- Checkpointer (SQLite) was mentioned in the plan; deferred. V1 keeps
  state in-memory during run_to_approval and passes it explicitly to
  resume_deploy. Adding a SQLite checkpointer is a Step 18 hardening
  item if needed.
- Real end-to-end flow (coder + tester + improver) will be exercised
  in Step 17 (API) and Step 18 (final testing).
## Step 17 — WIP (paused 2026-10-10 evening)

Completed today:
- API layer (routes_status, routes_ui, main.py wiring + slowapi)
- Jobs store (db/jobs_store.py: create/get/list/update + JSON blobs)
- ARQ queue + worker (workers/queue.py, workers/build_worker.py)
- Platform UI (5 templates: base, login, submit, jobs, job_status)
- Coder JSON extraction fix (prose/fence robust)
- Worker timeout 15 -> 30 min
- Requirements sanitizer before docker build
- Worker final status fix (aborted -> failed)
- Model migrations: 20b -> Kimi K2 (404) -> Qwen 3.8 (OTPM limit) -> 20b (with strong prompt)
- Coder prompt strengthened with per-file MUST lists
- Retry-After cap at BACKOFF_MAX

Paused reason:
- Groq free tier daily quota (TPD) exhausted after multiple end-to-end runs today.
- 120b TPD: 200K (exhausted), 20b TPD: 500K (exhausted), Qwen OTPM: 1000 (too small)
- Free tier resets daily. Test to resume after reset.

To resume (next session):
1. Start Docker Desktop (whale green)
2. Window A: cd Desktop\ai-company ; .\venv\Scripts\Activate.ps1
3. Window A: python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
4. Window B (new PS): cd Desktop\ai-company ; .\venv\Scripts\Activate.ps1
5. Window B: python -m arq workers.build_worker.WorkerSettings
6. Browser: http://127.0.0.1:8000/login (user1, approver1 passwords on user's paper)
7. New Build -> "Make a simple todo app with login"
8. Watch for awaiting_approval -> then approver1 login -> approve -> check live_url

Current primary model: openai/gpt-oss-20b (500K TPD)
Fallback: openai/gpt-oss-120b (200K TPD)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     