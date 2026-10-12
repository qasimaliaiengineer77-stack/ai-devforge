"""Add SQLite checkpointer to LangGraph workflow for resume support."""
from pathlib import Path

# ============================================================
# 1. graph/workflow.py - add checkpointer parameter
# ============================================================
p1 = Path("graph/workflow.py")
s1 = p1.read_text(encoding="utf-8")

# Update build_graph signature
old_sig = '''def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Returns the compiled graph. It is safe to reuse across jobs.
    """
    g = StateGraph(WorkflowState)'''

new_sig = '''def build_graph(checkpointer=None) -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence.
                      Use SqliteSaver for local dev, PostgresSaver for prod.

    Returns the compiled graph. It is safe to reuse across jobs.
    """
    g = StateGraph(WorkflowState)'''

if old_sig in s1:
    s1 = s1.replace(old_sig, new_sig, 1)
    print("Fix 1a: build_graph signature updated")
else:
    print("Fix 1a: signature anchor not found (already patched?)")

# Update compile call
old_compile = "    return g.compile()"
new_compile = "    return g.compile(checkpointer=checkpointer)"

if old_compile in s1:
    s1 = s1.replace(old_compile, new_compile, 1)
    print("Fix 1b: compile with checkpointer")
else:
    print("Fix 1b: compile anchor not found")

# Add resume_workflow function at end
resume_fn = '''


async def resume_workflow(job_id: str, resume_value: dict) -> WorkflowState:
    """
    Resume a paused workflow from the last checkpoint.

    Call this after the user resolves a failure via /resolve-failure.
    The checkpointer loads the exact state where the workflow stopped.
    """
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    # Create checkpointer (same DB as worker)
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    log = bind_job(logger, job_id=job_id)
    log.info("workflow_resume_start", job_id=job_id)

    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": job_id}}

    try:
        # Get current state
        state = graph.get_state(config)
        log.info("resume_state_found", next_steps=state.next)

        # Resume by passing the updated state
        # We update the state with the resume_value (which may include
        # corrected code files, deleted files, etc.)
        graph.update_state(config, resume_value)

        # Continue execution from where it left off
        result = await graph.ainvoke(None, config)
        log.info("workflow_resume_done", status=result.get("status"))
        return result
    except Exception as exc:
        log.error("workflow_resume_failed", err=str(exc))
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass
'''

if "async def resume_workflow" not in s1:
    s1 = s1.rstrip() + resume_fn + "\n"
    print("Fix 1c: resume_workflow function added")
else:
    print("Fix 1c: resume_workflow already exists")

p1.write_text(s1, encoding="utf-8")


# ============================================================
# 2. workers/build_worker.py - use checkpointer
# ============================================================
p2 = Path("workers/build_worker.py")
s2 = p2.read_text(encoding="utf-8")

# Add checkpointer import
old_import = "from graph.workflow import build_graph"
new_import = "from graph.workflow import build_graph"
# Add sqlite import after this
if "from langgraph.checkpoint.sqlite import SqliteSaver" not in s2:
    s2 = s2.replace(
        "import os\nimport time",
        "import os\nimport sqlite3\nimport time",
        1
    )
    s2 = s2.replace(
        "from graph.workflow import build_graph",
        "from graph.workflow import build_graph\nfrom langgraph.checkpoint.sqlite import SqliteSaver",
        1
    )
    print("Fix 2a: sqlite checkpointer imports added")

# Update graph = build_graph() to use checkpointer
old_graph = "        graph = build_graph()"
new_graph = '''        # Create SQLite checkpointer (persists state across process restarts)
        conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()

        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": job_id}}'''

if old_graph in s2:
    s2 = s2.replace(old_graph, new_graph, 1)
    print("Fix 2b: worker uses checkpointer")
else:
    print("Fix 2b: graph anchor not found")

# Update astream call to pass config
old_stream = '        async for event in graph.astream(initial, stream_mode="updates"):'
new_stream = '        async for event in graph.astream(initial, config, stream_mode="updates"):'

if old_stream in s2:
    s2 = s2.replace(old_stream, new_stream, 1)
    print("Fix 2c: astream passes config")
else:
    print("Fix 2c: astream anchor not found")

# Close connection in finally
old_finally = '''    except Exception as exc:
        log.error(
            "worker_exception",
            err=f"{type(exc).__name__}: {exc}",
        )'''
new_finally = '''    except Exception as exc:
        log.error(
            "worker_exception",
            err=f"{type(exc).__name__}: {exc}",
        )'''
# No change needed for now

p2.write_text(s2, encoding="utf-8")

print()
print("CHECKPOINTER ADDED")
print("Next: add resume endpoint in api/routes_status.py")