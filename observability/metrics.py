"""
AI Software Company — Metrics aggregator (Step 6)

Architecture references:
  Section 19  /metrics response shape (all fields)

Design:
  - In-memory counters (reset per process restart). Authoritative history
    lives in the audit_log and job tables (later steps).
  - Thread-safe via threading.Lock.
  - Cost/token/rag fields pull from observability.cost_tracker.
  - Job-related counters (jobs_total, jobs_success, jobs_failed, active_jobs)
    are updated by the API/queue in later steps. This module only aggregates.
  - Disk usage (MB) is read lazily from the filesystem.
  - Never stores or logs secrets.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from observability.cost_tracker import get_tracker


# --- Counters ------------------------------------------------------------
class MetricsState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jobs_total: int = 0
        self.jobs_success: int = 0
        self.jobs_failed: int = 0
        self.deployments_total: int = 0
        self.rollbacks_total: int = 0
        self.active_jobs: int = 0
        self.rate_limit_hits: int = 0
        self.sandbox_timeouts: int = 0
        self.artifacts_cleaned_total: int = 0
        self.images_pruned_total: int = 0
        # Running totals for averages
        self._build_time_total_ms: int = 0
        self._build_time_count: int = 0
        self._test_attempts_total: int = 0
        self._test_attempts_count: int = 0

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            if not hasattr(self, name):
                raise AttributeError(f"unknown metric: {name}")
            setattr(self, name, getattr(self, name) + amount)

    def decr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            if not hasattr(self, name):
                raise AttributeError(f"unknown metric: {name}")
            setattr(self, name, max(0, getattr(self, name) - amount))

    def record_build_time(self, ms: int) -> None:
        with self._lock:
            self._build_time_total_ms += int(ms or 0)
            self._build_time_count += 1

    def record_test_attempts(self, attempts: int) -> None:
        with self._lock:
            self._test_attempts_total += int(attempts or 0)
            self._test_attempts_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg_build = (
                self._build_time_total_ms / self._build_time_count
                if self._build_time_count else 0
            )
            avg_attempts = (
                self._test_attempts_total / self._test_attempts_count
                if self._test_attempts_count else 0
            )
            total_finished = self.jobs_success + self.jobs_failed
            success_rate = (
                self.jobs_success / total_finished
                if total_finished else 0.0
            )
            return {
                "jobs_total": self.jobs_total,
                "jobs_success": self.jobs_success,
                "jobs_failed": self.jobs_failed,
                "success_rate": round(success_rate, 4),
                "average_build_time": round(avg_build, 1),
                "average_test_attempts": round(avg_attempts, 2),
                "deployments_total": self.deployments_total,
                "rollbacks_total": self.rollbacks_total,
                "active_jobs": self.active_jobs,
                "rate_limit_hits": self.rate_limit_hits,
                "sandbox_timeouts": self.sandbox_timeouts,
                "artifacts_cleaned_total": self.artifacts_cleaned_total,
                "images_pruned_total": self.images_pruned_total,
            }

    def reset(self) -> None:
        with self._lock:
            self.jobs_total = 0
            self.jobs_success = 0
            self.jobs_failed = 0
            self.deployments_total = 0
            self.rollbacks_total = 0
            self.active_jobs = 0
            self.rate_limit_hits = 0
            self.sandbox_timeouts = 0
            self.artifacts_cleaned_total = 0
            self.images_pruned_total = 0
            self._build_time_total_ms = 0
            self._build_time_count = 0
            self._test_attempts_total = 0
            self._test_attempts_count = 0


_state = MetricsState()


def get_metrics_state() -> MetricsState:
    return _state


def _disk_usage_mb(path: Optional[Path] = None) -> float:
    """Return disk usage of the given path in MB. Returns 0 on error."""
    try:
        p = Path(path or os.getcwd())
        usage = shutil.disk_usage(str(p))
        return round(usage.used / (1024 * 1024), 1)
    except Exception:
        return 0.0


def full_metrics() -> dict:
    """
    Return the /metrics payload: Section 19 shape + cost_tracker fields.
    """
    snap = _state.snapshot()
    cost = get_tracker().snapshot()

    total_tokens = cost.get("total_tokens", 0)
    estimated_cost = cost.get("total_cost", 0.0)
    jobs_total = snap.get("jobs_total", 0) or 0
    cost_per_job = (estimated_cost / jobs_total) if jobs_total else 0.0
    rag_avg = cost.get("rag_tokens_avg", 0.0)

    return {
        # Section 19 required fields
        "jobs_total": snap["jobs_total"],
        "jobs_success": snap["jobs_success"],
        "jobs_failed": snap["jobs_failed"],
        "success_rate": snap["success_rate"],
        "average_build_time": snap["average_build_time"],
        "average_test_attempts": snap["average_test_attempts"],
        "total_tokens": total_tokens,
        "estimated_cost": round(estimated_cost, 8),
        "deployments_total": snap["deployments_total"],
        "rollbacks_total": snap["rollbacks_total"],
        "active_jobs": snap["active_jobs"],
        "rate_limit_hits": snap["rate_limit_hits"],
        "sandbox_timeouts": snap["sandbox_timeouts"],
        "cost_per_job": round(cost_per_job, 8),
        "rag_tokens_avg": rag_avg,
        "disk_usage_mb": _disk_usage_mb(),
        "artifacts_cleaned_total": snap["artifacts_cleaned_total"],
        "images_pruned_total": snap["images_pruned_total"],
        # extras (useful for dashboards, harmless for the spec)
        "total_calls": cost.get("total_calls", 0),
        "average_latency_ms": cost.get("average_latency_ms", 0.0),
        "fallback_calls": cost.get("fallback_calls", 0),
        "per_model": cost.get("per_model", {}),
        "per_agent": cost.get("per_agent", {}),
        "per_job": cost.get("per_job", {}),
    }