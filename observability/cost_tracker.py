"""
AI Software Company — Cost + token tracker (Step 6)

Architecture references:
  Section 20  Cost tracking fields:
              model, input_tokens, output_tokens, total_tokens,
              latency_ms, estimated_cost, rag_tokens
  Section 19  /metrics fields that depend on this tracker:
              total_tokens, estimated_cost, cost_per_job, rag_tokens_avg

Design:
  - In-memory aggregator (resets on process restart). This is intentional:
    the authoritative audit trail lives in the audit_log table (Step 3).
  - Thread-safe via threading.Lock because FastAPI + ARQ worker may share.
  - Pricing is expressed in USD per 1M tokens. Free-tier Groq pricing is
    $0, but we still record the structure so production pricing later is
    a one-line config change.
  - Never stores or logs secrets.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# --- Pricing (USD per 1,000,000 tokens) ---------------------------------
# Groq free tier = $0. These values are placeholders so the structure is
# ready for production pricing. Update when paid tier is enabled.
_PRICING_PER_1M = {
    "openai/gpt-oss-120b": {"input": 0.0, "output": 0.0},
    "openai/gpt-oss-20b":  {"input": 0.0, "output": 0.0},
    # legacy (kept for safety, in case old logs are re-processed)
    "llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},
    "llama-3.1-8b-instant":    {"input": 0.0, "output": 0.0},
}

_DEFAULT_PRICING = {"input": 0.0, "output": 0.0}


def price_for(model: str) -> dict:
    """Return {input, output} price per 1M tokens for a model."""
    return _PRICING_PER_1M.get(model, _DEFAULT_PRICING)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD estimate for one call."""
    p = price_for(model)
    return (
        (input_tokens / 1_000_000.0) * p["input"]
        + (output_tokens / 1_000_000.0) * p["output"]
    )


# --- Per-call record -----------------------------------------------------
@dataclass
class CallRecord:
    job_id: str
    agent: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    estimated_cost: float = 0.0
    rag_tokens: int = 0
    used_fallback: bool = False
    attempts: int = 0


# --- Tracker -------------------------------------------------------------
class CostTracker:
    """
    Aggregates AI call stats for /metrics and dashboards.
    Reset per process restart (by design).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_calls: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._total_latency_ms: int = 0

        self._per_model: dict[str, dict] = defaultdict(
            lambda: {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "latency_ms": 0,
            }
        )
        self._per_agent: dict[str, dict] = defaultdict(
            lambda: {
                "calls": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "latency_ms": 0,
            }
        )
        self._per_job: dict[str, dict] = defaultdict(
            lambda: {
                "calls": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "latency_ms": 0,
                "rag_tokens": 0,
            }
        )
        self._rag_tokens_total: int = 0
        self._fallback_calls: int = 0

    # --- Recording -------------------------------------------------------
    def record(
        self,
        *,
        job_id: str,
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: Optional[int] = None,
        latency_ms: int = 0,
        rag_tokens: int = 0,
        used_fallback: bool = False,
        attempts: int = 1,
    ) -> CallRecord:
        """Record one AI call. Returns the CallRecord."""
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        cost = estimate_cost(model, input_tokens, output_tokens)
        rec = CallRecord(
            job_id=job_id,
            agent=agent,
            model=model,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            total_tokens=int(total_tokens or 0),
            latency_ms=int(latency_ms or 0),
            estimated_cost=float(cost),
            rag_tokens=int(rag_tokens or 0),
            used_fallback=bool(used_fallback),
            attempts=int(attempts or 1),
        )

        with self._lock:
            self._total_calls += 1
            self._total_input_tokens += rec.input_tokens
            self._total_output_tokens += rec.output_tokens
            self._total_tokens += rec.total_tokens
            self._total_cost += rec.estimated_cost
            self._total_latency_ms += rec.latency_ms
            self._rag_tokens_total += rec.rag_tokens
            if rec.used_fallback:
                self._fallback_calls += 1

            pm = self._per_model[model]
            pm["calls"] += 1
            pm["input_tokens"] += rec.input_tokens
            pm["output_tokens"] += rec.output_tokens
            pm["total_tokens"] += rec.total_tokens
            pm["cost"] += rec.estimated_cost
            pm["latency_ms"] += rec.latency_ms

            pa = self._per_agent[agent]
            pa["calls"] += 1
            pa["total_tokens"] += rec.total_tokens
            pa["cost"] += rec.estimated_cost
            pa["latency_ms"] += rec.latency_ms

            pj = self._per_job[job_id]
            pj["calls"] += 1
            pj["total_tokens"] += rec.total_tokens
            pj["cost"] += rec.estimated_cost
            pj["latency_ms"] += rec.latency_ms
            pj["rag_tokens"] += rec.rag_tokens

        return rec

    # --- Snapshot --------------------------------------------------------
    def snapshot(self) -> dict:
        """Return a plain dict for /metrics."""
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._total_calls
                if self._total_calls else 0.0
            )
            return {
                "total_calls": self._total_calls,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_tokens": self._total_tokens,
                "total_cost": round(self._total_cost, 8),
                "average_latency_ms": round(avg_latency, 1),
                "rag_tokens_total": self._rag_tokens_total,
                "rag_tokens_avg": round(
                    self._rag_tokens_total / self._total_calls, 1
                ) if self._total_calls else 0.0,
                "fallback_calls": self._fallback_calls,
                "per_model": {k: dict(v) for k, v in self._per_model.items()},
                "per_agent": {k: dict(v) for k, v in self._per_agent.items()},
                "per_job": {k: dict(v) for k, v in self._per_job.items()},
            }

    def reset(self) -> None:
        """Clear all counters. Used in tests."""
        with self._lock:
            self._total_calls = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_tokens = 0
            self._total_cost = 0.0
            self._total_latency_ms = 0
            self._rag_tokens_total = 0
            self._fallback_calls = 0
            self._per_model.clear()
            self._per_agent.clear()
            self._per_job.clear()


# --- Process-wide singleton ---------------------------------------------
_tracker = CostTracker()


def get_tracker() -> CostTracker:
    """Return the process-wide CostTracker."""
    return _tracker


def record_call(**kwargs) -> CallRecord:
    """Convenience wrapper for get_tracker().record(**kwargs)."""
    return _tracker.record(**kwargs)