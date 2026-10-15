"""
AI Software Company — Groq configuration (Step 4)

Architecture references:
  Section 2  Tech stack: Groq primary + fallback models
  Section 6  Groq limits + retry strategy
  F3         max_tokens cap + asyncio.sleep between calls
  F10        Global Groq limit (Redis shared counter — later step)
  Notes      Provider quotas change. Treat values as config, not guarantees.
             llama-3.3-70b-versatile and llama-3.1-8b-instant were retired
             by Groq on 2026-08-16. Migrated to official replacements:
             openai/gpt-oss-120b and openai/gpt-oss-20b.
"""

import os

# --- Models ---------------------------------------------------------------
# Migration (2026-10-10): switched primary to Qwen 3.8 27B.
# Reasons:
#   - Qwen 3.8 27B: LiveCodeBench v6 90.3%, SWE-bench Pro 61.7%.
#     Top-tier coding quality, available on Groq free tier.
#   - 30 RPM / 200K TPD (fresh daily quota, unlike exhausted 120b).
#   - openai/gpt-oss-120b remains as fallback (excellent quality).
#   - Kimi K2 removed: not available on free tier (404 model_not_found).
#   - openai/gpt-oss-20b dropped: real-world instruction-following issues.
PRIMARY_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

# Models that accept `extra_body={"reasoning_effort": ...}` on Groq.
REASONING_MODELS = {
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
}


def supports_reasoning_effort(model: str) -> bool:
    """True if `model` accepts the reasoning_effort parameter."""
    return model in REASONING_MODELS

# --- Output token caps ----------------------------------------------------
# Free tier on Groq caps chat models at 8,000 TPM. We cap output per call
# so a single request stays well inside that budget.
PRIMARY_MAX_TOKENS = int(os.getenv("GROQ_PRIMARY_MAX_TOKENS", "4000"))
FALLBACK_MAX_TOKENS = int(os.getenv("GROQ_FALLBACK_MAX_TOKENS", "4000"))

# --- Reasoning effort (GPT-OSS models) ------------------------------------
# openai/gpt-oss-* are reasoning models. Allowed: "low", "medium", "high".
# Default "low" for speed and lower token usage. Override per agent later.
DEFAULT_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# --- Temperature default --------------------------------------------------
# GPT-OSS models recommend 0.5–0.7 range; per-agent overrides come later.
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_DEFAULT_TEMPERATURE", "0.2"))

# --- Rate limiting --------------------------------------------------------
# Global 25 RPM limiter (F10) across all users, implemented via aiolimiter.
GLOBAL_RPM = int(os.getenv("GROQ_GLOBAL_RPM", "25"))

# Small delay between Groq calls (Section 6, F3).
INTER_CALL_SLEEP_SECONDS = float(os.getenv("GROQ_INTER_CALL_SLEEP_SECONDS", "2"))

# Retry policy
MAX_RETRIES_PER_MODEL = int(os.getenv("GROQ_MAX_RETRIES_PER_MODEL", "3"))
BACKOFF_BASE_SECONDS = float(os.getenv("GROQ_BACKOFF_BASE_SECONDS", "2"))
BACKOFF_MAX_SECONDS = float(os.getenv("GROQ_BACKOFF_MAX_SECONDS", "1800"))

# --- Token budget (Coder, per Section 6) ----------------------------------
CODER_TPM_BUDGET = 8000
CODER_RAG_TOKEN_CAP = 800
CODER_OUTPUT_RESERVE = 4000