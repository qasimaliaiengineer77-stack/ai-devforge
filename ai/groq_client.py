"""
AI Software Company — Groq async client (Step 4)

Architecture references:
  Section 6  Groq rate limits + retry strategy
  F3         max_tokens cap + asyncio.sleep between calls
  F10        Global 25 RPM (aiolimiter)
  F28        Secret leak prevention (logs redacted)

Models (as of Step 4):
  Primary : openai/gpt-oss-120b   (Groq's replacement for llama-3.3-70b-versatile)
  Fallback: openai/gpt-oss-20b    (Groq's replacement for llama-3.1-8b-instant)

Behavior:
  - Process-wide AsyncLimiter (25 RPM) applied to every call.
  - asyncio.sleep(INTER_CALL_SLEEP_SECONDS) after each successful call.
  - On RateLimitError, parse Retry-After and wait, then retry.
  - On other errors, exponential backoff.
  - After MAX_RETRIES_PER_MODEL with primary model, fall back to fallback.
  - All log messages pass through redact() before emission.
  - Never logs the API key, session cookie, or CSRF tokens.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from groq import APIError, AsyncGroq, RateLimitError

from config import groq_config as cfg

load_dotenv()

logger = logging.getLogger("ai.groq_client")

# --- Secrets redaction (F28) ---------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9_\-]+"),
    re.compile(r"ghp_[A-Za-z0-9_\-]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"postgresql://[^\s\"']+"),
    re.compile(r"mysql://[^\s\"']+"),
    re.compile(r"ai_company_session=[^\s;]+"),
]


def redact(text: Any) -> str:
    """Redact secrets from any string before logging."""
    if text is None:
        return ""
    s = str(text)
    for pat in _SECRET_PATTERNS:
        s = pat.sub("[REDACTED]", s)
    return s


# --- Global rate limiter (F10) -------------------------------------------
_global_limiter = AsyncLimiter(max_rate=cfg.GLOBAL_RPM, time_period=60)

# --- Async client (singleton, lazy) --------------------------------------
_client: Optional[AsyncGroq] = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key or api_key.startswith("PUT_"):
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _client = AsyncGroq(api_key=api_key)
    return _client


# --- Result type ---------------------------------------------------------
@dataclass
class GroqResult:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 0
    used_fallback: bool = False
    raw: dict = field(default_factory=dict)


# --- Retry-After parsing -------------------------------------------------
def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Try to find a Retry-After header value from a Groq error."""
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None)
            if headers is not None:
                val = headers.get("retry-after") or headers.get("Retry-After")
                if val is not None:
                    return float(val)
    except Exception:
        pass
    return None


# --- Single API call -----------------------------------------------------
async def _call_once(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
) -> GroqResult:
    client = _get_client()
    t0 = time.perf_counter()
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Only GPT-OSS models accept reasoning_effort. Kimi K2 and others do not.
    if cfg.supports_reasoning_effort(model):
        kwargs["extra_body"] = {
            "reasoning_effort": cfg.DEFAULT_REASONING_EFFORT,
        }
    resp = await client.chat.completions.create(**kwargs)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    choice = ""
    if getattr(resp, "choices", None):
        choice = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tok = int(getattr(usage, "total_tokens", in_tok + out_tok) or 0)
    return GroqResult(
        content=choice,
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=total_tok,
        latency_ms=latency_ms,
    )


# --- Public entry point --------------------------------------------------
async def chat(
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> GroqResult:
    """
    Call Groq with rate limiting and fallback.

    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    temperature: override; default from config
    max_tokens: override; primary cap from config, fallback capped separately

    Raises RuntimeError if all models and retries fail.
    """
    temp = cfg.DEFAULT_TEMPERATURE if temperature is None else temperature
    primary_cap = cfg.PRIMARY_MAX_TOKENS if max_tokens is None else max_tokens
    fallback_cap = min(primary_cap, cfg.FALLBACK_MAX_TOKENS)

    # Skip 20b fallback (flaky) - jump straight to OpenRouter if 120b fails
    models_to_try = [
        (cfg.PRIMARY_MODEL, primary_cap, False),
    ]

    last_exc: Optional[Exception] = None
    total_attempts = 0

    for model, cap, is_fallback in models_to_try:
        for attempt in range(1, cfg.MAX_RETRIES_PER_MODEL + 1):
            total_attempts += 1
            try:
                async with _global_limiter:
                    result = await _call_once(model, messages, temp, cap)

                result.attempts = total_attempts
                result.used_fallback = is_fallback
                logger.info(
                    "groq ok model=%s tokens=%s latency_ms=%s fallback=%s",
                    model,
                    result.total_tokens,
                    result.latency_ms,
                    is_fallback,
                )
                await asyncio.sleep(cfg.INTER_CALL_SLEEP_SECONDS)
                return result

            except RateLimitError as exc:
                last_exc = exc
                raw_wait = _extract_retry_after(exc)
                # If the provider asks us to wait longer than our cap, do
                # not waste time retrying this model; move to the next one.
                if raw_wait is not None and raw_wait > cfg.BACKOFF_MAX_SECONDS:
                    logger.warning(
                        "groq rate_limit_skip model=%s retry_after=%.1fs "
                        "(> cap %.1fs) - moving to next model",
                        model, raw_wait, cfg.BACKOFF_MAX_SECONDS,
                    )
                    break
                wait = (
                    raw_wait
                    if raw_wait is not None
                    else min(
                        cfg.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                        cfg.BACKOFF_MAX_SECONDS,
                    )
                )
                logger.warning(
                    "groq rate_limit model=%s attempt=%d wait=%.1fs msg=%s",
                    model,
                    attempt,
                    wait,
                    redact(exc),
                )
                await asyncio.sleep(wait)

            except APIError as exc:
                last_exc = exc
                wait = min(
                    cfg.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                    cfg.BACKOFF_MAX_SECONDS,
                )
                logger.warning(
                    "groq api_error model=%s attempt=%d wait=%.1fs msg=%s",
                    model,
                    attempt,
                    wait,
                    redact(exc),
                )
                if attempt < cfg.MAX_RETRIES_PER_MODEL:
                    await asyncio.sleep(wait)

            except Exception as exc:
                last_exc = exc
                logger.error(
                    "groq unexpected model=%s attempt=%d msg=%s",
                    model,
                    attempt,
                    redact(exc),
                )
                break

        logger.warning("groq exhausted retries for model=%s, trying next", model)

    # ---- Fallback 1: Gemini ----
    try:
        from ai.gemini_client import chat as gemini_chat
        logger.warning("groq_exhausted_trying_gemini")
        g = await gemini_chat(messages, temperature=temp, max_tokens=primary_cap)
        if g.content.strip():
            return GroqResult(
                content=g.content,
                model="gemini/" + str(g.model),
                input_tokens=g.input_tokens,
                output_tokens=g.output_tokens,
                total_tokens=g.total_tokens,
                latency_ms=g.latency_ms,
                attempts=g.attempts,
                used_fallback=True,
            )
    except Exception as gexc:
        logger.error("gemini_fallback_failed err=%s", redact(gexc))

    # ---- Fallback 2: OpenRouter (free models) ----
    try:
        from ai.openrouter_client import chat as or_chat
        logger.warning("gemini_exhausted_trying_openrouter")
        orr = await or_chat(messages, temperature=temp, max_tokens=primary_cap)
        if orr.content.strip():
            return GroqResult(
                content=orr.content,
                model=orr.model,
                input_tokens=orr.input_tokens,
                output_tokens=orr.output_tokens,
                total_tokens=orr.total_tokens,
                latency_ms=orr.latency_ms,
                attempts=orr.attempts,
                used_fallback=True,
            )
    except Exception as oexc:
        logger.error("openrouter_fallback_failed err=%s", redact(oexc))

    raise RuntimeError(
        f"All providers failed. Groq: {redact(last_exc)}; "
        f"Gemini: {redact(gexc)}; OpenRouter: {redact(oexc)}"
    )