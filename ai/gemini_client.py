"""Google Gemini async client - fallback for Groq."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ai.gemini_client")

_SECRET_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9_\-]+"),
    re.compile(r"AIza[A-Za-z0-9_\-]+"),
    re.compile(r"ghp_[A-Za-z0-9_\-]+"),
]


def redact(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    for pat in _SECRET_PATTERNS:
        s = pat.sub("[REDACTED]", s)
    return s


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_RETRIES = 2


@dataclass
class GeminiResult:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 0


_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key or api_key.startswith("PUT_"):
            raise RuntimeError("GOOGLE_API_KEY not set in .env")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _client = genai.GenerativeModel(GEMINI_MODEL)
    return _client


def _messages_to_prompt(messages: list) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append("[SYSTEM INSTRUCTIONS]\n" + content)
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append("[Previous response]\n" + content)
    return "\n\n".join(parts)


async def chat(
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> GeminiResult:
    model = _get_client()
    prompt = _messages_to_prompt(messages)
    temp = 0.2 if temperature is None else temperature
    max_tok = 4000 if max_tokens is None else max_tokens

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            gen_config = {
                "temperature": temp,
                "max_output_tokens": max_tok,
            }
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=gen_config,
                ),
                timeout=60,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            text = ""
            try:
                text = resp.text or ""
            except Exception:
                if hasattr(resp, "candidates") and resp.candidates:
                    text = str(resp.candidates[0])[:5000]

            usage = getattr(resp, "usage_metadata", None)
            in_tok = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
            out_tok = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

            logger.info("gemini ok model=%s tokens=%s latency_ms=%s",
                        GEMINI_MODEL, in_tok + out_tok, latency_ms)
            return GeminiResult(
                content=text,
                model=GEMINI_MODEL,
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
                latency_ms=latency_ms,
                attempts=attempt,
            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("gemini timeout attempt=%d after 60s", attempt)
            raise RuntimeError("gemini_timeout_60s")
        except Exception as exc:
            last_exc = exc
            wait = min(2 ** attempt, 10)
            logger.warning("gemini error attempt=%d wait=%ds msg=%s",
                           attempt, wait, redact(exc))
            if attempt < MAX_RETRIES:
                await asyncio.sleep(wait)

    raise RuntimeError(
        "Gemini call failed after " + str(MAX_RETRIES) + " attempts: " + redact(last_exc)
    )