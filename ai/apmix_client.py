"""Apmix.ai async client - OpenAI-compatible. Primary coding model (Claude Sonnet 4.6 free)."""
from __future__ import annotations

import os
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ai.apmix_client")

APMIX_BASE_URL = os.getenv("APMIX_BASE_URL", "https://api.apmix.ai/v1").rstrip("/")
APMIX_API_KEY = os.getenv("APMIX_API_KEY", "").strip()
APMIX_MODEL = os.getenv("APMIX_MODEL", "claude-sonnet-4-6-free")
MAX_RETRIES = 2


@dataclass
class ApmixResult:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 0
    used_fallback: bool = False




def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` fences and prose around JSON."""
    if not text:
        return text
    s = text.strip()
    if s.startswith("```"):
        i = s.find("\n")
        if i >= 0:
            s = s[i + 1:]
        else:
            s = s[3:]
        s = s.rstrip()
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # If prose around, extract outermost { ... } or [ ... ]
    if s and not s.startswith("{") and not s.startswith("["):
        fb = s.find("{")
        lb = s.rfind("}")
        if fb >= 0 and lb > fb:
            s = s[fb:lb + 1]
    return s


def is_configured() -> bool:
    return bool(APMIX_API_KEY) and APMIX_API_KEY.startswith("apx_")


async def chat(messages: list, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> ApmixResult:
    if not is_configured():
        raise RuntimeError("APMIX_API_KEY not configured")
    temp = 0.1 if temperature is None else temperature
    max_tok = 8000 if max_tokens is None else max_tokens

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    APMIX_BASE_URL + "/chat/completions",
                    headers={
                        "Authorization": "Bearer " + APMIX_API_KEY,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": APMIX_MODEL,
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": max_tok,
                    },
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                last_exc = f"apmix http {r.status_code}: {r.text[:200]}"
                logger.warning("apmix http %d attempt=%d body=%s",
                               r.status_code, attempt, r.text[:200])
                continue
            data = r.json()
            choices = data.get("choices") or []
            content = ""
            if choices:
                content = (choices[0].get("message") or {}).get("content") or ""
            usage = data.get("usage") or {}
            in_tok = int(usage.get("prompt_tokens", 0) or 0)
            out_tok = int(usage.get("completion_tokens", 0) or 0)
            logger.info("apmix ok model=%s tokens=%d latency_ms=%d",
                        data.get("model", APMIX_MODEL), in_tok + out_tok, latency_ms)
            content = _strip_json_fences(content)
            return ApmixResult(
                content=content,
                model=data.get("model", APMIX_MODEL),
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
                latency_ms=latency_ms,
                attempts=attempt,
                used_fallback=False,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning("apmix error attempt=%d msg=%s", attempt, str(exc)[:200])
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2)

    raise RuntimeError(f"Apmix call failed after {MAX_RETRIES} attempts: {last_exc}")
