"""OpenRouter async client - dynamically fetches current free models."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ai.openrouter")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
MAX_RETRIES = 2

# Cache for free model IDs (refreshed once per session)
_free_models_cache: list[str] | None = None
_cache_lock = asyncio.Lock()


@dataclass
class ORResult:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 0


def _get_key() -> str:
    k = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not k or not k.startswith("sk-or"):
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")
    return k


async def _fetch_free_models() -> list[str]:
    """Fetch current free model IDs from OpenRouter API."""
    global _free_models_cache
    async with _cache_lock:
        if _free_models_cache is not None:
            return _free_models_cache

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(OPENROUTER_MODELS_URL)
                r.raise_for_status()
                data = r.json().get("data", [])

            # Prefer coding-capable models with :free suffix
            preferred_keywords = ["coder", "deepseek", "qwen3", "llama-3.3", "gpt-oss"]
            free_ids = []
            other_free = []
            for m in data:
                mid = m.get("id", "")
                pricing = m.get("pricing", {}) or {}
                # Free if both prompt and completion are "0"
                is_free = (str(pricing.get("prompt", "")) == "0" and
                           str(pricing.get("completion", "")) == "0")
                if not is_free or ":free" not in mid:
                    continue
                if any(kw in mid.lower() for kw in preferred_keywords):
                    free_ids.append(mid)
                else:
                    other_free.append(mid)

            _free_models_cache = free_ids + other_free
            logger.info("openrouter_free_models_fetched count=%d",
                        len(_free_models_cache))
            return _free_models_cache
        except Exception as exc:
            logger.warning("openrouter_free_models_fetch_failed err=%s",
                           type(exc).__name__)
            # Fallback to router which auto-picks a free model
            _free_models_cache = ["openrouter/free"]
            return _free_models_cache


async def chat(
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ORResult:
    key = _get_key()
    temp = 0.2 if temperature is None else temperature
    max_tok = 4000 if max_tokens is None else max_tokens

    models = await _fetch_free_models()
    if not models:
        models = ["openrouter/free"]

    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Software Company",
    }

    last_exc = None
    for model in models:
        body = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
        }
        for attempt in range(1, MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    r = await client.post(OPENROUTER_URL,
                                          headers=headers, json=body)
                latency_ms = int((time.perf_counter() - t0) * 1000)

                if r.status_code == 429:
                    logger.warning("openrouter 429 model=%s", model)
                    break
                if r.status_code >= 400:
                    logger.warning("openrouter http %d model=%s body=%s",
                                   r.status_code, model, r.text[:150])
                    break

                data = r.json()
                choice = data.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content", "") or ""
                usage = data.get("usage", {}) or {}
                in_tok = int(usage.get("prompt_tokens", 0) or 0)
                out_tok = int(usage.get("completion_tokens", 0) or 0)

                logger.info("openrouter ok model=%s tokens=%d",
                            model, in_tok + out_tok)
                return ORResult(
                    content=text, model=model,
                    input_tokens=in_tok, output_tokens=out_tok,
                    total_tokens=in_tok + out_tok,
                    latency_ms=latency_ms, attempts=attempt,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning("openrouter err model=%s err=%s",
                               model, type(exc).__name__)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)

    raise RuntimeError("OpenRouter failed all free models: " + str(last_exc))