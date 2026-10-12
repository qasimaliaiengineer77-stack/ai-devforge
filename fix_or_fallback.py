"""Add OpenRouter as final fallback in groq_client.py."""
from pathlib import Path

p = Path("ai/groq_client.py")
s = p.read_text(encoding="utf-8")

# Find the final raise and replace with OpenRouter fallback + raise
old = '''    raise RuntimeError(
        f"All providers failed. Groq: {redact(last_exc)}; Gemini: {redact(gexc)}"
    )'''

new = '''    # Try OpenRouter as final fallback
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
    )'''

if old in s:
    s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    print("FIXED: OpenRouter fallback added")
elif "openrouter_fallback_failed" in s:
    print("Already patched")
else:
    print("ANCHOR NOT FOUND — searching for alternative...")
    # Try alternative anchor
    import re
    m = re.search(r'raise RuntimeError\(\s*f"All providers failed\.', s)
    if m:
        # Find the end of this raise block
        end = s.find(")", m.start() + 20)
        if end != -1:
            end = s.find("\n", end) + 1
            s = s[:m.start()] + new + "\n" + s[end:]
            p.write_text(s, encoding="utf-8")
            print("FIXED via regex fallback")
    else:
        print("MANUAL FIX NEEDED")