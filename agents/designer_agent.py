"""
AI Software Company — Designer Agent (Step 9)

Architecture references:
  Section 3   Agent 3: Designer
  Section 5   Designer: Temp=0.4, JSON mode
              Output: {colors, fonts, spacing, responsive}
              Rule: 5 palettes, 60-30-10, mobile-first
              Validate: contrast >= 4.5 (WCAG AA)
  Section 8   5 palettes (exact hex values, mandatory set)
  Section 16  AppState keys: colors, fonts

Public API:
  async def run_designer_agent(pm_brief, plan, job_id, user_id, trace_id) -> dict
      returns {"colors": {...}, "fonts": {...}, "spacing": {...}, "responsive": {...}}
      raises DesignerAgentError on unrecoverable failure.

Design:
  - Pure orchestration.
  - WCAG AA contrast (>= 4.5) is enforced locally on the chosen palette's
    "text_on_background" pair. The 5 palettes in Section 8 already pass this,
    but we re-check so a model cannot silently regress.
  - Every color value is validated as #RRGGBB.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ai import groq_client
from config import groq_config as cfg
from observability.cost_tracker import record_call
from observability.logger import bind_job, get_logger
from security import prompt_guard as pg

logger = get_logger("agents.designer")

MIN_CONTRAST = 4.5
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

# --- Section 8 palettes (reference, used inside the system prompt) -------
SECTION_8_PALETTES = [
    {"name": "Fintech",  "bg": "#FFFFFF", "text": "#0F172A", "accent": "#2563EB"},
    {"name": "Tech",     "bg": "#0B0F19", "text": "#F8FAFC", "accent": "#818CF8"},
    {"name": "Health",   "bg": "#FAFAF9", "text": "#1E3A8A", "accent": "#047857"},
    {"name": "Creative", "bg": "#FFFFFF", "text": "#1E1E1E", "accent": "#BE123C"},
    {"name": "Luxury",   "bg": "#121212", "text": "#E2E8F0", "accent": "#B8860B"},
]

# --- WCAG contrast math --------------------------------------------------
def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _relative_luminance(hex_color: str) -> float:
    r, g, b = _hex_to_rgb(hex_color)
    def _chan(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4
    return 0.2126 * _chan(r) + 0.7152 * _chan(g) + 0.0722 * _chan(b)


def contrast_ratio(fg: str, bg: str) -> float:
    l1 = _relative_luminance(fg)
    l2 = _relative_luminance(bg)
    lighter, darker = (l1, l2) if l1 >= l2 else (l2, l1)
    return (lighter + 0.05) / (darker + 0.05)


# --- System prompt -------------------------------------------------------
_PALETTES_TEXT = "\n".join(
    f'  - {p["name"]}: bg={p["bg"]}, text={p["text"]}, accent={p["accent"]}'
    for p in SECTION_8_PALETTES
)

DESIGNER_SYSTEM_PROMPT = f"""You are the Designer agent in an AI software company.

You receive a product brief and a technical plan. Produce a design brief as a
single JSON object with exactly these keys:

{{
  "colors": {{
    "palettes": [
      {{"name": "<one of: Fintech|Tech|Health|Creative|Luxury>",
        "bg": "#RRGGBB", "text": "#RRGGBB", "accent": "#RRGGBB",
        "ratio_60": "bg", "ratio_30": "text", "ratio_10": "accent"}}
    ],
    "chosen": "<one of the palette names>"
  }},
  "fonts": {{
    "heading": "<Google Fonts family name>",
    "body": "<Google Fonts family name>",
    "base_size_px": 16
  }},
  "spacing": {{
    "unit_px": 8,
    "radius_px": 8
  }},
  "responsive": {{
    "breakpoints_px": [640, 768, 1024],
    "touch_target_min_px": 44,
    "font_min_px": 16
  }}
}}

STRICT RULES:
- "palettes" MUST contain EXACTLY these 5 palettes, with EXACTLY these hex values:
{_PALETTES_TEXT}
- You MUST NOT invent new colors or change any hex value.
- "chosen" MUST be one of: Fintech, Tech, Health, Creative, Luxury.
- "fonts.heading" and "fonts.body": real Google Fonts family names
  (e.g., "Inter", "Roboto", "Poppins", "Source Sans 3").
- "fonts.base_size_px" MUST be >= 16.
- "spacing.unit_px" MUST be a positive integer (recommended 8).
- "spacing.radius_px" MUST be a positive integer.
- "responsive.breakpoints_px" MUST be exactly [640, 768, 1024].
- "responsive.touch_target_min_px" MUST be >= 44.
- "responsive.font_min_px" MUST be >= 16.
- Output MUST be valid JSON: double-quoted keys/strings, no trailing commas,
  no comments, no code fences.
- Return ONLY the JSON object.
"""

DESIGNER_REPAIR_PROMPT = """Your previous response was not valid.

Return ONLY a single valid JSON object with exactly these keys:

{"colors": {"palettes": [{"name": "<Fintech|Tech|Health|Creative|Luxury>", "bg": "#RRGGBB", "text": "#RRGGBB", "accent": "#RRGGBB", "ratio_60": "bg", "ratio_30": "text", "ratio_10": "accent"}], "chosen": "<name>"}, "fonts": {"heading": "<name>", "body": "<name>", "base_size_px": 16}, "spacing": {"unit_px": 8, "radius_px": 8}, "responsive": {"breakpoints_px": [640, 768, 1024], "touch_target_min_px": 44, "font_min_px": 16}}

Rules: exactly 5 palettes with the exact Section 8 hex values, chosen in {Fintech, Tech, Health, Creative, Luxury}. Return ONLY valid JSON.
"""


class DesignerAgentError(RuntimeError):
    """Raised when the Designer agent cannot produce a valid brief."""


# --- Validation ----------------------------------------------------------
def _validate_palettes(data: Any) -> tuple[bool, str, dict]:
    if not isinstance(data, dict):
        return False, "colors_not_object", {}
    palettes = data.get("palettes")
    chosen = data.get("chosen")
    if not isinstance(palettes, list):
        return False, "palettes_not_list", {}
    if len(palettes) != 5:
        return False, f"palettes_count={len(palettes)} (need exactly 5)", {}

    names_seen: set[str] = set()
    for p in palettes:
        if not isinstance(p, dict):
            return False, "palette_not_object", {}
        name = str(p.get("name", "")).strip()
        if name not in {sp["name"] for sp in SECTION_8_PALETTES}:
            return False, f"palette_name_invalid:{name}", {}
        if name in names_seen:
            return False, f"palette_duplicate:{name}", {}
        names_seen.add(name)

        for key in ("bg", "text", "accent"):
            v = str(p.get(key, "")).strip()
            if not _HEX_RE.match(v):
                return False, f"palette_hex_bad:{name}:{key}={v}", {}

        # WCAG AA on text-on-bg pair for this palette
        ratio = contrast_ratio(p["text"], p["bg"])
        if ratio < MIN_CONTRAST:
            return False, f"palette_contrast_low:{name}={ratio:.2f}", {}

    if not isinstance(chosen, str) or chosen.strip() not in names_seen:
        return False, f"chosen_invalid:{chosen}", {}

    return True, "ok", {
        "palettes": [
            {
                "name": p["name"].strip(),
                "bg": p["bg"].strip(),
                "text": p["text"].strip(),
                "accent": p["accent"].strip(),
                "ratio_60": p.get("ratio_60", "bg"),
                "ratio_30": p.get("ratio_30", "text"),
                "ratio_10": p.get("ratio_10", "accent"),
            }
            for p in palettes
        ],
        "chosen": chosen.strip(),
    }


def _validate_fonts(data: Any) -> tuple[bool, str, dict]:
    if not isinstance(data, dict):
        return False, "fonts_not_object", {}
    heading = str(data.get("heading", "")).strip()
    body = str(data.get("body", "")).strip()
    base = data.get("base_size_px", 16)
    if not heading or not body:
        return False, "font_family_empty", {}
    try:
        base_i = int(base)
    except Exception:
        return False, "base_size_bad", {}
    if base_i < 16:
        return False, f"base_size_too_small:{base_i}", {}
    return True, "ok", {"heading": heading, "body": body, "base_size_px": base_i}


def _validate_spacing(data: Any) -> tuple[bool, str, dict]:
    if not isinstance(data, dict):
        return False, "spacing_not_object", {}
    try:
        unit = int(data.get("unit_px", 8))
        radius = int(data.get("radius_px", 8))
    except Exception:
        return False, "spacing_bad_int", {}
    if unit <= 0 or radius <= 0:
        return False, "spacing_nonpositive", {}
    return True, "ok", {"unit_px": unit, "radius_px": radius}


def _validate_responsive(data: Any) -> tuple[bool, str, dict]:
    if not isinstance(data, dict):
        return False, "responsive_not_object", {}
    bps = data.get("breakpoints_px")
    if bps != [640, 768, 1024]:
        return False, f"breakpoints_bad:{bps}", {}
    try:
        touch = int(data.get("touch_target_min_px", 44))
        fmin = int(data.get("font_min_px", 16))
    except Exception:
        return False, "responsive_bad_int", {}
    if touch < 44:
        return False, f"touch_too_small:{touch}", {}
    if fmin < 16:
        return False, f"font_min_too_small:{fmin}", {}
    return True, "ok", {
        "breakpoints_px": [640, 768, 1024],
        "touch_target_min_px": touch,
        "font_min_px": fmin,
    }


def _validate_brief(data: Any) -> tuple[bool, str, dict]:
    if not isinstance(data, dict):
        return False, "not_object", {}

    ok, reason, colors = _validate_palettes(data.get("colors"))
    if not ok:
        return False, reason, {}

    ok, reason, fonts = _validate_fonts(data.get("fonts"))
    if not ok:
        return False, reason, {}

    ok, reason, spacing = _validate_spacing(data.get("spacing"))
    if not ok:
        return False, reason, {}

    ok, reason, responsive = _validate_responsive(data.get("responsive"))
    if not ok:
        return False, reason, {}

    return True, "ok", {
        "colors": colors,
        "fonts": fonts,
        "spacing": spacing,
        "responsive": responsive,
    }


# --- Main entry point ---------------------------------------------------
async def run_designer_agent(
    pm_brief: dict,
    plan: dict,
    job_id: str,
    user_id: str,
    trace_id: str,
) -> dict:
    """
    Run the Designer agent: (PM brief + Architect plan) -> design brief.
    """
    log = bind_job(logger, job_id=job_id, user_id=user_id, trace_id=trace_id)
    log.info("agent_started", agent="designer")

    combined = {
        "pm_brief": pm_brief,
        "technical_plan": plan,
    }
    combined_text = json.dumps(combined, ensure_ascii=False, indent=2)
    san = pg.sanitize_user_input(combined_text)
    if not san.ok:
        log.warning("designer_input_rejected", reason=san.reason)
        raise DesignerAgentError(f"invalid_input:{san.reason}")

    messages = pg.build_messages(DESIGNER_SYSTEM_PROMPT, san.cleaned)

    # --- First attempt ---
    try:
        result = await groq_client.chat(
            messages,
            temperature=0.4,
            max_tokens=cfg.PRIMARY_MAX_TOKENS,
        )
    except Exception as exc:
        log.error("designer_groq_failed", reason=str(exc))
        raise DesignerAgentError("groq_failed") from exc

    try:
        record_call(
            job_id=job_id, agent="designer", model=result.model,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            total_tokens=result.total_tokens, latency_ms=result.latency_ms,
            rag_tokens=0, used_fallback=result.used_fallback,
            attempts=result.attempts,
        )
    except Exception:
        pass

    checked = pg.validate_json_output(result.content, required_keys=[])
    if checked.ok:
        ok, reason, brief = _validate_brief(checked.data)
        if ok:
            log.info("agent_completed", agent="designer",
                     chosen=brief["colors"]["chosen"])
            return brief
        else:
            log.warning("designer_schema_invalid", reason=reason)
            last_reason = reason
    else:
        log.warning("designer_json_invalid", reason=checked.reason)
        last_reason = checked.reason

    # --- Repair attempts (max 2) ---
    for repair_attempt in range(1, 3):
        repair_messages = [
            {"role": "system", "content": DESIGNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"<user_input>\n{san.cleaned}\n</user_input>"},
            {"role": "assistant", "content": result.content[:2000]},
            {"role": "user", "content": DESIGNER_REPAIR_PROMPT},
        ]
        try:
            result = await groq_client.chat(
                repair_messages,
                temperature=0.2,
                max_tokens=cfg.PRIMARY_MAX_TOKENS,
            )
        except Exception as exc:
            log.error("designer_repair_groq_failed",
                      reason=str(exc), attempt=repair_attempt)
            raise DesignerAgentError("groq_failed_on_repair") from exc

        try:
            record_call(
                job_id=job_id, agent="designer", model=result.model,
                input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                total_tokens=result.total_tokens, latency_ms=result.latency_ms,
                rag_tokens=0, used_fallback=result.used_fallback,
                attempts=result.attempts,
            )
        except Exception:
            pass

        checked = pg.validate_json_output(result.content, required_keys=[])
        if checked.ok:
            ok, reason, brief = _validate_brief(checked.data)
            if ok:
                log.info("agent_completed", agent="designer",
                         repair_attempts=repair_attempt,
                         chosen=brief["colors"]["chosen"])
                return brief
            last_reason = reason
        else:
            last_reason = checked.reason

    log.error("designer_failed", reason=last_reason)
    raise DesignerAgentError(f"designer_validation_failed:{last_reason}")


# --- Convenience ---------------------------------------------------------
def pretty(brief: dict) -> str:
    """Return a readable string of the design brief (for CLI tests)."""
    return json.dumps(brief, indent=2, ensure_ascii=False)