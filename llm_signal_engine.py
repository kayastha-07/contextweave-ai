"""
ContextWeave — llm_signal_engine.py
=====================================
LLM-powered behavioral signal extractor using Google Gemini.
Gemini is fully optional and controlled by environment variables.

DESIGN RULES:
  - Gemini extracts structured signals ONLY (no advice, no predictions)
  - All reasoning stays inside the existing behavioral engine pipeline
  - If Gemini is disabled or fails, keyword fallback runs immediately
  - No retries — one attempt, then fallback. Clean and fast.

ENVIRONMENT VARIABLES (.env file):
  GEMINI_API_KEY = your-api-key-here   (required to use Gemini)
  USE_GEMINI     = true                (set to false to disable Gemini entirely)

CONTROL:
  USE_GEMINI=true  + valid key  → Gemini is used, keyword is fallback
  USE_GEMINI=false              → keyword scoring always, Gemini never called
  USE_GEMINI=true  + no key     → keyword scoring, warning logged once

PUBLIC API:
    analyze_note_with_gemini(note_text: str) -> dict
    clear_cache()       — wipe memory + disk cache
    get_cache_stats()   — {"cached_notes", "hits", "misses"}

RETURN SCHEMA (guaranteed regardless of which path runs):
    {
        "cognitive_score" : int   (0–100),
        "action_count"    : int   (0–5),
        "delay_count"     : int   (0–5),
        "stress_signals"  : int   (0–5),
        "sentiment"       : str   ("positive" | "neutral" | "negative"),
        "themes"          : list[str]
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature flag
# Read once at startup. Change in .env and restart server to toggle.
# ─────────────────────────────────────────────────────────────────────────────

def _use_gemini_enabled() -> bool:
    """Return True only if USE_GEMINI env var is explicitly set to 'true'."""
    return os.getenv("USE_GEMINI", "false").strip().lower() == "true"


# ─────────────────────────────────────────────────────────────────────────────
# Gemini client — lazy init, only runs when USE_GEMINI=true
# ─────────────────────────────────────────────────────────────────────────────

_client:           object | None = None
_GEMINI_AVAILABLE: bool          = False
_GEMINI_MODEL                    = "gemini-2.0-flash-lite"


def _init_gemini() -> None:
    """
    Initialise the Gemini client.
    Called once at import time. Silently skipped when USE_GEMINI != true.
    """
    global _client, _GEMINI_AVAILABLE

    if not _use_gemini_enabled():
        logger.info(
            "[LLMSignal] USE_GEMINI=false — "
            "keyword fallback active, Gemini will not be called."
        )
        return

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "[LLMSignal] USE_GEMINI=true but GEMINI_API_KEY is not set — "
            "keyword fallback active."
        )
        return

    try:
        from google import genai          # type: ignore  (pip install google-genai)
        _client           = genai.Client(api_key=api_key)
        _GEMINI_AVAILABLE = True
        logger.info(
            f"[LLMSignal] Gemini ready — model: {_GEMINI_MODEL}"
        )
    except ImportError:
        logger.warning(
            "[LLMSignal] google-genai package not found. "
            "Run: pip install google-genai"
        )
    except Exception as exc:
        logger.warning(f"[LLMSignal] Gemini init failed: {exc}")


_init_gemini()


# ─────────────────────────────────────────────────────────────────────────────
# Extraction prompt
#
# Hard constraints in the text:
#   1. Return ONLY a raw JSON object — no prose, no markdown outside it
#   2. No advice, predictions, or recommendations
#   3. Integer counts bounded 0–5 so downstream math is always safe
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a behavioral signal extractor for a cognitive analytics system.

YOUR ONLY JOB: read the note and return ONE raw JSON object with structured
behavioral signals. No advice. No predictions. No text outside the JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cognitive_score  (integer 0–100)
  Estimated mental clarity and productive output.
  50 = neutral. Higher = more focus/completion. Lower = avoidance/stress.

action_count  (integer 0–5)
  Distinct productive-action signals: completing tasks, submitting work,
  studying actively, hitting goals, focused sessions, exercising.

delay_count  (integer 0–5)
  Distinct avoidance/procrastination signals: saying "later/tomorrow/soon",
  skipping plans, distracted by social media or games, postponing work.

stress_signals  (integer 0–5)
  Distinct stress/overwhelm signals: deadline pressure, exam anxiety,
  too many tasks, burnout language, urgency, fear of failure.

sentiment  (EXACTLY one of: "positive", "neutral", "negative")
  positive = energised, accomplished, motivated.
  negative = anxious, defeated, drained, frustrated.
  neutral  = factual, balanced, no clear emotional tone.

themes  (array of 1–4 short lowercase strings)
  Main topics. Choose from: study, work, health, fitness, deadline,
  meeting, project, social, sleep, focus, distraction, creativity,
  planning, recovery, exam, coding, reading, family, finance, personal-growth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETURN ONLY this JSON, nothing else:
{
  "cognitive_score": <int 0-100>,
  "action_count":    <int 0-5>,
  "delay_count":     <int 0-5>,
  "stress_signals":  <int 0-5>,
  "sentiment":       "<positive|neutral|negative>",
  "themes":          ["<theme1>", "<theme2>"]
}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Note to analyze:
{NOTE}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Persistent cache  (survives server restarts)
#
# Key  = MD5 of lowercased, stripped note text
# Rule = only successful Gemini responses are cached
#        keyword-fallback results are never cached
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_FILE             = "llm_signal_cache.json"
_cache: dict[str, dict] = {}
_hits                   = 0
_misses                 = 0


def _load_cache() -> None:
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r") as f:
                _cache.update(json.load(f))
            logger.info(
                f"[LLMSignal] Cache loaded — {len(_cache)} notes restored from disk."
            )
        except Exception:
            pass  # corrupt cache — start fresh, no crash


def _save_cache() -> None:
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(_cache, f, indent=2)
    except Exception as exc:
        logger.warning(f"[LLMSignal] Cache save failed: {exc}")


_load_cache()


def _make_key(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def clear_cache() -> None:
    """Wipe all cached signals from memory and disk."""
    global _hits, _misses
    _cache.clear()
    _hits   = 0
    _misses = 0
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)


def get_cache_stats() -> dict:
    return {
        "cached_notes": len(_cache),
        "hits":         _hits,
        "misses":       _misses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing + field validation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict:
    """Extract JSON from Gemini's raw response, handling markdown fences."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]+?\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON in response: {raw[:200]!r}")


def _validate(raw: dict) -> dict:
    """Coerce and clamp all fields. Downstream engines always get clean data."""

    def to_int(val, lo: int, hi: int, default: int) -> int:
        try:
            return max(lo, min(hi, int(float(str(val)))))
        except (TypeError, ValueError):
            return default

    sentiment = str(raw.get("sentiment", "neutral")).lower().strip()
    if sentiment not in {"positive", "neutral", "negative"}:
        sentiment = "neutral"

    raw_themes = raw.get("themes", [])
    themes = [
        str(t).lower().strip()
        for t in (raw_themes if isinstance(raw_themes, list) else [])
        if str(t).strip()
    ][:4]

    return {
        "cognitive_score": to_int(raw.get("cognitive_score"), 0, 100, 50),
        "action_count":    to_int(raw.get("action_count"),    0,   5,  0),
        "delay_count":     to_int(raw.get("delay_count"),     0,   5,  0),
        "stress_signals":  to_int(raw.get("stress_signals"),  0,   5,  0),
        "sentiment":       sentiment,
        "themes":          themes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Keyword fallback  (original main.py logic — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_fallback(text: str) -> dict:
    """Keyword-based signal extraction. Fast, offline, zero API calls."""
    t = text.lower()

    action_words = [
        "start", "finish", "complete", "submit", "prepare",
        "study", "done", "accomplished", "published", "deployed",
    ]
    delay_words = [
        "later", "tomorrow", "soon", "after",
        "postpone", "skip", "cancel", "scroll", "distract",
    ]
    stress_words = [
        "deadline", "urgent", "exam", "pressure", "overwhelmed",
        "anxious", "stressed", "burnout", "behind", "overdue",
    ]

    action = min(sum(w in t for w in action_words), 5)
    delay  = min(sum(w in t for w in delay_words),  5)
    stress = min(sum(w in t for w in stress_words), 5)
    score  = max(0, min(100, 50 + action * 5 - delay * 4 - stress * 3))

    if action > delay + stress:
        sentiment = "positive"
    elif delay + stress > action + 1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "cognitive_score": score,
        "action_count":    action,
        "delay_count":     delay,
        "stress_signals":  stress,
        "sentiment":       sentiment,
        "themes":          [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_note_with_gemini(note_text: str) -> dict:
    """
    Extract structured behavioral signals from a single note.

    Decision flow:
        1. Blank input         → neutral defaults (instant)
        2. Cache hit           → cached result (instant, zero API calls)
        3. USE_GEMINI=false    → keyword fallback (instant)
        4. Gemini unavailable  → keyword fallback (instant)
        5. Gemini call         → ONE attempt only, no retries
           ├─ success          → validate → cache → return
           └─ any failure      → keyword fallback (one clean log line)
    """
    global _hits, _misses

    # ── 1. Blank input ────────────────────────────────────────────────────────
    if not note_text or not note_text.strip():
        return {
            "cognitive_score": 50,
            "action_count":    0,
            "delay_count":     0,
            "stress_signals":  0,
            "sentiment":       "neutral",
            "themes":          [],
        }

    key = _make_key(note_text)

    # ── 2. Cache hit ──────────────────────────────────────────────────────────
    if key in _cache:
        _hits += 1
        return _cache[key]

    _misses += 1

    # ── 3 & 4. Feature flag / availability check ──────────────────────────────
    if not _GEMINI_AVAILABLE or _client is None:
        return _keyword_fallback(note_text)

    # ── 5. Single Gemini attempt — no retries ─────────────────────────────────
    try:
        prompt   = _EXTRACTION_PROMPT.replace("{NOTE}", note_text.strip())
        t0       = time.time()
        response = _client.models.generate_content(   # type: ignore
            model    = _GEMINI_MODEL,
            contents = prompt,
        )
        elapsed = time.time() - t0

        signals = _validate(_parse_response(response.text))

        logger.info(
            f"[LLMSignal] Gemini OK ({elapsed:.2f}s) — "
            f"score:{signals['cognitive_score']}  "
            f"A:{signals['action_count']}  "
            f"D:{signals['delay_count']}  "
            f"S:{signals['stress_signals']}  "
            f"sentiment:{signals['sentiment']}  "
            f"themes:{signals['themes']}"
        )

        _cache[key] = signals
        _save_cache()
        return signals

    except Exception:
        logger.warning("[LLMSignal] Gemini unavailable — using keyword fallback.")
        return _keyword_fallback(note_text)