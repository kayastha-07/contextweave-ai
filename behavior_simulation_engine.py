"""
ContextWeave — behavior_simulation_engine.py
=============================================
Parses a free-text hypothetical scenario and estimates:
  - predicted cognitive score
  - risk level
  - recommended action

Reuses the same scoring weights as the existing behavioral engine
so simulation results are consistent with real reports.

Usage:
    from behavior_simulation_engine import simulate_scenario
    result = simulate_scenario("I slept 4 hours and have 6 meetings tomorrow")
"""

from __future__ import annotations

import re
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Signal extraction from free text
# ─────────────────────────────────────────────────────────────────────────────

# Negative signals — things that lower cognitive score / raise risk
_STRESS_SIGNALS = [
    "meeting", "meetings", "deadline", "deadlines", "exam", "exams",
    "urgent", "pressure", "overdue", "late", "behind", "crisis",
    "overwhelmed", "anxious", "anxiety", "stressed", "stress",
    "presentation", "interview", "review", "audit",
]

_SLEEP_DEPRIVATION_PATTERNS = [
    (r"\b([1-4])\s*hours?\s*(of\s*)?sleep", "low"),   # 1–4 hrs → heavy penalty
    (r"\b([5-6])\s*hours?\s*(of\s*)?sleep", "medium"), # 5–6 hrs → moderate penalty
    (r"\b([7-9])\s*hours?\s*(of\s*)?sleep", "good"),   # 7–9 hrs → no penalty
]

_DELAY_SIGNALS = [
    "postpone", "delay", "later", "tomorrow", "reschedule",
    "skip", "cancel", "avoid", "procrastinat",
]

# Positive signals — things that boost cognitive score
_ACTION_SIGNALS = [
    "finish", "complete", "done", "submit", "start", "plan",
    "prepared", "ready", "focused", "gym", "exercise", "workout",
    "meditat", "rest", "sleep well", "slept well", "healthy",
    "organised", "organized", "scheduled",
]

_HIGH_MEETING_PATTERN = re.compile(r"\b([4-9]|1[0-9])\s*(meetings?|calls?|sessions?)\b", re.I)
_LOW_SLEEP_PATTERN    = re.compile(r"\b([1-4])\s*hours?\s*(of\s*)?sleep\b", re.I)
_MED_SLEEP_PATTERN    = re.compile(r"\b([5-6])\s*hours?\s*(of\s*)?sleep\b", re.I)


def _count_signals(text: str, signals: list[str]) -> int:
    text = text.lower()
    return sum(1 for s in signals if s in text)


def _extract_features(scenario: str) -> dict[str, int | float]:
    text = scenario.lower()

    stress  = _count_signals(text, _STRESS_SIGNALS)
    delay   = _count_signals(text, _DELAY_SIGNALS)
    action  = _count_signals(text, _ACTION_SIGNALS)

    # Sleep penalty
    sleep_penalty = 0
    if _LOW_SLEEP_PATTERN.search(text):
        sleep_penalty = 20
    elif _MED_SLEEP_PATTERN.search(text):
        sleep_penalty = 10

    # Heavy meeting load
    meeting_penalty = 0
    m = _HIGH_MEETING_PATTERN.search(text)
    if m:
        n_meetings = int(m.group(1))
        meeting_penalty = min(n_meetings * 2, 16)   # cap at 16

    return {
        "stress":          stress,
        "delay":           delay,
        "action":          action,
        "sleep_penalty":   sleep_penalty,
        "meeting_penalty": meeting_penalty,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Score + risk computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(features: dict) -> int:
    score = 50
    score += features["action"]  * 5
    score -= features["stress"]  * 3
    score -= features["delay"]   * 4
    score -= features["sleep_penalty"]
    score -= features["meeting_penalty"]
    return max(0, min(100, score))


def _compute_risk(score: int, features: dict) -> str:
    if score < 35 or features["sleep_penalty"] >= 20:
        return "High"
    if score < 55 or features["stress"] >= 3:
        return "Moderate"
    return "Low"


def _build_recommendation(score: int, risk: str, features: dict) -> str:
    parts = []

    if features["sleep_penalty"] >= 20:
        parts.append("Prioritize sleep — operating on under 5 hours severely impairs decision-making.")

    if features["meeting_penalty"] >= 8:
        parts.append("Heavy meeting load detected — block at least two focus periods between calls.")

    if features["stress"] >= 3:
        parts.append("Multiple stress signals found — consider deferring non-critical tasks.")

    if features["delay"] >= 2:
        parts.append("Delay patterns present — set one firm commitment for the day to anchor momentum.")

    if features["action"] >= 3 and risk == "Low":
        parts.append("Strong execution signals — this looks like a productive day ahead.")

    if not parts:
        if score >= 65:
            parts.append("Conditions look favorable. Maintain your current routine and stay consistent.")
        elif score >= 45:
            parts.append("Moderate conditions. Reduce cognitive load where possible and protect focus time.")
        else:
            parts.append("Challenging conditions ahead. Focus on one key task and allow recovery time.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def simulate_scenario(scenario: str) -> dict[str, Any]:
    """
    Parse a free-text hypothetical scenario and return a simulation result.

    Returns:
        {
            "predicted_score":  int   (0–100),
            "risk":             str   ("Low" | "Moderate" | "High"),
            "recommendation":   str,
            "signals_detected": dict  (breakdown for transparency)
        }
    """
    if not scenario or not scenario.strip():
        return {
            "predicted_score":  50,
            "risk":             "Unknown",
            "recommendation":   "Please describe a scenario to simulate.",
            "signals_detected": {},
        }

    features = _extract_features(scenario)
    score    = _compute_score(features)
    risk     = _compute_risk(score, features)
    rec      = _build_recommendation(score, risk, features)

    return {
        "predicted_score":  score,
        "risk":             risk,
        "recommendation":   rec,
        "signals_detected": {
            "stress_signals":    features["stress"],
            "delay_signals":     features["delay"],
            "action_signals":    features["action"],
            "sleep_penalty":     features["sleep_penalty"],
            "meeting_penalty":   features["meeting_penalty"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    tests = [
        "I slept 4 hours and have 6 meetings tomorrow",
        "Finished my project, slept 8 hours, feeling great",
        "Deadline tomorrow, behind on work, feeling stressed",
        "Going to the gym, planned my day, 2 meetings",
    ]

    for t in tests:
        r = simulate_scenario(t)
        print(f"Scenario : {t}")
        print(f"Score    : {r['predicted_score']}  Risk: {r['risk']}")
        print(f"Advice   : {r['recommendation']}")
        print()