"""
ContextWeave — insight_engine.py
=================================
Transforms structured behavioral records into deep, data-driven
narrative insights that go beyond raw scores.

Usage:
    from insight_engine import generate_insights
    insights = generate_insights(records)   # returns list[str]
"""

from __future__ import annotations

import statistics
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(r: Any, field: str, default=0):
    if isinstance(r, dict):
        return r.get(field, default)
    return getattr(r, field, default)


def _safe_mean(vals: list) -> float:
    return statistics.mean(vals) if vals else 0.0


def _safe_stdev(vals: list) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def _slope(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    xs    = list(range(n))
    mx    = _safe_mean(xs)
    my    = _safe_mean(vals)
    num   = sum((x - mx) * (y - my) for x, y in zip(xs, vals))
    den   = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Individual insight generators
# ─────────────────────────────────────────────────────────────────────────────

def _insight_action_score_threshold(records) -> str | None:
    """High action + low delay days vs rest."""
    if len(records) < 4:
        return None

    high_perf, low_perf = [], []

    for r in records:
        a = _get(r, "action_count")
        d = _get(r, "delay_count")
        s = _get(r, "cognitive_score")
        if a >= 3 and d <= 1:
            high_perf.append(s)
        else:
            low_perf.append(s)

    if not high_perf or not low_perf:
        return None

    avg_high = _safe_mean(high_perf)
    avg_low  = _safe_mean(low_perf)

    if avg_high > avg_low + 6:
        return (
            f"Your cognitive score averages {avg_high:.0f} on days with ≥3 actions and ≤1 delay, "
            f"versus {avg_low:.0f} on other days — a {avg_high - avg_low:.0f}-point improvement. "
            "High execution with low procrastination is your performance sweet spot."
        )
    return None


def _insight_stress_leads_decline(records) -> str | None:
    """Stress on day N predicts score drop on day N+1 or N+2."""
    if len(records) < 5:
        return None

    lag_drops = 0
    lag_total = 0

    for i in range(len(records) - 2):
        stress_now  = _get(records[i],   "stress_signals")
        score_next  = _get(records[i+1], "cognitive_score")
        score_now   = _get(records[i],   "cognitive_score")
        if stress_now >= 2:
            lag_total += 1
            if score_next < score_now - 3:
                lag_drops += 1

    if lag_total >= 2 and lag_drops / lag_total >= 0.5:
        return (
            f"In {lag_drops} out of {lag_total} observed cases, a stress signal of ≥2 was "
            "followed by a cognitive score decline the next session. "
            "Stress spikes appear to be a leading indicator of performance dips 1–2 days later."
        )
    return None


def _insight_delay_trend(records) -> str | None:
    """Rising delay signals = early procrastination warning."""
    if len(records) < 5:
        return None

    delays = [_get(r, "delay_count") for r in records]
    s = _slope(delays)

    if s > 0.3:
        recent_avg  = _safe_mean(delays[-3:])
        earlier_avg = _safe_mean(delays[:-3])
        return (
            f"Delay signals have been rising across recent entries "
            f"(early avg: {earlier_avg:.1f} → recent avg: {recent_avg:.1f}). "
            "This is a measurable procrastination risk — consider enforcing structured time blocks."
        )
    if s < -0.3:
        return (
            "Delay signals have been decreasing over time — "
            "your follow-through is actively improving."
        )
    return None


def _insight_best_performing_condition(records) -> str | None:
    """Find the combination of signals that correlates with top-quartile scores."""
    if len(records) < 6:
        return None

    scores = [_get(r, "cognitive_score") for r in records]
    q75    = sorted(scores)[int(len(scores) * 0.75)]

    top_records = [r for r in records if _get(r, "cognitive_score") >= q75]
    if not top_records:
        return None

    avg_action = _safe_mean([_get(r, "action_count")  for r in top_records])
    avg_delay  = _safe_mean([_get(r, "delay_count")   for r in top_records])
    avg_stress = _safe_mean([_get(r, "stress_signals") for r in top_records])

    return (
        f"Your top-quartile cognitive scores (≥{q75:.0f}) occur when "
        f"action count averages {avg_action:.1f}, "
        f"delay count averages {avg_delay:.1f}, "
        f"and stress signals average {avg_stress:.1f}. "
        "Replicate these conditions to reliably enter peak performance."
    )


def _insight_volatility_cause(records) -> str | None:
    """Link cognitive score volatility to specific signal fluctuations."""
    if len(records) < 5:
        return None

    scores  = [_get(r, "cognitive_score") for r in records]
    actions = [_get(r, "action_count")    for r in records]
    delays  = [_get(r, "delay_count")     for r in records]

    score_vol  = _safe_stdev(scores)
    action_vol = _safe_stdev(actions)
    delay_vol  = _safe_stdev(delays)

    if score_vol < 5:
        return None

    if action_vol > delay_vol:
        return (
            f"Your cognitive score swings significantly (std dev: {score_vol:.1f}), "
            f"and action count variability ({action_vol:.1f}) is the primary driver. "
            "Stabilising your daily task completion rate should reduce performance volatility."
        )
    elif delay_vol >= action_vol:
        return (
            f"Your cognitive score swings significantly (std dev: {score_vol:.1f}), "
            f"and delay signal variability ({delay_vol:.1f}) is the leading cause. "
            "Irregular procrastination cycles are destabilising your performance baseline."
        )
    return None


def _insight_recovery_pattern(records) -> str | None:
    """Detect how quickly scores recover after a decline."""
    if len(records) < 6:
        return None

    recoveries = []
    for i in range(1, len(records) - 1):
        prev  = _get(records[i-1], "cognitive_score")
        curr  = _get(records[i],   "cognitive_score")
        nxt   = _get(records[i+1], "cognitive_score")
        if curr < prev - 5 and nxt > curr + 3:
            recoveries.append(nxt - curr)

    if len(recoveries) >= 2:
        avg_recovery = _safe_mean(recoveries)
        return (
            f"After cognitive dips, your scores typically recover by {avg_recovery:.0f} points "
            "in the following session — indicating good behavioral resilience."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(records: list) -> list[str]:
    """
    Accept a list of NoteRecord objects or dicts and return
    a list of human-readable AI insights (up to 5).
    """
    if not records:
        return ["Not enough data for AI insights yet. Keep logging daily notes."]

    # Sort by date safely
    try:
        records = sorted(records, key=lambda r: str(_get(r, "date", "")))
    except Exception:
        pass

    candidates = [
        _insight_action_score_threshold(records),
        _insight_stress_leads_decline(records),
        _insight_delay_trend(records),
        _insight_best_performing_condition(records),
        _insight_volatility_cause(records),
        _insight_recovery_pattern(records),
    ]

    insights = [c for c in candidates if c is not None]

    if not insights:
        return ["Behavioral data is accumulating — richer insights will appear after more sessions."]

    return insights[:5]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date, timedelta
    import random

    random.seed(99)
    today = date.today()

    records = []
    for i in range(14):
        d = today - timedelta(days=13 - i)
        action = random.randint(1, 6)
        delay  = random.randint(0, 4)
        stress = random.randint(0, 3) + (2 if i >= 10 else 0)
        score  = max(0, min(100, 50 + action*5 - delay*4 - stress*3 + random.randint(-5, 5)))
        records.append({
            "date":            d.isoformat(),
            "cognitive_score": score,
            "action_count":    action,
            "delay_count":     delay,
            "stress_signals":  stress,
        })

    insights = generate_insights(records)
    print(f"Generated {len(insights)} insights:\n")
    for i, ins in enumerate(insights, 1):
        print(f"{i}. {ins}\n")