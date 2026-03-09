"""
ContextWeave — behavior_pattern_engine.py
==========================================
Analyzes historical NoteRecord objects and detects human-readable
behavioral patterns such as procrastination cycles, stress spikes,
action/delay imbalance, cognitive score variability, and anomalies.

Usage:
    from behavior_pattern_engine import discover_behavior_patterns
    patterns = discover_behavior_patterns(records)  # returns list[str]
"""

from __future__ import annotations

import statistics
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(record: Any, field: str, default=0):
    """Safely read a field from either a NoteRecord object or a dict."""
    if isinstance(record, dict):
        return record.get(field, default)
    return getattr(record, field, default)


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _linear_slope(values: list[float]) -> float:
    """Simple OLS slope — positive means rising, negative means falling."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = _safe_mean(xs)
    my = _safe_mean(values)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Individual pattern detectors
# ─────────────────────────────────────────────────────────────────────────────

def _detect_action_delay_imbalance(records) -> str | None:
    """
    Checks whether delay signals consistently outweigh action signals
    across recent records — a procrastination indicator.
    """
    recent = records[-7:] if len(records) >= 7 else records

    action_totals = [_get(r, "action_count") for r in recent]
    delay_totals  = [_get(r, "delay_count")  for r in recent]

    avg_action = _safe_mean(action_totals)
    avg_delay  = _safe_mean(delay_totals)

    if avg_delay == 0 and avg_action == 0:
        return None

    ratio = avg_delay / (avg_action + 1)  # +1 avoids division by zero

    if ratio >= 1.5:
        return (
            f"Delay signals are significantly outweighing action signals over the past "
            f"{len(recent)} days (delay avg: {avg_delay:.1f} vs action avg: {avg_action:.1f}), "
            "indicating a developing procrastination cycle."
        )

    if avg_action > avg_delay * 1.5:
        return (
            f"Action signals are consistently stronger than delay signals "
            f"(action avg: {avg_action:.1f} vs delay avg: {avg_delay:.1f}). "
            "Execution momentum is healthy."
        )

    return None


def _detect_stress_before_score_decline(records) -> str | None:
    """
    Checks if stress signals rose in the window just before cognitive
    score started declining — early warning pattern.
    """
    if len(records) < 5:
        return None

    # split into first half and second half
    mid = len(records) // 2

    early_stress = _safe_mean([_get(r, "stress_signals") for r in records[:mid]])
    late_stress  = _safe_mean([_get(r, "stress_signals") for r in records[mid:]])

    early_score  = _safe_mean([_get(r, "cognitive_score") for r in records[:mid]])
    late_score   = _safe_mean([_get(r, "cognitive_score") for r in records[mid:]])

    stress_rose  = late_stress  > early_stress + 0.5
    score_fell   = late_score   < early_score  - 3

    if stress_rose and score_fell:
        return (
            f"Stress signals rose from an average of {early_stress:.1f} to {late_stress:.1f} "
            f"while cognitive scores declined from {early_score:.1f} to {late_score:.1f}. "
            "Elevated stress appears to be dampening cognitive performance."
        )

    if stress_rose and not score_fell:
        return (
            f"Stress signals are rising (avg {early_stress:.1f} → {late_stress:.1f}) "
            "but cognitive performance has held steady so far. Monitor closely."
        )

    return None


def _detect_procrastination_cycle(records) -> str | None:
    """
    Looks for repeating bursts of high delay followed by recovery —
    a cyclical procrastination pattern.
    """
    if len(records) < 6:
        return None

    delay_series = [_get(r, "delay_count") for r in records]
    threshold    = _safe_mean(delay_series) + _safe_stdev(delay_series) * 0.6

    spikes = sum(1 for d in delay_series if d > threshold)
    spike_rate = spikes / len(delay_series)

    if spike_rate >= 0.4:
        return (
            f"Delay spikes appear in {spikes} out of {len(delay_series)} recorded sessions "
            f"({spike_rate*100:.0f}% of the time), suggesting a recurring procrastination cycle "
            "rather than isolated incidents."
        )

    return None


def _detect_cognitive_score_variability(records) -> str | None:
    """
    High stdev in cognitive scores suggests inconsistent performance —
    no stable routine.
    """
    if len(records) < 4:
        return None

    scores = [_get(r, "cognitive_score") for r in records]
    stdev  = _safe_stdev(scores)
    mean   = _safe_mean(scores)

    if stdev > 18:
        return (
            f"Cognitive scores are highly volatile (std dev: {stdev:.1f}, mean: {mean:.1f}). "
            "Performance swings sharply between sessions, suggesting an inconsistent daily routine."
        )

    if stdev < 5 and mean >= 55:
        return (
            f"Cognitive scores are remarkably stable (std dev: {stdev:.1f}, mean: {mean:.1f}). "
            "Consistent routines appear to be sustaining solid performance."
        )

    return None


def _detect_score_trend(records) -> str | None:
    """
    A simple slope check over the full history to surface directional trends.
    """
    if len(records) < 4:
        return None

    scores = [_get(r, "cognitive_score") for r in records]
    slope  = _linear_slope(scores)

    if slope >= 1.2:
        return (
            f"Cognitive scores show a clear upward trend "
            f"(+{slope:.2f} points/session on average). "
            "Behavioral momentum is building positively."
        )

    if slope <= -1.2:
        return (
            f"Cognitive scores are trending downward "
            f"({slope:.2f} points/session on average). "
            "A course correction in daily habits may be needed."
        )

    return None


def _detect_high_score_action_correlation(records) -> str | None:
    """
    Checks whether high-action days correlate with higher cognitive scores.
    """
    if len(records) < 5:
        return None

    scores  = [_get(r, "cognitive_score") for r in records]
    actions = [_get(r, "action_count")    for r in records]

    mean_score  = _safe_mean(scores)
    high_action_scores = [
        scores[i] for i, a in enumerate(actions)
        if a > _safe_mean(actions)
    ]

    if not high_action_scores:
        return None

    avg_high_action_score = _safe_mean(high_action_scores)

    if avg_high_action_score > mean_score + 5:
        return (
            f"Days with above-average action counts yield a cognitive score of "
            f"{avg_high_action_score:.1f} on average, versus {mean_score:.1f} overall. "
            "High action correlates strongly with better cognitive performance."
        )

    return None


def _detect_stress_spike_anomaly(records) -> str | None:
    """
    Flags if recent stress levels are anomalously high relative to baseline.
    """
    if len(records) < 6:
        return None

    stress_series = [_get(r, "stress_signals") for r in records]
    baseline      = stress_series[:-3]
    recent        = stress_series[-3:]

    baseline_avg = _safe_mean(baseline)
    recent_avg   = _safe_mean(recent)

    if recent_avg > baseline_avg + 1.5:
        return (
            f"Stress signals have spiked in the last 3 sessions "
            f"(avg: {recent_avg:.1f}) compared to the earlier baseline "
            f"(avg: {baseline_avg:.1f}). This is an early burnout warning signal."
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def discover_behavior_patterns(records: list) -> list[str]:
    """
    Analyzes a list of NoteRecord objects (or dicts with the same fields)
    and returns 3–5 human-readable behavioral insight strings.

    Fields consumed per record:
        cognitive_score, action_count, delay_count, stress_signals
    """
    if not records:
        return ["Not enough behavioral data to detect patterns yet."]

    # Sort by date if possible
    try:
        records = sorted(records, key=lambda r: _get(r, "date", ""))
    except Exception:
        pass

    # Run all detectors
    candidates = [
        _detect_action_delay_imbalance(records),
        _detect_stress_before_score_decline(records),
        _detect_procrastination_cycle(records),
        _detect_cognitive_score_variability(records),
        _detect_score_trend(records),
        _detect_high_score_action_correlation(records),
        _detect_stress_spike_anomaly(records),
    ]

    # Filter out None results
    patterns = [p for p in candidates if p is not None]

    if not patterns:
        return ["Behavioral patterns are still stabilizing — keep logging daily notes for richer insights."]

    # Return at most 5 patterns
    return patterns[:5]