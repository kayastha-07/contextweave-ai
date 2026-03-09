"""
ContextWeave Behavioral Intelligence Engine
============================================
Five interlocking modules that evolve your system from reactive analysis
into a proactive, adaptive behavioral intelligence layer.

Modules:
  1. BehavioralTrajectoryTracker  — improvement / decline / stability detection
  2. AIStrategyEngine             — adaptive mode selection
  3. PredictiveRiskModel          — burnout / procrastination / overload risk
  4. WeeklyBehavioralMemory       — 7-day drift detection
  5. AIRecommendationEngine       — context + trajectory-aware suggestions

Integration:
  All modules accept/return plain Python dicts so they slot directly
  into your existing FastAPI report pipeline with zero new heavy deps.

Dependencies: statistics (stdlib), collections (stdlib), datetime (stdlib)
Optional:     spaCy (already in your stack — used for enriched signal extraction)
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared data contract
# ─────────────────────────────────────────────────────────────────────────────

class NoteRecord:
    """
    Minimal wrapper around a daily note entry.

    Fields produced by your current system that we consume:
      date           : date
      cognitive_score: float  0-100
      action_count   : int
      delay_count    : int
      stress_signals : int
      themes         : list[str]
      raw_text       : str
    """

    __slots__ = (
        "date",
        "cognitive_score",
        "action_count",
        "delay_count",
        "stress_signals",
        "themes",
        "raw_text",
    )

    def __init__(
        self,
        date: date,
        cognitive_score: float,
        action_count: int = 0,
        delay_count: int = 0,
        stress_signals: int = 0,
        themes: Optional[list[str]] = None,
        raw_text: str = "",
    ):
        self.date = date
        self.cognitive_score = cognitive_score
        self.action_count = action_count
        self.delay_count = delay_count
        self.stress_signals = stress_signals
        self.themes = themes or []
        self.raw_text = raw_text

    @classmethod
    def from_dict(cls, d: dict) -> "NoteRecord":
        return cls(
            date=d["date"] if isinstance(d["date"], date) else date.fromisoformat(d["date"]),
            cognitive_score=float(d.get("cognitive_score", 50)),
            action_count=int(d.get("action_count", 0)),
            delay_count=int(d.get("delay_count", 0)),
            stress_signals=int(d.get("stress_signals", 0)),
            themes=list(d.get("themes", [])),
            raw_text=str(d.get("raw_text", "")),
        )

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "cognitive_score": self.cognitive_score,
            "action_count": self.action_count,
            "delay_count": self.delay_count,
            "stress_signals": self.stress_signals,
            "themes": self.themes,
            "raw_text": self.raw_text,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _linear_slope(values: list[float]) -> float:
    """
    Returns the slope of the best-fit line through (index, value) pairs.
    Positive slope = upward trend; negative = downward trend.
    Uses least-squares without any external library.
    """
    n = len(values)
    if n < 2:
        return 0.0

    xs = list(range(n))
    mean_x = _safe_mean(xs)
    mean_y = _safe_mean(values)

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    denominator = sum((x - mean_x) ** 2 for x in xs)

    return numerator / denominator if denominator != 0 else 0.0


def _normalise(value: float, low: float, high: float) -> float:
    """Clamp a value to [0, 1] between low and high."""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — Behavioral Trajectory Tracker
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralTrajectoryTracker:
    """
    Detects whether the user's behavior is improving, declining, or stable.
    """

    IMPROVEMENT_THRESHOLD = 0.8
    DECLINE_THRESHOLD = -0.8

    def __init__(self, window: int = 14):
        self.window = window

    def _composite(self, record: NoteRecord) -> float:
        """
        Collapse a NoteRecord into a single behavioral health value (0-100).
        """

        total_signals = record.action_count + record.delay_count + 1
        execution_ratio = record.action_count / total_signals

        stress_penalty = min(record.stress_signals * 3, 20)

        score = (
            record.cognitive_score * 0.55
            + execution_ratio * 30
            - stress_penalty
        )

        return max(0.0, min(100.0, score))

    def analyse(self, records: list[Any]) -> dict[str, Any]:

        # Normalize records so engine always receives NoteRecord objects
        normalized_records = [
            NoteRecord.from_dict(r) if isinstance(r, dict) else r
            for r in records
        ]

        # Now safe to sort
        records_sorted = sorted(normalized_records, key=lambda r: r.date)

        window_records = records_sorted[-self.window:]

        if len(window_records) < 3:
            return {
                "trajectory": "insufficient_data",
                "confidence": 0.0,
                "slope": 0.0,
                "volatility": 0.0,
                "composite_scores": [],
                "window_used": len(window_records),
                "narrative": "Not enough data yet — keep logging daily notes.",
            }

        scores = [self._composite(r) for r in window_records]

        slope = _linear_slope(scores)
        vol = statistics.stdev(scores) if len(scores) > 1 else 0.0

        if slope >= self.IMPROVEMENT_THRESHOLD:
            trajectory = "improving"
        elif slope <= self.DECLINE_THRESHOLD:
            trajectory = "declining"
        else:
            trajectory = "stable"

        slope_confidence = _normalise(abs(slope), 0, 3)
        vol_penalty = _normalise(vol, 0, 30)
        confidence = round(slope_confidence * (1 - vol_penalty * 0.4), 2)

        narrative_map = {
            "improving": f"Scores trending upward (+{slope:.2f}/day) over {len(window_records)} days.",
            "declining": f"Scores trending downward ({slope:.2f}/day) — attention needed.",
            "stable": f"Behavior is stable (slope={slope:.2f}/day, volatility={vol:.1f}).",
        }

        return {
            "trajectory": trajectory,
            "confidence": confidence,
            "slope": round(slope, 3),
            "volatility": round(vol, 2),
            "composite_scores": [round(s, 1) for s in scores],
            "window_used": len(window_records),
            "narrative": narrative_map[trajectory],
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — AI Strategy Engine
# ─────────────────────────────────────────────────────────────────────────────

class AIStrategyEngine:
    """
    Selects one of four adaptive operating modes based on:
      • current cognitive score
      • trajectory direction + confidence
      • risk signals

    Modes
    -----
    Recovery Mode   — restore energy and reduce load
    Structure Mode  — stabilize chaotic patterns
    Growth Mode     — push forward while momentum exists
    Stability Mode  — maintain current systems
    """

    MODE_PROFILES: dict[str, dict] = {
        "Recovery": {
            "priority_focus": [
                "Reduce task load to your top 3 daily priorities only",
                "Schedule mandatory recovery time",
                "Identify and remove the biggest energy drain",
            ],
            "color_signal": "red",
        },
        "Structure": {
            "priority_focus": [
                "Define one fixed morning routine",
                "Use 90-minute focus blocks",
                "Close open loops — do, defer, or drop",
            ],
            "color_signal": "amber",
        },
        "Growth": {
            "priority_focus": [
                "Advance the highest-leverage skill or project",
                "Increase challenge complexity deliberately",
                "Document lessons to compound learning",
            ],
            "color_signal": "green",
        },
        "Stability": {
            "priority_focus": [
                "Audit what systems are already working",
                "Introduce one small improvement",
                "Use cognitive surplus for strategic thinking",
            ],
            "color_signal": "blue",
        },
    }

    def select_mode(
        self,
        cognitive_score: float,
        trajectory_result: dict,
        risk_result: Optional[dict] = None,
    ) -> dict[str, Any]:

        # Safe reads (prevents crashes if keys missing)
        trajectory = trajectory_result.get("trajectory", "stable")
        confidence = float(trajectory_result.get("confidence", 0.5))
        volatility = float(trajectory_result.get("volatility", 0.0))

        risk_result = risk_result or {}
        burnout_risk = float(risk_result.get("burnout_risk", 0.0))
        overload_risk = float(risk_result.get("overload_risk", 0.0))

        # ── Decision logic ──────────────────────────────────────────────────

        is_recovery = (
            cognitive_score < 40
            or burnout_risk > 0.65
            or (trajectory == "declining" and confidence > 0.6)
        )

        is_structure = (
            not is_recovery
            and (
                volatility > 15
                or overload_risk > 0.6
                or (trajectory == "stable" and cognitive_score < 55)
            )
        )

        is_growth = (
            not is_recovery
            and not is_structure
            and trajectory == "improving"
            and cognitive_score >= 60
            and burnout_risk < 0.4
        )

        # ── Mode selection ──────────────────────────────────────────────────

        if is_recovery:
            mode = "Recovery"
            intensity = min(1.0, 0.5 + burnout_risk * 0.5 + (1 - cognitive_score / 100) * 0.4)

            rationale = (
                f"Cognitive score {cognitive_score:.0f}/100 with "
                f"{'declining trajectory' if trajectory == 'declining' else 'burnout signals'}."
                " Priority is recovery before performance."
            )

        elif is_structure:
            mode = "Structure"
            intensity = min(1.0, 0.4 + (volatility / 30) * 0.6)

            rationale = (
                f"Behavior volatility ({volatility:.1f}) indicates inconsistent patterns."
                " Building stronger routines will stabilize output."
            )

        elif is_growth:
            mode = "Growth"
            intensity = min(1.0, 0.5 + (cognitive_score - 60) / 80 + confidence * 0.3)

            rationale = (
                f"Improving trajectory with cognitive score {cognitive_score:.0f}/100."
                " Conditions support pushing forward."
            )

        else:
            mode = "Stability"
            intensity = 0.6

            rationale = (
                f"Cognitive score {cognitive_score:.0f}/100 with stable trajectory."
                " Maintain current systems and avoid disruption."
            )

        profile = self.MODE_PROFILES.get(mode, self.MODE_PROFILES["Stability"])

        return {
            "mode": mode,
            "rationale": rationale,
            "priority_focus": profile["priority_focus"],
            "mode_intensity": round(float(intensity), 2),
            "color_signal": profile["color_signal"],
        }
    
# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — Predictive Risk Model
# ─────────────────────────────────────────────────────────────────────────────

class PredictiveRiskModel:
    """
    Predicts three behavioral risk levels from recent trend data.

    Risk signals
    ------------
    Burnout risk     — sustained high stress + declining cognitive scores
    Procrastination  — delay signals rising while action signals fall
    Overload risk    — action volume increasing but cognitive score dropping
                       (doing more but thinking worse = overloaded, not productive)

    Each risk is a float 0-1 (0 = none, 1 = critical).
    A label (low / moderate / high / critical) and explanation are also returned.

    Returns
    -------
    dict with keys:
      burnout_risk          : float
      procrastination_risk  : float
      overload_risk         : float
      burnout_label         : str
      procrastination_label : str
      overload_label        : str
      dominant_risk         : str   (highest-scoring risk name)
      risk_summary          : str   (narrative paragraph)
    """

    RISK_THRESHOLDS = {
        "low":      (0.0,  0.30),
        "moderate": (0.30, 0.55),
        "high":     (0.55, 0.75),
        "critical": (0.75, 1.01),
    }

    def _label(self, score: float) -> str:
        for label, (lo, hi) in self.RISK_THRESHOLDS.items():
            if lo <= score < hi:
                return label
        return "critical"

    def _burnout_risk(self, records: list[NoteRecord]) -> float:
        """
        Burnout compounds when stress is sustained AND performance is declining.
        Uses a weighted time-decay so recent days count more.
        """
        if not records:
            return 0.0

        n = len(records)
        weights = [math.exp(i / n) for i in range(n)]   # recency weighting
        total_w = sum(weights)

        stress_score = sum(
            w * min(r.stress_signals / 5, 1.0) for w, r in zip(weights, records)
        ) / total_w

        perf_scores = [r.cognitive_score for r in records]
        perf_slope  = _linear_slope(perf_scores)
        decline_factor = _normalise(-perf_slope, 0, 5)   # negative slope = higher risk

        # Combined: stress presence + performance decline
        burnout = stress_score * 0.55 + decline_factor * 0.45
        return round(min(1.0, burnout), 3)

    def _procrastination_risk(self, records: list[NoteRecord]) -> float:
        """
        Procrastination risk rises when delay ratio trends upward
        and action ratio trends downward — both must confirm the pattern.
        """
        if len(records) < 3:
            return 0.0

        delay_ratios  = []
        action_ratios = []

        for r in records:
            total = r.action_count + r.delay_count + 1
            delay_ratios.append(r.delay_count / total)
            action_ratios.append(r.action_count / total)

        delay_slope  = _linear_slope(delay_ratios)   # positive = rising delays
        action_slope = _linear_slope(action_ratios)  # negative = falling actions

        recent_delay = _safe_mean(delay_ratios[-3:])  # absolute recent level

        slope_signal  = _normalise(delay_slope, -0.1, 0.15)
        action_signal = _normalise(-action_slope, -0.1, 0.15)
        level_signal  = recent_delay                              # already 0-1

        risk = slope_signal * 0.35 + action_signal * 0.30 + level_signal * 0.35
        return round(min(1.0, risk), 3)

    def _overload_risk(self, records: list[NoteRecord]) -> float:
        """
        Overload: action volume is high/rising but cognitive quality is declining.
        This is the 'doing more, thinking worse' trap.
        """
        if len(records) < 3:
            return 0.0

        action_counts  = [float(r.action_count) for r in records]
        cognitive_vals = [r.cognitive_score for r in records]

        action_slope = _linear_slope(action_counts)   # positive = more doing
        cog_slope    = _linear_slope(cognitive_vals)  # negative = degrading quality

        recent_load  = _safe_mean(action_counts[-3:])
        load_signal  = _normalise(recent_load, 0, 10)

        divergence = _normalise(action_slope, -0.5, 2.0) * _normalise(-cog_slope, -2, 2)

        risk = load_signal * 0.4 + divergence * 0.6
        return round(min(1.0, risk), 3)

    def analyse(self, records: list[Any], window: int = 10) -> dict[str, Any]:

        # Normalize records so engine always works with NoteRecord objects
        normalized_records = [
        NoteRecord.from_dict(r) if isinstance(r, dict) else r
        for r in records
    ]

        records_sorted = sorted(normalized_records, key=lambda r: r.date)
        window_records = records_sorted[-window:]

        burnout  = self._burnout_risk(window_records)
        procrast = self._procrastination_risk(window_records)
        overload = self._overload_risk(window_records)

        risks = {
            "burnout": burnout,
            "procrastination": procrast,
            "overload": overload,
       }

        dominant = max(risks, key=risks.get)

        narrative_parts = []

        if burnout >= 0.55:
            narrative_parts.append(
                f"Burnout risk is {self._label(burnout)} ({burnout:.0%}): "
                "sustained stress combined with declining performance suggests energy depletion."
            )

        if procrast >= 0.55:
            narrative_parts.append(
                f"Procrastination risk is {self._label(procrast)} ({procrast:.0%}): "
                "delay patterns are rising while intentional action is falling."
            )

        if overload >= 0.55:
            narrative_parts.append(
                f"Overload risk is {self._label(overload)} ({overload:.0%}): "
                "high activity volume is degrading cognitive quality — doing more, achieving less."
            )

        if not narrative_parts:
            narrative_parts = ["All behavioral risk levels are within healthy range."]

        return {
            "burnout_risk": burnout,
            "procrastination_risk": procrast,
            "overload_risk": overload,
            "burnout_label": self._label(burnout),
            "procrastination_label": self._label(procrast),
            "overload_label": self._label(overload),
            "dominant_risk": dominant,
            "risk_summary": " ".join(narrative_parts),
       }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Weekly Behavioral Memory
# ─────────────────────────────────────────────────────────────────────────────

class WeeklyBehavioralMemory:
    """
    Compares this week (last 7 days) vs. last week (days 8-14)
    to surface behavioral drift across five dimensions.

    Dimensions
    ----------
    cognitive_drift    — change in average cognitive score
    stress_drift       — change in average stress signals
    execution_drift    — change in action/delay ratio
    theme_drift        — new themes appearing / old themes disappearing
    consistency_drift  — change in day-to-day variance (more/less predictable)

    Returns
    -------
    dict with keys:
      this_week_avg       : dict (averages for last 7 days)
      last_week_avg       : dict (averages for prior 7 days)
      cognitive_drift     : float (positive = improved)
      stress_drift        : float (negative = less stress = good)
      execution_drift     : float (positive = more action, less delay)
      consistency_drift   : float (negative = more erratic)
      new_themes          : list[str]
      dropped_themes      : list[str]
      drift_severity      : "none" | "minor" | "moderate" | "significant"
      drift_narrative     : str
    """

    def _week_stats(self, records: list[NoteRecord]) -> dict:
        if not records:
            return {
                "avg_cognitive": 0.0,
                "avg_stress": 0.0,
                "avg_action": 0.0,
                "avg_delay": 0.0,
                "exec_ratio": 0.0,
                "variance": 0.0,
                "themes": set(),
            }

        cognitive = [r.cognitive_score for r in records]
        stress    = [float(r.stress_signals) for r in records]
        action    = [float(r.action_count) for r in records]
        delay     = [float(r.delay_count) for r in records]
        total_ad  = sum(action) + sum(delay) + 1

        themes = set()
        for r in records:
            themes.update(r.themes)

        return {
            "avg_cognitive": round(_safe_mean(cognitive), 2),
            "avg_stress":    round(_safe_mean(stress), 2),
            "avg_action":    round(_safe_mean(action), 2),
            "avg_delay":     round(_safe_mean(delay), 2),
            "exec_ratio":    round(sum(action) / total_ad, 3),
            "variance":      round(statistics.stdev(cognitive) if len(cognitive) > 1 else 0.0, 2),
            "themes":        themes,
        }

    def compare(self, records: list[Any]) -> dict[str, Any]:

        # Normalize records so we always work with NoteRecord objects
        normalized_records = [
            NoteRecord.from_dict(r) if isinstance(r, dict) else r
            for r in records
        ]

        if not normalized_records:
            return {
                "this_week_avg": {},
                "last_week_avg": {},
                "cognitive_drift": 0,
                "stress_drift": 0,
                "execution_drift": 0,
                "consistency_drift": 0,
                "new_themes": [],
                "dropped_themes": [],
                "drift_severity": "none",
                "drift_narrative": "No behavioral data yet.",
            }

        today = max((r.date for r in normalized_records), default=date.today())

        this_week = [r for r in normalized_records if (today - r.date).days < 7]
        last_week = [r for r in normalized_records if 7 <= (today - r.date).days < 14]

        tw = self._week_stats(this_week)
        lw = self._week_stats(last_week)

        cognitive_drift = round(tw["avg_cognitive"] - lw["avg_cognitive"], 2)
        stress_drift = round(tw["avg_stress"] - lw["avg_stress"], 2)
        execution_drift = round(tw["exec_ratio"] - lw["exec_ratio"], 3)
        consistency_drift = round(-(tw["variance"] - lw["variance"]), 2)

        new_themes = list(tw["themes"] - lw["themes"])
        dropped_themes = list(lw["themes"] - tw["themes"])

        drift_signals = [
            abs(cognitive_drift) / 10,
            abs(stress_drift) / 5,
            abs(execution_drift) * 3,
            abs(consistency_drift) / 15,
        ]

        drift_magnitude = _safe_mean(drift_signals)

        if drift_magnitude < 0.05:
            severity = "none"
        elif drift_magnitude < 0.15:
            severity = "minor"
        elif drift_magnitude < 0.30:
            severity = "moderate"
        else:
            severity = "significant"

        parts = []

        if abs(cognitive_drift) >= 5:
            direction = "improved" if cognitive_drift > 0 else "declined"
            parts.append(f"Cognitive score {direction} by {abs(cognitive_drift):.1f} points vs last week.")

        if stress_drift >= 1:
            parts.append(f"Stress signals increased by {stress_drift:.1f} — monitor for escalation.")
        elif stress_drift <= -1:
            parts.append(f"Stress signals decreased by {abs(stress_drift):.1f} — a positive shift.")

        if execution_drift >= 0.1:
            parts.append("Execution ratio improved: more action relative to delay.")
        elif execution_drift <= -0.1:
            parts.append("Execution ratio declined: delays are outpacing intentional actions.")

        if new_themes:
            parts.append(f"New themes emerging: {', '.join(new_themes[:3])}.")

        if dropped_themes:
            parts.append(f"Themes that faded: {', '.join(dropped_themes[:3])}.")

        if not parts:
            parts = ["Behavioral patterns are consistent with last week — no notable drift."]

        tw_out = {k: list(v) if isinstance(v, set) else v for k, v in tw.items()}
        lw_out = {k: list(v) if isinstance(v, set) else v for k, v in lw.items()}

        return {
            "this_week_avg": tw_out,
            "last_week_avg": lw_out,
            "cognitive_drift": cognitive_drift,
            "stress_drift": stress_drift,
            "execution_drift": execution_drift,
            "consistency_drift": consistency_drift,
            "new_themes": new_themes,
            "dropped_themes": dropped_themes,
            "drift_severity": severity,
            "drift_narrative": " ".join(parts),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — AI Recommendation Engine
# ─────────────────────────────────────────────────────────────────────────────

class AIRecommendationEngine:
    """
    Generates context-aware, trajectory-aware, adaptive recommendations.
    Recommendations are NOT static rules. They are composed from:
      1. Active mode    (from AIStrategyEngine)
      2. Dominant risk  (from PredictiveRiskModel)
      3. Drift signals  (from WeeklyBehavioralMemory)
      4. Trajectory     (from BehavioralTrajectoryTracker)
      5. Recent themes  (from note records)

    Output
    ------
    dict with keys:
      primary_recommendation   : str  (the single most important action)
      supporting_actions       : list[str]  (2-3 supporting steps)
      avoid_this_week          : list[str]  (anti-patterns to skip)
      adaptation_note          : str  (why this advice differs from last week)
      confidence               : float 0-1
    """

    # ── Recommendation fragments keyed by (mode, risk, trajectory) ──────────

    _PRIMARY = {
        ("Recovery", "burnout"):         "Block 60-90 minutes today as a non-negotiable restoration window — no tasks, no screens.",
        ("Recovery", "procrastination"): "Identify the one task you've been avoiding longest. Do only the first 5 minutes of it — nothing more.",
        ("Recovery", "overload"):        "Cut today's task list in half. Postpone everything that doesn't have a hard external deadline.",
        ("Structure", "burnout"):        "Design a minimal viable daily structure — three anchors: start time, one deep work block, end time.",
        ("Structure", "procrastination"):"Create a 'decision-free' morning: pre-decide your first task the night before so you start without friction.",
        ("Structure", "overload"):       "Implement a daily task cap: maximum 5 items on your list. Ruthlessly defer the rest.",
        ("Growth", "burnout"):           "Your trajectory is positive — protect it by scheduling recovery proactively before stress accumulates.",
        ("Growth", "procrastination"):   "Use forward momentum to tackle a meaningful challenge you've deferred. The resistance is lower now than it will be.",
        ("Growth", "overload"):          "Channel your activity toward depth, not breadth — one ambitious project beats five shallow ones.",
        ("Stability", "burnout"):        "You're stable but carrying background stress. Introduce a 10-minute daily decompression ritual.",
        ("Stability", "procrastination"):"Audit your open loops — list everything mentally 'in progress' and make a concrete decision on each.",
        ("Stability", "overload"):       "Your volume is high but sustainable. Set a weekly review to proactively shed tasks before they accumulate.",
    }

    _DEFAULT_PRIMARY = {
        "Recovery":  "Prioritise rest and reduce complexity. Focus only on what is essential today.",
        "Structure": "Build one reliable routine anchor and repeat it every day this week.",
        "Growth":    "Identify your highest-leverage opportunity and dedicate your best cognitive hours to it.",
        "Stability": "Protect your current systems. Introduce one measured improvement without disrupting flow.",
    }

    _SUPPORTING = {
        "Recovery": [
            "Sleep 7-8 hours — treat this as a performance metric, not a lifestyle choice.",
            "Decline non-urgent commitments for the next 48 hours.",
        ],
        "Structure": [
            "Write tomorrow's top 3 tasks tonight before closing your workday.",
            "Remove one recurring distraction from your environment this week.",
        ],
        "Growth": [
            "Allocate your first 90 minutes each day to your top priority before checking messages.",
            "Track one meaningful progress metric daily — what gets measured improves.",
        ],
        "Stability": [
            "Document one process you rely on — reduce the cognitive cost of repeating it.",
            "Schedule a weekly 30-minute review to maintain awareness without micromanagement.",
        ],
    }

    _AVOID = {
        "Recovery": [
            "Taking on new commitments or projects this week",
            "Skipping sleep to catch up on work",
            "Measuring yourself against peak-performance standards while recovering",
        ],
        "Structure": [
            "Reacting to every inbound request without a triage filter",
            "Multi-tasking across more than two concurrent projects",
            "Letting undefined tasks pile up without assigning them a clear next action",
        ],
        "Growth": [
            "Confusing busyness with progress — activity that doesn't compound isn't growth",
            "Neglecting recovery — growth is built during rest, not during effort",
            "Diluting focus by starting new things before current initiatives have traction",
        ],
        "Stability": [
            "Introducing large changes during a period of stable performance",
            "Ignoring early stress signals because 'everything seems fine'",
            "Letting maintenance tasks accumulate until they become a crisis",
        ],
    }

def generate(
    self,
    mode_result: dict,
    risk_result: dict,
    trajectory_result: dict,
    drift_result: dict,
    recent_records: list[Any],
) -> dict[str, Any]:

    # Normalize records to NoteRecord objects
    normalized_records = [
        NoteRecord.from_dict(r) if isinstance(r, dict) else r
        for r in recent_records
    ]

    mode = mode_result.get("mode", "Stability")
    dom_risk = risk_result.get("dominant_risk", "burnout")
    trajectory = trajectory_result.get("trajectory", "stable")

    # Primary recommendation
    key = (mode, dom_risk)
    primary = self._PRIMARY.get(
        key,
        self._DEFAULT_PRIMARY.get(mode, "Focus on your next most important action.")
    )

    supporting = list(self._SUPPORTING.get(mode, []))

    drift_severity = drift_result.get("drift_severity", "none")
    new_themes = drift_result.get("new_themes", [])

    if drift_severity in ("moderate", "significant") and new_themes:
        supporting.append(
            f"New themes have appeared in your notes ({', '.join(new_themes[:2])}). "
            "Decide explicitly whether these deserve attention or should be filtered out."
        )

    avoid = list(self._AVOID.get(mode, []))

    adaptation_parts = []
    cog_drift = drift_result.get("cognitive_drift", 0.0)

    if trajectory == "improving" and cog_drift > 3:
        adaptation_parts.append(
            "You're in a positive momentum phase — recommendations shift toward capitalising on it."
        )
    elif trajectory == "declining" and cog_drift < -3:
        adaptation_parts.append(
            "Performance has dipped week-over-week — recommendations shift toward protection and recovery."
        )
    elif drift_severity == "significant":
        adaptation_parts.append(
            "Significant behavioral drift detected — strategy adjusted to address emerging patterns."
        )
    else:
        adaptation_parts.append(
            "Patterns are consistent with previous week — strategy reinforces what is working."
        )

    risk_label = risk_result.get(f"{dom_risk}_label", "moderate")
    if risk_label in ("high", "critical"):
        adaptation_parts.append(
            f"{dom_risk.capitalize()} risk is {risk_label} — this influences today's recommendation."
        )

    traj_conf = trajectory_result.get("confidence", 0.5)

    risk_max = max(
        risk_result.get("burnout_risk", 0),
        risk_result.get("procrastination_risk", 0),
        risk_result.get("overload_risk", 0),
    )

    confidence = round(traj_conf * 0.6 + min(risk_max * 1.2, 1.0) * 0.4, 2)

    return {
        "primary_recommendation": primary,
        "supporting_actions": supporting,
        "avoid_this_week": avoid,
        "adaptation_note": " ".join(adaptation_parts),
        "confidence": confidence,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR — Drop-in for your FastAPI report endpoint
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralIntelligencePipeline:
    """
    Orchestrates all five modules into a single structured output
    ready for your dashboard and report generation.

    Usage (FastAPI)
    ---------------
        pipeline = BehavioralIntelligencePipeline()

        @app.post("/report/behavioral")
        async def behavioral_report(payload: ReportRequest):
            records = [NoteRecord.from_dict(n) for n in payload.notes]
            return pipeline.run(records, current_cognitive_score=payload.cognitive_score)

    Returns
    -------
    A single dict with top-level keys:
      trajectory, strategy, risks, weekly_memory, recommendations, meta
    """

    def __init__(
        self,
        trajectory_window: int = 14,
        risk_window:       int = 10,
    ):
        self.trajectory_tracker = BehavioralTrajectoryTracker(window=trajectory_window)
        self.strategy_engine    = AIStrategyEngine()
        self.risk_model         = PredictiveRiskModel()
        self.weekly_memory      = WeeklyBehavioralMemory()
        self.rec_engine         = AIRecommendationEngine()

    def run(self, records: list[Any], current_cognitive_score: Optional[float] = None) -> dict[str, Any]:

        if not records:
            return {"error": "No note records supplied to pipeline."}

        # Convert dict records into NoteRecord objects
        normalized_records = []
        for r in records:
            if isinstance(r, dict):
                normalized_records.append(NoteRecord.from_dict(r))
            else:
                normalized_records.append(r)

        # Sort by date
        records_sorted = sorted(normalized_records, key=lambda r: r.date)

        # Use latest score if not provided
        if current_cognitive_score is None:
            current_cognitive_score = records_sorted[-1].cognitive_score

        # Stage 1 — trajectory
        trajectory_result = self.trajectory_tracker.analyse(records_sorted)

        # Stage 2 — risk
        risk_result = self.risk_model.analyse(records_sorted)

        # Stage 3 — strategy
        strategy_result = self.strategy_engine.select_mode(
            cognitive_score=current_cognitive_score,
            trajectory_result=trajectory_result,
            risk_result=risk_result,
        )

        # Stage 4 — weekly memory
        drift_result = self.weekly_memory.compare(records_sorted)

        # Stage 5 — recommendations
        rec_result = self.rec_engine.generate(
            mode_result=strategy_result,
            risk_result=risk_result,
            trajectory_result=trajectory_result,
            drift_result=drift_result,
            recent_records=records_sorted[-7:],
        )

        return {
            "trajectory": trajectory_result,
            "strategy": strategy_result,
            "risks": risk_result,
            "weekly_memory": drift_result,
            "recommendations": rec_result,
            "meta": {
                "records_analysed": len(records_sorted),
                "current_score": current_cognitive_score,
                "analysis_date": date.today().isoformat(),
                "pipeline_version": "1.0.0",
            },
        }
 
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Integration Stub
# ─────────────────────────────────────────────────────────────────────────────

"""
Add this to your existing main.py / router file:

    from contextweave_behavioral_engine import BehavioralIntelligencePipeline, NoteRecord
    from fastapi import APIRouter
    from pydantic import BaseModel

    router = APIRouter()
    pipeline = BehavioralIntelligencePipeline()   # singleton — instantiate once

    class NotePayload(BaseModel):
        date: str
        cognitive_score: float
        action_count: int = 0
        delay_count: int = 0
        stress_signals: int = 0
        themes: list[str] = []
        raw_text: str = ""

    class BehavioralReportRequest(BaseModel):
        notes: list[NotePayload]
        current_cognitive_score: float | None = None

    @router.post("/report/behavioral-intelligence")
    async def behavioral_intelligence_report(request: BehavioralReportRequest):
        records = [NoteRecord.from_dict(n.dict()) for n in request.notes]
        return pipeline.run(records, request.current_cognitive_score)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run: python contextweave_behavioral_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import random

    random.seed(42)
    today = date.today()

    # Simulate 21 days of notes with gradual improvement then stress spike
    def _make_records() -> list[NoteRecord]:
        records = []
        base_score = 45.0
        for i in range(21):
            d = today - timedelta(days=20 - i)
            trend = i * 0.8                          # gradual improvement
            stress_spike = 4 if i >= 17 else 0       # stress spike in last 4 days
            score = min(85, base_score + trend + random.uniform(-5, 5))
            records.append(NoteRecord(
                date=d,
                cognitive_score=round(score, 1),
                action_count=random.randint(2, 7),
                delay_count=random.randint(0, 3) + (2 if i >= 17 else 0),
                stress_signals=random.randint(0, 2) + stress_spike,
                themes=random.sample(
                    ["focus", "deadlines", "energy", "meetings", "planning", "health"],
                    k=random.randint(1, 3),
                ),
                raw_text=f"Day {i+1} note text.",
            ))
        return records

    records = _make_records()
    pipeline = BehavioralIntelligencePipeline()
    result = pipeline.run(records)

    print(json.dumps(result, indent=2, default=str))