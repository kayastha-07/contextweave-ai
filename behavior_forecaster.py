"""
ContextWeave — Behavior Forecaster
====================================
Predicts the next 7 days of cognitive scores using historical behavioral data.

Considers:
  - Cognitive score trend (linear regression slope)
  - Stress signal trajectory
  - Delay ratio (delay / total signals)
  - Action count momentum

Output:
  - 7 predicted daily scores
  - Confidence bounds (lower, upper) per day
  - Narrative summary per day
  - Overall forecast label

Dependencies: statistics, math (stdlib only — no numpy required)
"""

from __future__ import annotations

import json
import math
import os
import statistics
from datetime import date, timedelta
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _linear_regression(values: list[float]) -> tuple[float, float]:
    """
    Returns (slope, intercept) for a simple OLS line through the values.
    x = index (0, 1, 2, ...), y = values.
    """
    n = len(values)
    if n < 2:
        return 0.0, values[0] if values else 50.0

    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)

    slope = num / den if den != 0 else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _residual_std(values: list[float], slope: float, intercept: float) -> float:
    """
    Standard deviation of residuals — used for confidence interval width.
    """
    if len(values) < 2:
        return 10.0  # default uncertainty when data is thin

    fitted = [slope * i + intercept for i in range(len(values))]
    residuals = [v - f for v, f in zip(values, fitted)]
    return statistics.stdev(residuals) if len(residuals) > 1 else 10.0


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _confidence_label(width: float) -> str:
    if width <= 8:
        return "High"
    elif width <= 16:
        return "Moderate"
    else:
        return "Low"


def _score_narrative(score: float, stress_trend: str, delay_trend: str) -> str:
    """Human-readable label for a predicted score."""
    if score >= 72:
        base = "Strong cognitive day expected"
    elif score >= 55:
        base = "Moderate focus expected"
    elif score >= 40:
        base = "Below-average performance risk"
    else:
        base = "Low-focus day likely — recovery advised"

    modifiers = []
    if stress_trend == "rising":
        modifiers.append("stress rising")
    if delay_trend == "rising":
        modifiers.append("delay tendency increasing")
    if stress_trend == "falling" and delay_trend == "falling":
        modifiers.append("positive momentum building")

    if modifiers:
        return f"{base} ({', '.join(modifiers)})"
    return base


def _trend_direction(values: list[float]) -> str:
    """Simple trend label for a series."""
    if len(values) < 3:
        return "stable"
    slope, _ = _linear_regression(values)
    if slope > 0.3:
        return "rising"
    if slope < -0.3:
        return "falling"
    return "stable"


# ─────────────────────────────────────────────────────────────────────────────
# Main forecaster
# ─────────────────────────────────────────────────────────────────────────────

class BehaviorForecaster:
    """
    Predicts the next FORECAST_DAYS cognitive scores from historical
    behavior_history.json data (or from an injected record list).

    Usage
    -----
        forecaster = BehaviorForecaster()
        result = forecaster.forecast()          # reads behavior_history.json
        # — or —
        result = forecaster.forecast(records)   # inject list of dicts

    Each record dict must contain at minimum:
        { "date": "YYYY-MM-DD", "score": float }
    Optional keys (improve accuracy if present):
        "stress_signals", "delay_count", "action_count"
    """

    FORECAST_DAYS = 7
    HISTORY_FILE = "behavior_history.json"
    Z_95 = 1.96   # z-score for ~95 % confidence interval

    def forecast(
        self,
        records: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """
        Returns a dict with keys:
          - forecast_days  : list of per-day dicts
          - overall_label  : str  (e.g. "Improving", "Stable", "Declining")
          - base_score     : float  (last observed score)
          - data_quality   : str  ("sufficient" | "limited" | "insufficient")
          - message        : str  (human-readable summary)
        """

        # ── 1. Load data ───────────────────────────────────────────────────
        if records is None:
            records = self._load_history()

        if not records:
            return self._empty_forecast("No behavioral history found.")

        # Sort by date
        try:
            records = sorted(records, key=lambda r: r.get("date", ""))
        except Exception:
            pass

        # ── 2. Extract feature series ──────────────────────────────────────
        scores         = [float(r.get("score", 50))          for r in records]
        stress_series  = [float(r.get("stress_signals", 0))  for r in records]
        delay_series   = [float(r.get("delay_count", 0))     for r in records]
        action_series  = [float(r.get("action_count", 0))    for r in records]

        n = len(scores)
        data_quality = "sufficient" if n >= 10 else ("limited" if n >= 4 else "insufficient")

        if n < 2:
            return self._empty_forecast(
                "Not enough data for forecasting — keep logging daily notes."
            )

        # ── 3. Fit trend models ────────────────────────────────────────────
        score_slope,  score_intercept  = _linear_regression(scores)
        stress_slope, _                = _linear_regression(stress_series)
        delay_slope,  _                = _linear_regression(delay_series)
        action_slope, _                = _linear_regression(action_series)

        residual_std = _residual_std(scores, score_slope, score_intercept)

        # Trend directions for narrative
        stress_trend = _trend_direction(stress_series)
        delay_trend  = _trend_direction(delay_series)
        action_trend = _trend_direction(action_series)

        # ── 4. Compute composite adjustment factor ─────────────────────────
        # Positive action momentum boosts predicted score slightly.
        # Rising stress or delay dampens it.
        def _daily_adjustment(day_ahead: int) -> float:
            stress_adj = -(stress_slope * day_ahead * 1.5)
            delay_adj  = -(delay_slope  * day_ahead * 1.2)
            action_adj =  (action_slope * day_ahead * 0.8)
            return stress_adj + delay_adj + action_adj

        # ── 5. Generate 7-day forecast ─────────────────────────────────────
        base_x       = n  # next point index after history
        today_date   = date.today()
        forecast_days = []

        # Widen CI as we go further into the future (uncertainty compounds)
        for i in range(1, self.FORECAST_DAYS + 1):
            x_i         = base_x + i - 1
            raw_score   = score_slope * x_i + score_intercept
            adjustment  = _daily_adjustment(i)
            predicted   = _clamp(raw_score + adjustment)

            # Confidence interval — widens with forecast horizon
            horizon_factor = 1 + (i - 1) * 0.15
            half_width = self.Z_95 * residual_std * horizon_factor
            lower = _clamp(predicted - half_width)
            upper = _clamp(predicted + half_width)
            ci_width = upper - lower

            forecast_date = (today_date + timedelta(days=i)).isoformat()

            forecast_days.append({
                "day":               i,
                "date":              forecast_date,
                "predicted_score":   round(predicted, 1),
                "lower_bound":       round(lower, 1),
                "upper_bound":       round(upper, 1),
                "confidence":        _confidence_label(ci_width),
                "narrative":         _score_narrative(predicted, stress_trend, delay_trend),
            })

        # ── 6. Overall forecast label ──────────────────────────────────────
        first_pred = forecast_days[0]["predicted_score"]
        last_pred  = forecast_days[-1]["predicted_score"]
        delta      = last_pred - first_pred

        if delta > 5:
            overall_label = "📈 Improving"
        elif delta < -5:
            overall_label = "📉 Declining"
        else:
            overall_label = "➡ Stable"

        # ── 7. Human-readable summary ──────────────────────────────────────
        avg_pred = sum(d["predicted_score"] for d in forecast_days) / self.FORECAST_DAYS
        message = (
            f"7-day average predicted score: {avg_pred:.1f}. "
            f"Trend: {overall_label.split()[-1]}. "
        )
        if stress_trend == "rising":
            message += "Rising stress may dampen focus. Consider recovery blocks. "
        if delay_trend == "rising":
            message += "Delay patterns increasing — enforce structured time blocks. "
        if action_trend == "rising":
            message += "Action momentum building — capitalise on it. "
        if data_quality == "limited":
            message += "(Note: limited history — forecast accuracy improves with more data.)"

        return {
            "forecast_days":  forecast_days,
            "overall_label":  overall_label,
            "base_score":     round(scores[-1], 1),
            "data_quality":   data_quality,
            "message":        message.strip(),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_history(self) -> list[dict]:
        if not os.path.exists(self.HISTORY_FILE):
            return []
        try:
            with open(self.HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _empty_forecast(self, message: str) -> dict[str, Any]:
        today = date.today()
        return {
            "forecast_days": [
                {
                    "day":             i,
                    "date":            (today + timedelta(days=i)).isoformat(),
                    "predicted_score": 50.0,
                    "lower_bound":     30.0,
                    "upper_bound":     70.0,
                    "confidence":      "Low",
                    "narrative":       "Insufficient data — defaulting to baseline.",
                }
                for i in range(1, BehaviorForecaster.FORECAST_DAYS + 1)
            ],
            "overall_label": "➡ Stable",
            "base_score":    50.0,
            "data_quality":  "insufficient",
            "message":       message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function — drop-in call from run_behavioral_ai()
# ─────────────────────────────────────────────────────────────────────────────

def forecast_cognitive_trajectory(records: Optional[list[dict]] = None) -> dict[str, Any]:
    """
    Thin wrapper for easy import in main.py.

    Returns the full forecast dict from BehaviorForecaster.forecast().
    """
    return BehaviorForecaster().forecast(records)


def format_forecast_for_report(forecast: dict[str, Any]) -> str:
    """
    Converts the forecast dict into a formatted report string
    ready to append to generate_daily_report() output.
    """
    lines = [
        "\n🔮 PREDICTED COGNITIVE TRAJECTORY (Next 7 Days)",
        "─" * 48,
    ]

    lines.append(f"Overall Direction : {forecast['overall_label']}")
    lines.append(f"Last Observed Score: {forecast['base_score']}")
    lines.append(f"Data Quality       : {forecast['data_quality'].capitalize()}")
    lines.append(f"\n{forecast['message']}\n")

    lines.append(f"{'Day':<5} {'Date':<13} {'Score':>6}  {'Range':^15}  {'Conf.':<10}  Narrative")
    lines.append("─" * 80)

    for d in forecast["forecast_days"]:
        score_bar = _ascii_bar(d["predicted_score"])
        lines.append(
            f"Day {d['day']:<2} {d['date']:<13} "
            f"{d['predicted_score']:>5.1f}  "
            f"[{d['lower_bound']:>4.1f}–{d['upper_bound']:<5.1f}]  "
            f"{d['confidence']:<10}  {d['narrative']}"
        )

    lines.append("─" * 80)
    return "\n".join(lines)


def _ascii_bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date, timedelta
    import random

    random.seed(7)
    today = date.today()

    # Simulate 14 days of history with gradual improvement + late stress spike
    mock_records = []
    for i in range(14):
        d = today - timedelta(days=13 - i)
        score = 45 + i * 1.5 + random.uniform(-4, 4)
        stress = max(0, random.randint(0, 2) + (3 if i >= 11 else 0))
        delay  = max(0, random.randint(0, 2) - (1 if i >= 8 else 0))
        action = random.randint(2, 6)
        mock_records.append({
            "date":           d.isoformat(),
            "score":          round(score, 1),
            "stress_signals": stress,
            "delay_count":    delay,
            "action_count":   action,
        })

    forecaster = BehaviorForecaster()
    result = forecaster.forecast(mock_records)

    print(format_forecast_for_report(result))