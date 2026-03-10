"""
ContextWeave — forecasting_engine.py
Predicts cognitive score for next 3 days using linear regression on numpy.
"""
import numpy as np
import json, os
from datetime import datetime, timedelta


def forecast_scores(user_id: int, days_ahead: int = 3) -> dict:
    """
    Returns predicted scores for next N days.
    Uses weighted linear regression — recent data has more influence.
    """
    history_file = f"behavior_history_{user_id}.json"

    if not os.path.exists(history_file):
        return _empty_forecast(days_ahead)

    try:
        with open(history_file) as f:
            history = json.load(f)
    except Exception:
        return _empty_forecast(days_ahead)

    scores = [h["score"] for h in history if "score" in h]

    if len(scores) < 3:
        return _empty_forecast(days_ahead)

    scores  = scores[-14:]  # use last 14 days max
    n       = len(scores)
    x       = np.arange(n, dtype=float)
    y       = np.array(scores, dtype=float)

    # Weighted — recent points matter more
    weights = np.linspace(0.5, 1.0, n)
    x_w     = x * weights
    y_w     = y * weights

    # Linear regression
    coeffs  = np.polyfit(x, y, 1, w=weights)
    slope, intercept = coeffs[0], coeffs[1]

    predictions = []
    today = datetime.now().date()

    for i in range(1, days_ahead + 1):
        raw_pred   = slope * (n + i - 1) + intercept
        clamped    = max(0, min(100, round(raw_pred, 1)))
        date_label = (today + timedelta(days=i)).strftime("%b %d")
        predictions.append({
            "day":   i,
            "date":  date_label,
            "score": clamped,
        })

    # Trend direction
    if slope > 0.5:   trend = "rising"
    elif slope < -0.5: trend = "falling"
    else:              trend = "stable"

    # Confidence (based on how consistent past scores are)
    std = float(np.std(y))
    confidence = "high" if std < 8 else "medium" if std < 15 else "low"

    return {
        "predictions":  predictions,
        "trend":        trend,
        "confidence":   confidence,
        "slope":        round(float(slope), 2),
        "based_on_days": n,
    }


def _empty_forecast(days_ahead: int) -> dict:
    today = datetime.now().date()
    return {
        "predictions": [
            {"day": i, "date": (today + timedelta(days=i)).strftime("%b %d"), "score": None}
            for i in range(1, days_ahead + 1)
        ],
        "trend":        "unknown",
        "confidence":   "none",
        "slope":        0.0,
        "based_on_days": 0,
    }


def format_forecast_for_report(forecast: dict) -> str:
    if not forecast or not forecast.get("predictions"):
        return ""
    lines = ["📅 3-Day Behavioral Forecast", "─" * 40]
    for p in forecast["predictions"]:
        score = p["score"]
        bar   = "█" * int((score or 0) // 10) if score else "—"
        lines.append(f"  {p['date']}  :  {score if score else 'Insufficient data'}  {bar}")
    lines.append(f"  Trend: {forecast['trend'].capitalize()}  |  Confidence: {forecast['confidence'].capitalize()}")
    return "\n".join(lines)