"""
ContextWeave — profile_engine.py
Builds a user's Behavioral DNA summary from their note history.
"""
import json, os, statistics
from datetime import datetime, timedelta


def build_behavioral_dna(user_id: int, notes: list) -> dict:
    """
    Returns a dict with all fields needed for the public profile page.
    """
    if not notes:
        return _empty_dna()

    # ── Score history from behavior_history file ──────────────────────────────
    history_file = f"behavior_history_{user_id}.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file) as f:
                history = json.load(f)
        except Exception:
            history = []

    scores = [h["score"] for h in history if "score" in h]
    avg_score = round(statistics.mean(scores[-7:]), 1) if scores else 50

    # ── Trend ─────────────────────────────────────────────────────────────────
    trend = "stable"
    if len(scores) >= 3:
        if scores[-1] > scores[0] + 5:   trend = "improving"
        elif scores[-1] < scores[0] - 5: trend = "declining"

    # ── Streak (consecutive days with notes) ──────────────────────────────────
    streak = _calculate_streak(notes)

    # ── Dominant pattern from notes ───────────────────────────────────────────
    action_words  = ["start","finish","submit","complete","study","prepare","build","code","work"]
    delay_words   = ["later","tomorrow","soon","postpone","delay","procrastinat"]
    stress_words  = ["deadline","exam","urgent","pressure","anxious","stressed"]
    social_words  = ["meeting","rahul","tarun","ayush","abhay","shikha","team","friend"]

    text_all = " ".join(notes).lower()
    counts = {
        "Executor":  sum(w in text_all for w in action_words),
        "Procrastinator": sum(w in text_all for w in delay_words),
        "High-Stress Performer": sum(w in text_all for w in stress_words),
        "Collaborator": sum(w in text_all for w in social_words),
    }
    dominant_pattern = max(counts, key=counts.get)

    pattern_descs = {
        "Executor":              "High action orientation, task-completion focused",
        "Procrastinator":        "Delay signals recurring — working on breaking the cycle",
        "High-Stress Performer": "Performs under pressure, stress-driven motivation",
        "Collaborator":          "People-oriented, thrives in team environments",
    }

    # ── Archetype (combination) ───────────────────────────────────────────────
    archetype, archetype_desc, archetype_emoji, top_trait = _compute_archetype(
        avg_score, trend, dominant_pattern, streak
    )

    return {
        "avg_score":        avg_score,
        "trend":            trend,
        "streak":           streak,
        "dominant_pattern": dominant_pattern,
        "pattern_desc":     pattern_descs[dominant_pattern],
        "archetype":        archetype,
        "archetype_desc":   archetype_desc,
        "archetype_emoji":  archetype_emoji,
        "top_trait":        top_trait,
    }


def _calculate_streak(notes: list) -> int:
    """Count consecutive days with at least one note up to today."""
    dates = set()
    for note in notes:
        try:
            date_str = note.split("]")[0].replace("[","").strip()
            dates.add(datetime.strptime(date_str, "%Y-%m-%d").date())
        except Exception:
            pass

    if not dates:
        return 0

    streak = 0
    day = datetime.today().date()
    while day in dates:
        streak += 1
        day -= timedelta(days=1)
    return streak


def _compute_archetype(score, trend, pattern, streak):
    """Map behavioral signals to a named archetype."""

    if score >= 65 and trend == "improving" and pattern == "Executor":
        return (
            "The Momentum Builder",
            "You're in a growth phase — execution is strong, scores are rising, "
            "and your behavioral data shows consistent forward motion. "
            "You set targets and follow through. Rare.",
            "🚀", "High Executor"
        )

    if pattern == "High-Stress Performer" and score >= 55:
        return (
            "The Pressure Converter",
            "You transform stress into output. Deadlines and pressure don't break you — "
            "they activate you. Your cognitive scores hold or rise under stress signals "
            "where others decline.",
            "⚡", "Stress-Driven"
        )

    if pattern == "Procrastinator" and trend == "improving":
        return (
            "The Recovering Strategist",
            "Your behavioral data shows a history of delay patterns — but the trend "
            "is shifting. You're actively working against your own procrastination cycle. "
            "That self-awareness is itself a high-performance trait.",
            "🔄", "Self-Aware"
        )

    if pattern == "Collaborator" and score >= 50:
        return (
            "The Network Thinker",
            "People and collaborative contexts are central to your behavioral patterns. "
            "You think in relationships and shared goals. Your best work happens "
            "when others are involved.",
            "🤝", "Collaborative"
        )

    if streak >= 5:
        return (
            "The Consistent Operator",
            "Consistency is your superpower. While others fluctuate, you show up. "
            "Your behavioral streak data shows a reliable daily logging habit — "
            "the foundation of all high performance.",
            "🎯", f"{streak}-Day Streak"
        )

    if score < 40:
        return (
            "The Rebuilder",
            "Current behavioral data suggests a recovery phase — cognitive load is high, "
            "delay signals are elevated. But recognising the pattern is the first step. "
            "Your next phase starts now.",
            "🌱", "Recovery Mode"
        )

    return (
        "The Balanced Thinker",
        "Your behavioral profile shows a stable, balanced pattern across action, "
        "stress, and focus signals. You maintain equilibrium — a rare trait in "
        "high-stimulus environments.",
        "🧠", "Balanced"
    )


def _empty_dna() -> dict:
    return {
        "avg_score": 50,
        "trend": "stable",
        "streak": 0,
        "dominant_pattern": "Building...",
        "pattern_desc": "Add more notes to unlock your behavioral DNA",
        "archetype": "The Emerging Mind",
        "archetype_desc": "Your behavioral profile is still forming. Keep logging daily notes — "
                          "patterns will surface within a few days.",
        "archetype_emoji": "🌱",
        "top_trait": "New User",
    }