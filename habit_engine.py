import json
import statistics
import os

def discover_behavior_patterns():

    file_path = "behavior_history.json"

    if not os.path.exists(file_path):
        return "Not enough behavioral data yet."

    with open(file_path, "r") as f:
        data = json.load(f)

    scores = [d["score"] for d in data if "score" in d]

    if len(scores) < 10:
        return "Learning behavior patterns — more data required."

    avg_score = statistics.mean(scores)
    recent_avg = statistics.mean(scores[-5:])

    insights = []

    if recent_avg > avg_score:
        insights.append("Productivity improving over recent sessions.")

    if recent_avg < avg_score:
        insights.append("Recent productivity below baseline.")

    volatility = statistics.stdev(scores)

    if volatility > 15:
        insights.append("Behavior volatility detected — routines inconsistent.")
    else:
        insights.append("Behavior patterns stable.")

    return " | ".join(insights)