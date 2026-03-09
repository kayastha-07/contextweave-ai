import json
import statistics

def detect_behavior_anomaly():

    try:
        with open("behavior_history.json", "r") as f:
            data = json.load(f)
    except:
        return "Behavior history not available"

    scores = [d["score"] for d in data if "score" in d]

    if len(scores) < 8:
        return "Learning behavior pattern — more data needed"

    baseline = scores[:-4]
    recent = scores[-4:]

    baseline_avg = statistics.mean(baseline)
    recent_avg = statistics.mean(recent)

    difference = recent_avg - baseline_avg

    if difference < -12:
        return "⚠ Productivity anomaly detected: recent focus dropped."

    if difference > 12:
        return "⚡ Performance spike detected."

    return "Behavior stable"