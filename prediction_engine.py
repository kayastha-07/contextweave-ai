import json
import numpy as np

def predict_next_score():

    try:
        with open("behavior_history.json","r") as f:
            data = json.load(f)
    except:
        return None

    scores = [d["score"] for d in data]

    if len(scores) < 5:
        return None

    x = np.arange(len(scores))
    y = np.array(scores)

    slope, intercept = np.polyfit(x, y, 1)

    next_x = len(scores)
    predicted = slope * next_x + intercept

    return round(predicted,2)