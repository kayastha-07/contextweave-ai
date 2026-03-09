from behavioral_engine import (
    NoteRecord,
    BehavioralTrajectoryTracker,
    AIStrategyEngine,
    PredictiveRiskModel
)
from prediction_engine import predict_next_score
from anomaly_engine import detect_behavior_anomaly
from habit_engine import discover_behavior_patterns
from behavior_forecaster import forecast_cognitive_trajectory, format_forecast_for_report
from behavior_pattern_engine import discover_behavior_patterns
from llm_signal_engine import analyze_note_with_gemini

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    AI_AVAILABLE = True
except:
    model = None
    AI_AVAILABLE = False
import spacy
from datetime import datetime

import json

def save_behavior_state(ai_output):

    import json
    import os
    from datetime import datetime

    file_path = "behavior_history.json"

    # Build clean state record
    state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "score": ai_output.get("score", 0),
        "trajectory": ai_output.get("trajectory", "stable"),
        "strategy": ai_output.get("strategy", "observe"),
        "risk": ai_output.get("risks", ai_output.get("risk", "none"))
    }

    # Load existing history safely
    history = []

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                history = json.load(f)
        except:
            # file exists but corrupted
            history = []

    # Append new state
    history.append(state)

    # Save back
    with open(file_path, "w") as f:
        json.dump(history, f, indent=2)

def calculate_cognitive_score(notes):

    delay_words = ["later", "tomorrow", "soon", "after"]
    action_words = ["start", "finish", "submit", "complete", "prepare", "study"]
    stress_words = ["pressure", "urgent", "deadline", "exam", "meeting"]

    delay = 0
    action = 0
    stress = 0

    for note in notes:
        text = note.lower()

        for word in delay_words:
            if word in text:
                delay += 1

        for word in action_words:
            if word in text:
                action += 1

        for word in stress_words:
            if word in text:
                stress += 1

    score = 50 + (action * 5) - (delay * 4) - (stress * 3)

    return max(0, min(100, score))

def detect_trajectory(past_scores):

    if len(past_scores) < 3:
        return "insufficient"

    recent = past_scores[-3:]

    if recent[-1] > recent[0] + 5:
        return "improving"

    elif recent[-1] < recent[0] - 5:
        return "declining"

    else:
        return "stable"
    
def select_strategy(score, trajectory):

    if trajectory == "declining" or score < 40:
        return "Recovery"

    elif trajectory == "improving" and score > 60:
        return "Growth"

    elif trajectory == "stable":
        return "Stability"

    else:
        return "Structure"
    
def run_behavioral_ai(notes):

    # Build records
    records_dict = build_note_records(notes)

    # Convert dictionaries → SimpleNamespace objects so engine can use r.date
    from types import SimpleNamespace
    records = []
    for r in records_dict:
        if isinstance(r, dict):
            records.append(SimpleNamespace(**r))
        else:
            records.append(r)

    cognitive_score = calculate_cognitive_score(notes)

    tracker          = BehavioralTrajectoryTracker()
    trajectory_result = tracker.analyse(records)

    risk_model   = PredictiveRiskModel()
    risk_result  = risk_model.analyse(records)

    strategy_engine = AIStrategyEngine()
    strategy_result = strategy_engine.select_mode(
        cognitive_score,
        trajectory_result,
        risk_result
    )

    try:
        prediction = predict_next_score()
    except:
        prediction = None

    try:
        anomaly = detect_behavior_anomaly()
    except:
        anomaly = "Behavior analysis initializing"

    dominant_risk = risk_result.get("dominant_risk", "none")

    advice = "Maintain systems — consistency is working."
    if dominant_risk == "burnout":
        advice = "Prioritize recovery before performance."
    elif dominant_risk == "procrastination":
        advice = "Reduce distractions and enforce focus blocks."

    if cognitive_score > 70:
        focus_state  = "Deep Work"
        focus_advice = "Excellent focus. Maintain current routine."
    elif cognitive_score > 40:
        focus_state  = "Moderate Focus"
        focus_advice = "Some distraction detected. Try shorter focus cycles."
    else:
        focus_state  = "Low Focus"
        focus_advice = "High distraction risk. Remove interruptions."

    # ── NEW: rich pattern detection ───────────────────────────────────────────
    # discover_behavior_patterns accepts the raw dict records directly
    detected_patterns = discover_behavior_patterns(records_dict)

    # ── Forecast (from behavior_forecaster, added in previous session) ────────
    try:
        forecast = forecast_cognitive_trajectory()
    except Exception:
        forecast = None

    return {
        "score":           cognitive_score,
        "trajectory":      trajectory_result.get("trajectory", "stable"),
        "strategy":        strategy_result.get("mode", "observe"),
        "risks":           dominant_risk,
        "strategy_reason": strategy_result.get("rationale", "No reasoning"),
        "weekly":          "stable",
        "advice":          advice,
        "prediction":      prediction,
        "anomaly":         anomaly,
        "focus_state":     focus_state,
        "focus_advice":    focus_advice,
        "patterns":        detected_patterns,   # ← now a list[str]
        "forecast":        forecast,
    }

nlp = spacy.load("en_core_web_sm")

from behavioral_engine import (
    NoteRecord,
    BehavioralTrajectoryTracker,
    AIStrategyEngine,
    PredictiveRiskModel
)
from datetime import datetime


def take_input():
    user_text = input("Enter your note for today: ")
    return user_text

def save_to_file(text):
    from datetime import datetime

    date = datetime.now().strftime("%Y-%m-%d")

    with open("memory.txt", "a") as file:
        file.write(f"[{date}] {text}\n")

def confirm_saved():
    print("Your note has been stored successfully.")

def read_notes():
    try:
        with open("memory.txt", "r") as file:
            notes = file.readlines()
            return notes
    except FileNotFoundError:
        return []
    
def extract_people(doc):
    people = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.append(ent.text)

    return list(set(people))

def summarize_notes(notes):
    if not notes:
        return "No notes found."

    urgency = []
    tasks = []
    all_people = []

    for note in notes:
        doc = nlp(note)

        # Extract real people only
        people = extract_people(doc)
        all_people.extend(people)

        # Extract timelines
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                urgency.append(ent.text)

        # Extract meaningful actions
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                tasks.append(token.lemma_)

    insight = []

    # People
    unique_people = list(set(all_people))
    if unique_people:
        insight.append("People involved: " + ", ".join(unique_people))
    else:
        insight.append("People involved: None detected")

    # Timelines
    if urgency:
        insight.append("Upcoming timelines: " + ", ".join(set(urgency)))

    # Actions
    if tasks:
        insight.append("Action words detected: " + ", ".join(set(tasks)))

    if len(insight) == 1 and insight[0] == "People involved: None detected":
        return "Nothing meaningful detected."

    return "Context Insight:\n" + "\n".join(insight)

def generate_priority(insight_text):
    priority_keywords = ["deadline", "tomorrow", "urgent", "meeting", "submit", "exam", "health"]

    priority = []

    for word in priority_keywords:
        if word in insight_text.lower():
            priority.append(word)

    if priority:
        return "⚠ Priority Alert: " + ", ".join(set(priority))

    return "No urgent priorities detected."

def detect_patterns(notes):
    word_count = {}

    for note in notes:
        doc = nlp(note)

        for token in doc:
            if (
                token.pos_ in ["NOUN", "PROPN", "VERB"]
                and not token.is_stop
                and not token.is_punct
                and len(token.text) > 2
            ):
                word = token.lemma_.lower()

                if word not in word_count:
                    word_count[word] = 0
                word_count[word] += 1

    repeated = [word for word, count in word_count.items() if count > 2]

    if repeated:
        return "🔁 Repeated focus areas: " + ", ".join(repeated[:5])

    return "No strong patterns yet."

def recent_focus(notes):
    recent = notes[-5:]  # last 5 notes
    focus_words = ["deadline", "meeting", "exam", "submit"]

    detected = []

    for note in recent:
        for word in focus_words:
            if word in note.lower():
                detected.append(word)

    if detected:
        return "🕒 Recent focus: " + ", ".join(set(detected))

    return "No recent urgency."

def predict_pressure(notes):
    pressure_words = ["deadline", "exam", "submit", "urgent", "tomorrow"]
    risk_score = 0

    for note in notes[-7:]:   # check last 7 notes
        for word in pressure_words:
            if word in note.lower():
                risk_score += 1

    if risk_score >= 5:
        return "🚨 Rising pressure detected — upcoming commitments may stack."

    elif risk_score >= 3:
        return "⚡ Moderate pressure building."

    else:
        return "Low upcoming pressure."
    
def generate_daily_report(summary, priority, patterns, recent, pressure, suggestion, intent, stall, semantic_stall, ai_insight):

    report = "\n📊 DAILY CONTEXT REPORT\n"
    report += "-------------------------\n"
    report += summary + "\n\n"
    report += priority + "\n"
    report += patterns + "\n"
    report += recent + "\n"
    report += pressure + "\n\n"
    report += intent + "\n\n"
    report += suggestion + "\n"
    report += stall + "\n"
    report += semantic_stall + "\n"
    report += "\n🧠 AI Behavior Analysis:\n"
    report += ai_insight + "\n"
    report += "-------------------------\n"
    return report

def generate_suggestion(summary, priority, patterns, behavior):
    advice = []

    if "deadline" in priority.lower():
        advice.append("Plan your work immediately.")

    if "health" in summary.lower():
        advice.append("Protect time for wellbeing.")

    if "exam" in summary.lower():
        advice.append("Allocate dedicated study time.")

    if "meeting" in summary.lower():
        advice.append("Prepare agenda beforehand.")

    if "project" in patterns.lower():
        advice.append("This is recurring — consistency matters.")

    if not advice:
        base = "No strong suggestion right now."
    else:
        base = " ".join(advice)

    # Tone adaptation
    if behavior == "low_follow_through":
        return "💡 Strong Suggestion: " + base + " Consider scheduling it now."
    elif behavior == "high_follow_through":
        return "💡 Suggestion: " + base + " You’re doing well—stay consistent."
    else:
        return "💡 Suggestion: " + base

def record_feedback():
    feedback = input("Did you follow today's suggestion? (yes/no): ").lower()

    with open("feedback.txt", "a") as file:
        file.write(feedback + "\n")

    print("Feedback recorded.")

def analyze_feedback():
    try:
        with open("feedback.txt", "r") as file:
            feedbacks = file.readlines()

        yes_count = sum(1 for f in feedbacks if "yes" in f)
        no_count = sum(1 for f in feedbacks if "no" in f)

        if no_count > yes_count:
            return "⚠ You often ignore suggestions — consider stronger planning."

        elif yes_count > no_count:
            return "✅ You follow through well — keep consistency."

        else:
            return "Neutral response pattern."

    except:
        return "No feedback data yet."
    
def get_behavior_profile():
    try:
        with open("feedback.txt", "r") as file:
            feedbacks = [f.strip().lower() for f in file.readlines()]

        yes_count = sum(1 for f in feedbacks if "yes" in f)
        no_count = sum(1 for f in feedbacks if "no" in f)

        if no_count > yes_count:
            return "low_follow_through"
        elif yes_count > no_count:
            return "high_follow_through"
        else:
            return "neutral"
    except:
        return "neutral"
    
def detect_intent(notes):
    intents = {"plan":0, "delay":0, "decide":0, "start":0, "finish":0}

    for note in notes:
        doc = nlp(note)
        for token in doc:
            if token.lemma_ in intents and token.pos_ == "VERB":
                intents[token.lemma_] += 1

    active = [k for k,v in intents.items() if v > 0]

    if active:
        return "🧭 Intent signals: " + ", ".join(active)
    return "No strong intent signals."

def detect_stall(notes):
    delay_words = ["tomorrow", "later", "soon", "after", "next"]

    delay_count = 0

    for note in notes:
        for word in delay_words:
            if word in note.lower():
                delay_count += 1

    if delay_count >= 3:
        return "⚠️ Repeated postponement detected."
    
    return "No delay pattern."

def semantic_similarity(note1, note2):
    emb1 = model.encode(note1)
    emb2 = model.encode(note2)

    similarity = (emb1 @ emb2) / ((emb1 @ emb1)**0.5 * (emb2 @ emb2)**0.5)

    return similarity

def detect_semantic_stall(notes):
    if not AI_AVAILABLE or model is None:
        return "AI analysis unavailable."

    if len(notes) < 3:
        return "Not enough data for semantic analysis."

    try:
        embeddings = model.encode(notes)
        similarity = embeddings @ embeddings.T

        avg_similarity = similarity.mean()

        if avg_similarity > 0.8:
            return "High repetition in thinking patterns detected."
        else:
            return "Healthy thought variation."
    except:
        return "AI semantic analysis failed."

def get_today_notes(notes):
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    today_notes = []

    for note in notes:
        if today in note:
            today_notes.append(note)

    return today_notes

def predict_behavior_risk(notes):
    delay_words = ["later", "tomorrow", "soon", "after"]
    count = 0

    for note in notes:
        for word in delay_words:
            if word in note.lower():
                count += 1

    if count >= 3:
        return "⚠️ Repeating delay pattern detected."
    return "No strong behavioral risk."

def analyze_behavior_state(notes):

    delay_words = ["later", "tomorrow", "evening", "after", "not now", "soon"]
    action_words = ["start", "finish", "submit", "complete", "prepare", "study"]
    stress_words = ["pressure", "urgent", "deadline", "exam", "meeting"]

    delay_score = 0
    action_score = 0
    stress_score = 0

    for note in notes:
        note = note.lower()

        for word in delay_words:
            if word in note:
                delay_score += 1

        for word in action_words:
            if word in note:
                action_score += 1

        for word in stress_words:
            if word in note:
                stress_score += 1

    return delay_score, action_score, stress_score

def generate_ai_insight(notes):

    delay, action, stress = analyze_behavior_state(notes)

    if delay > action:
        behavior = "⚠️ Delay tendency increasing."
    elif action > delay:
        behavior = "✅ Execution momentum detected."
    else:
        behavior = "Balanced cognitive state."

    if stress > 3:
        stress_state = "⚡ High stress signals detected."
    elif stress > 0:
        stress_state = "Moderate pressure building."
    else:
        stress_state = "Low stress signals."

    return behavior + "\n" + stress_state

def calculate_cognitive_score(notes):

    delay_words = ["later", "tomorrow", "soon", "after"]
    action_words = ["start", "finish", "submit", "complete", "prepare", "study"]
    stress_words = ["pressure", "urgent", "deadline", "exam", "meeting"]

    delay = 0
    action = 0
    stress = 0

    for note in notes:
        note = note.lower()

        for word in delay_words:
            if word in note:
                delay += 1

        for word in action_words:
            if word in note:
                action += 1

        for word in stress_words:
            if word in note:
                stress += 1

    score = 50
    score += action * 5
    score -= delay * 4
    score -= stress * 3

    return max(0, min(100, score))

from behavioral_engine import NoteRecord
from datetime import datetime

def build_note_records(notes):

    records = []

    for note in notes:

        # Extract date from [YYYY-MM-DD] format
        try:
            date_str = note.split("]")[0].replace("[", "")
            note_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            note_date = datetime.today().date()

        # Basic signals
        text = note.lower()

        action_words = ["start", "finish", "complete", "submit", "prepare", "study"]
        delay_words = ["later", "tomorrow", "soon", "after"]
        stress_words = ["deadline", "urgent", "exam", "pressure"]

        action_count = sum(word in text for word in action_words)
        delay_count = sum(word in text for word in delay_words)
        stress_signals = sum(word in text for word in stress_words)

        cognitive_score = 50 + (action_count * 5) - (delay_count * 4) - (stress_signals * 3)
        cognitive_score = max(0, min(100, cognitive_score))

        records.append(
            NoteRecord(
                date=note_date,
                cognitive_score=cognitive_score,
                action_count=action_count,
                delay_count=delay_count,
                stress_signals=stress_signals,
                themes=[],
                raw_text=note
            )
        )

    return records

def convert_notes_to_records(notes):

    records = []

    for note in notes:
        try:
            date_str = note.split("]")[0].replace("[", "")
            note_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            note_date = datetime.today().date()

        delay_words = ["later", "tomorrow", "after", "soon"]
        action_words = ["start", "finish", "submit", "prepare", "study"]
        stress_words = ["deadline", "urgent", "exam", "pressure"]

        delay_count = sum(1 for w in delay_words if w in note.lower())
        action_count = sum(1 for w in action_words if w in note.lower())
        stress_count = sum(1 for w in stress_words if w in note.lower())

        cognitive_score = calculate_cognitive_score([note])

        records.append(
            NoteRecord(
                date=note_date,
                cognitive_score=cognitive_score,
                action_count=action_count,
                delay_count=delay_count,
                stress_signals=stress_count,
                raw_text=note
            )
        )

    return records

def calculate_cognitive_score(notes):

    delay_words = ["later", "tomorrow", "after"]
    action_words = ["start", "finish", "complete", "submit"]

    delay = 0
    action = 0

    for note in notes:
        note = note.lower()
        for word in delay_words:
            if word in note:
                delay += 1
        for word in action_words:
            if word in note:
                action += 1

    score = 50 + (action * 5) - (delay * 5)

    return max(0, min(100, score))

from behavioral_engine import NoteRecord
from datetime import datetime

def build_note_records(notes):

    records = []

    for note in notes:

        # Extract date from stored format: [YYYY-MM-DD] text
        try:
            date_part = note.split("]")[0].replace("[", "")
            note_date = datetime.strptime(date_part, "%Y-%m-%d").date()
            note_text = note.split("]", 1)[1].strip()
        except:
            note_date = datetime.today().date()
            note_text = note.strip()

        # LLM signal extraction (falls back to keywords if Gemini unavailable)
        signals = analyze_note_with_gemini(note_text)

        record = NoteRecord(
            date            = note_date,
            cognitive_score = signals["cognitive_score"],
            action_count    = signals["action_count"],
            delay_count     = signals["delay_count"],
            stress_signals  = signals["stress_signals"],
            themes          = signals.get("themes", []),
            raw_text        = note
        )

        records.append(record)

    return records

def run_behavioral_ai(notes):

    records = build_note_records(notes)

    cognitive_score = calculate_cognitive_score(notes)

    tracker = BehavioralTrajectoryTracker()
    trajectory_result = tracker.analyse(records)

    risk_model = PredictiveRiskModel()
    risk_result = risk_model.analyse(records)

    strategy_engine = AIStrategyEngine()
    strategy_result = strategy_engine.select_mode(
        cognitive_score,
        trajectory_result,
        risk_result
    )

    # Prediction engine
    try:
        prediction = predict_next_score()
    except:
        prediction = "Prediction engine initializing"

    # Anomaly detection
    try:
        anomaly = detect_behavior_anomaly()
    except:
        anomaly = "Behavior learning phase"

    dominant_risk = risk_result.get("dominant_risk", "none")

    advice = "🔄 Maintain systems — consistency is working."

    if dominant_risk == "burnout":
        advice = "Prioritize recovery before performance."

    elif dominant_risk == "procrastination":
        advice = "Reduce distractions and enforce focus blocks."

    # Focus analysis
    if cognitive_score > 70:
        focus_state = "deep"
        focus_advice = "🚀 Strong focus signals. Continue current workflow."

    elif cognitive_score > 40:
        focus_state = "focused"
        focus_advice = "Good momentum. Maintain structured work."

    else:
        focus_state = "distracted"
        focus_advice = "⚠ Distraction risk rising. Reduce interruptions."

    return {
        "score": cognitive_score,
        "trajectory": trajectory_result.get("trajectory","stable"),
        "strategy": strategy_result.get("mode","observe"),
        "risks": dominant_risk,
        "strategy_reason": strategy_result.get("rationale","No reasoning available"),
        "weekly": "Stable weekly direction.",
        "advice": advice,
        "focus_state": focus_state,
        "focus_advice": focus_advice,
        "prediction": prediction,
        "anomaly": anomaly
    }

def select_ai_strategy(trajectory, cognitive_score):

    if trajectory == "declining" or cognitive_score < 40:
        return {
            "mode": "Recovery",
            "advice": "Reduce workload and focus on restoration."
        }

    elif trajectory == "improving" and cognitive_score > 60:
        return {
            "mode": "Growth",
            "advice": "Increase challenge and pursue meaningful progress."
        }

    elif trajectory == "stable":
        return {
            "mode": "Stability",
            "advice": "Maintain systems and avoid unnecessary change."
        }

    else:
        return {
            "mode": "Structure",
            "advice": "Build routine and improve consistency."
        }

def clean_notes(notes):

    cleaned = []

    for note in notes:
        if "]" in note:
            cleaned.append(note.split("]")[1])
        else:
            cleaned.append(note)

    return cleaned

from datetime import datetime
from behavioral_engine import NoteRecord


def build_note_records(notes):
    """
    Convert raw note strings into NoteRecord objects.

    Signal extraction is now handled by analyze_note_with_gemini().
    If Gemini is unavailable the function falls back to keyword scoring
    automatically — the rest of the pipeline never knows the difference.

    Expected note format:  "[YYYY-MM-DD] note text here"

    Returns: list[NoteRecord]
    Consumed by: BehavioralTrajectoryTracker, PredictiveRiskModel,
                 AIStrategyEngine, BehaviorPatternEngine  (all unchanged)
    """
    records = []

    for note in notes:

        # ── Extract date ──────────────────────────────────────────────────────
        try:
            date_str    = note.split("]")[0].replace("[", "").strip()
            record_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            note_text   = note.split("]", 1)[1].strip()
        except Exception:
            record_date = datetime.now().date()
            note_text   = note.strip()

        # ── LLM signal extraction (auto-falls back to keywords on failure) ────
        signals = analyze_note_with_gemini(note_text)

        # ── Build NoteRecord — same contract as before ────────────────────────
        records.append(
            NoteRecord(
                date            = record_date,
                cognitive_score = signals["cognitive_score"],
                action_count    = signals["action_count"],
                delay_count     = signals["delay_count"],
                stress_signals  = signals["stress_signals"],
                themes          = signals.get("themes", []),
                raw_text        = note,
            )
        )

    return records


def calculate_cognitive_score(notes):

    if not notes:
        return 50

    score = 50

    for note in notes:
        if "start" in note.lower():
            score += 3
        if "finish" in note.lower():
            score += 3
        if "deadline" in note.lower():
            score -= 4
        if "later" in note.lower():
            score -= 3

    return max(0, min(100, score))

def detect_weekly_drift():

    try:
        with open("behavior_memory.json", "r") as f:
            memory = json.load(f)
    except:
        return "No long-term data yet."

    if len(memory) < 7:
        return "Not enough weekly data."

    last_week = memory[-7:]

    improving = sum(1 for day in last_week if day["trajectory"] == "improving")
    declining = sum(1 for day in last_week if day["trajectory"] == "declining")

    if declining > improving:
        return "⚠ Behavioral drift trending downward this week."
    elif improving > declining:
        return "📈 Positive weekly behavioral shift detected."
    else:
        return "Stable weekly direction."
    
def adaptive_ai_advice(ai):

    trajectory = ai.get("trajectory", "stable")
    risk = ai.get("risks", "low")

    if trajectory == "declining":
        return "⚠ Reduce workload. Focus only on critical tasks."

    elif risk == "burnout":
        return "🧘 Prioritize recovery before performance."

    elif trajectory == "improving":
        return "🚀 Increase challenge — growth phase detected."

    elif trajectory == "stable":
        return "🔄 Maintain systems — consistency is working."

    else:
        return "Observe patterns — no strong shift yet."

import json
from datetime import datetime


def generate_weekly_insight():

    try:
        with open("behavior_history.json", "r") as f:
            history = json.load(f)

        if len(history) < 2:
            return "Not enough history for weekly insight."

        last_week = history[-7:]

        scores = [entry["score"] for entry in last_week]
        avg_score = sum(scores) / len(scores)

        # trajectory trend
        trend = scores[-1] - scores[0]

        if trend > 5:
            trend_text = "Improving"
        elif trend < -5:
            trend_text = "Declining"
        else:
            trend_text = "Stable"

        # dominant risk
        risks = [entry["risk"] for entry in last_week]
        dominant_risk = max(set(risks), key=risks.count)

        insight = (
            f"📊 Weekly Behavioral Insight\n"
            f"Average Score: {round(avg_score,1)}\n"
            f"Trend: {trend_text}\n"
            f"Dominant Risk: {dominant_risk}"
        )

        return insight

    except:
        return "Weekly insight unavailable."

def detect_focus_drift(notes):

    if len(notes) < 3:
        return "insufficient_data"

    distraction_words = [
        "youtube","instagram","reels","scrolling",
        "wasting time","procrastination","gaming"
    ]

    study_words = [
        "study","exam","project","coding",
        "lecture","assignment","learning"
    ]

    distraction_score = 0
    study_score = 0

    for n in notes:
        text = n.lower()

        for w in distraction_words:
            if w in text:
                distraction_score += 1

        for w in study_words:
            if w in text:
                study_score += 1

    if distraction_score > study_score:
        return "focus_drift"

    if study_score > distraction_score:
        return "focused"

    return "neutral"

def focus_recovery_strategy(state):

    if state == "focus_drift":
        return "⚠ Focus drifting. Try a 25-minute deep work session."

    if state == "focused":
        return "🚀 Strong focus signals. Continue current workflow."

    return "Maintain balanced attention."

def main():
    print("1. Add Note")
    print("2. View Summary")
    print("3. Quick Add (one-line)")

    choice = input("Choose option: ")

    if choice == "1":
        note = take_input()
        save_to_file(note)
        confirm_saved()

        view = input("Generate report now? (y/n): ").lower()
        if view == "y":
            notes = read_notes()
            today_notes = get_today_notes(notes)
            ai_insight = generate_ai_insight(today_notes)
            summary = summarize_notes(today_notes)
            priority = generate_priority(summary)
            patterns = detect_patterns(today_notes)
            recent = recent_focus(today_notes)
            pressure = predict_pressure(today_notes)
            intent = detect_intent(today_notes)

            stall = detect_stall(notes)
            semantic_stall = detect_semantic_stall(notes)

            behavior_profile = get_behavior_profile()
            suggestion = generate_suggestion(summary, priority, patterns, behavior_profile)

            report = generate_daily_report(summary, priority, patterns, recent, pressure, suggestion, intent, stall, semantic_stall, ai_insight)

            print(report)
            record_feedback()

    elif choice == "2":
        notes = read_notes()
        today_notes = get_today_notes(notes)

        summary = summarize_notes(today_notes)
        priority = generate_priority(summary)
        patterns = detect_patterns(today_notes)
        recent = recent_focus(today_notes)
        pressure = predict_pressure(today_notes)
        intent = detect_intent(today_notes)

        stall = detect_stall(notes)

        behavior_profile = get_behavior_profile()
        suggestion = generate_suggestion(summary, priority, patterns, behavior_profile)

        report = generate_daily_report(summary, priority, patterns, recent, pressure, suggestion, intent, stall)

        print(report)
        record_feedback()

    elif choice == "3":
        note = input("Quick note: ")
        save_to_file(note)
        print("Saved.")

        view = input("Generate report now? (y/n): ").lower()
        if view == "y":
            notes = read_notes()
            today_notes = get_today_notes(notes)

            summary = summarize_notes(today_notes)
            priority = generate_priority(summary)
            patterns = detect_patterns(today_notes)
            recent = recent_focus(today_notes)
            pressure = predict_pressure(today_notes)
            intent = detect_intent(today_notes)

            stall = detect_stall(notes)

            behavior_profile = get_behavior_profile()
            suggestion = generate_suggestion(summary, priority, patterns, behavior_profile)

            report = generate_daily_report(summary, priority, patterns, recent, pressure, suggestion, intent, stall)

            print(report)
            record_feedback()

    else:
        print("Invalid choice")