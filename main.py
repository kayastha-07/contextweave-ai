from behavioral_engine import (
    NoteRecord,
    BehavioralTrajectoryTracker,
    AIStrategyEngine,
    PredictiveRiskModel
)
from prediction_engine import predict_next_score
from anomaly_engine import detect_behavior_anomaly
from habit_engine import discover_behavior_patterns

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    AI_AVAILABLE = True
except:
    model = None
    AI_AVAILABLE = False
# spacy removed — replaced with pure-Python NLP (no C++ build dependencies)
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

    # Convert dictionaries → objects so engine can use r.date
    from types import SimpleNamespace
    records = []

    for r in records_dict:
        if isinstance(r, dict):
            records.append(SimpleNamespace(**r))
        else:
            records.append(r)

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
        focus_state = "Deep Work"
        focus_advice = "Excellent focus. Maintain current routine."

    elif cognitive_score > 40:
        focus_state = "Moderate Focus"
        focus_advice = "Some distraction detected. Try shorter focus cycles."

    else:
        focus_state = "Low Focus"
        focus_advice = "High distraction risk. Remove interruptions."

    patterns = discover_behavior_patterns()

    return {
        "score": cognitive_score,
        "trajectory": trajectory_result.get("trajectory", "stable"),
        "strategy": strategy_result.get("mode", "observe"),
        "risks": dominant_risk,
        "strategy_reason": strategy_result.get("rationale", "No reasoning"),
        "weekly": "stable",
        "advice": advice,
        "prediction": prediction,
        "anomaly": anomaly,
        "focus_state": focus_state,
        "focus_advice": focus_advice,
        "patterns": patterns
    }

# ── Pure-Python NLP replacement (no spaCy / no C++ build deps) ──────────────
import re as _re

# Common English stopwords (replaces spaCy's is_stop)
_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it",
    "its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were",
    "be","been","being","have","has","had","having","do","does","did","doing",
    "a","an","the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","both","each","few","more","most","other","some",
    "such","no","nor","not","only","own","same","so","than","too","very","s",
    "t","can","will","just","don","should","now","d","ll","m","o","re","ve",
    "y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
    "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
}

# Common English verbs (intent detection)
_VERB_ROOTS = {
    "plan","delay","decide","start","finish","complete","submit","study",
    "work","meet","review","check","write","read","prepare","attend",
    "cancel","skip","postpone","focus","rest","exercise","call","send",
}

# Common name patterns — simple heuristic (Capitalised word not at start of sentence)
def _extract_people(text):
    words = text.split()
    people = []
    for i, w in enumerate(words):
        clean = w.strip(".,!?;:")
        if (i > 0 and clean and clean[0].isupper()
                and clean.lower() not in _STOPWORDS
                and len(clean) > 2
                and not clean.isupper()):   # skip ALL-CAPS acronyms
            people.append(clean)
    return list(set(people))

def _tokenize(text):
    """Return lowercase word tokens, no punctuation."""
    return [w.lower() for w in _re.findall(r"[a-zA-Z']+", text) if len(w) > 1]

def _lemma(word):
    """Tiny rule-based lemmatizer covering common English inflections."""
    w = word.lower()
    for suffix, replacement in [
        ("ying","y"),("ied","y"),("ies","y"),
        ("ing",""),("ed",""),("er",""),("est",""),("ly",""),
        ("tion","t"),("ness",""),("ment",""),("ful",""),("less",""),
    ]:
        if w.endswith(suffix) and len(w) - len(suffix) > 2:
            return w[:-len(suffix)] + replacement
    return w

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

    time_keywords = ["today","tomorrow","monday","tuesday","wednesday","thursday",
                     "friday","saturday","sunday","morning","evening","night",
                     "week","month","deadline","hour","minutes"]

    for note in notes:
        # People (capitalised word heuristic)
        all_people.extend(extract_people(note))

        # Timeline keywords
        note_lower = note.lower()
        for kw in time_keywords:
            if kw in note_lower:
                urgency.append(kw)

        # Action verbs
        for word in _tokenize(note):
            lemma = _lemma(word)
            if lemma in _VERB_ROOTS and word not in _STOPWORDS:
                tasks.append(lemma)

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
        for word in _tokenize(note):
            if word not in _STOPWORDS and len(word) > 2:
                lemma = _lemma(word)
                word_count[lemma] = word_count.get(lemma, 0) + 1

    repeated = [word for word, count in sorted(word_count.items(),
                key=lambda x: -x[1]) if count > 2]

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
    intent_variants = {
        "plan":   ["plan","plans","planned","planning"],
        "delay":  ["delay","delays","delayed","delaying","postpone","postponed"],
        "decide": ["decide","decided","deciding","decision"],
        "start":  ["start","starts","started","starting","begin","began","beginning"],
        "finish": ["finish","finished","finishing","complete","completed","done"],
    }
    for note in notes:
        words = _tokenize(note)
        for intent, variants in intent_variants.items():
            for word in words:
                if word in variants:
                    intents[intent] += 1

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

        # Extract date from your stored note format
        # Example: [2026-03-03] Study ML
        try:
            date_part = note.split("]")[0].replace("[", "")
            note_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except:
            note_date = datetime.today().date()

        # Simple signal extraction (can upgrade later)
        text = note.lower()

        action_words = ["start", "finish", "submit", "complete", "study", "prepare"]
        delay_words = ["later", "tomorrow", "soon", "after"]
        stress_words = ["deadline", "exam", "urgent", "pressure"]

        action_count = sum(word in text for word in action_words)
        delay_count = sum(word in text for word in delay_words)
        stress_signals = sum(word in text for word in stress_words)

        record = NoteRecord(
            date=note_date,
            cognitive_score=calculate_cognitive_score([note]),
            action_count=action_count,
            delay_count=delay_count,
            stress_signals=stress_signals,
            themes=[],
            raw_text=note
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

def build_note_records(notes):

    records = []

    for note in notes:

        # extract date from stored format: [YYYY-MM-DD] text
        try:
            date_str = note.split("]")[0].replace("[","")
            record_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            record_date = datetime.now().date()

        # basic behavior signals
        delay_words = ["later", "tomorrow", "after", "soon"]
        action_words = ["start", "finish", "submit", "complete", "prepare"]
        stress_words = ["urgent", "pressure", "deadline", "exam"]

        delay = sum(word in note.lower() for word in delay_words)
        action = sum(word in note.lower() for word in action_words)
        stress = sum(word in note.lower() for word in stress_words)

        cognitive = 50 + (action * 5) - (delay * 4) - (stress * 3)
        cognitive = max(0, min(100, cognitive))

        records.append({
            "date": record_date,
            "cognitive_score": cognitive,
            "action_count": action,
            "delay_count": delay,
            "stress_signals": stress,
            "themes": [],
            "raw_text": note
        })

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