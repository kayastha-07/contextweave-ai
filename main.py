"""
ContextWeave — main.py  (clean, deduplicated)
All duplicate function definitions removed.
spaCy fully replaced with pure-Python NLP.
"""

import json
import os
import re as _re
from datetime import datetime

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
except Exception:
    model = None
    AI_AVAILABLE = False


# ── Pure-Python NLP ───────────────────────────────────────────────────────────

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

_VERB_ROOTS = {
    "plan","delay","decide","start","finish","complete","submit","study",
    "work","meet","review","check","write","read","prepare","attend",
    "cancel","skip","postpone","focus","rest","exercise","call","send",
}


def _extract_people(text):
    words = text.split()
    people = []
    for i, word in enumerate(words):
        clean = _re.sub(r"[^A-Za-z]", "", word)
        if (len(clean) >= 2 and clean[0].isupper()
                and clean.lower() not in _STOPWORDS and i != 0):
            people.append(clean)
    return list(set(people))


def _tokenize(text):
    return _re.findall(r"[a-z]+", text.lower())


def _lemma(word):
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


# ── File helpers ──────────────────────────────────────────────────────────────

def save_to_file(text):
    date = datetime.now().strftime("%Y-%m-%d")
    with open("memory.txt", "a") as f:
        f.write(f"[{date}] {text}\n")

def confirm_saved():
    print("Your note has been stored successfully.")

def read_notes():
    try:
        with open("memory.txt", "r") as f:
            return f.readlines()
    except FileNotFoundError:
        return []

def save_behavior_state(ai_output):
    file_path = "behavior_history.json"
    state = {
        "date":       datetime.now().strftime("%Y-%m-%d"),
        "score":      ai_output.get("score", 0),
        "trajectory": ai_output.get("trajectory", "stable"),
        "strategy":   ai_output.get("strategy", "observe"),
        "risk":       ai_output.get("risks", ai_output.get("risk", "none")),
    }
    history = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(state)
    with open(file_path, "w") as f:
        json.dump(history, f, indent=2)


# ── Core scoring ──────────────────────────────────────────────────────────────

def calculate_cognitive_score(notes):
    delay_words  = ["later", "tomorrow", "soon", "after"]
    action_words = ["start", "finish", "submit", "complete", "prepare", "study"]
    stress_words = ["pressure", "urgent", "deadline", "exam", "meeting"]
    delay = action = stress = 0
    for note in notes:
        text = note.lower()
        delay  += sum(w in text for w in delay_words)
        action += sum(w in text for w in action_words)
        stress += sum(w in text for w in stress_words)
    return max(0, min(100, 50 + action*5 - delay*4 - stress*3))

def detect_trajectory(past_scores):
    if len(past_scores) < 3:
        return "insufficient"
    r = past_scores[-3:]
    if r[-1] > r[0] + 5:   return "improving"
    if r[-1] < r[0] - 5:   return "declining"
    return "stable"

def select_strategy(score, trajectory):
    if trajectory == "declining" or score < 40:  return "Recovery"
    if trajectory == "improving" and score > 60: return "Growth"
    if trajectory == "stable":                   return "Stability"
    return "Structure"


# ── Record builder ────────────────────────────────────────────────────────────

def build_note_records(notes):
    records = []
    action_words  = ["start","finish","submit","complete","study","prepare"]
    delay_words   = ["later","tomorrow","soon","after"]
    stress_words  = ["deadline","exam","urgent","pressure"]
    for note in notes:
        try:
            date_str  = note.split("]")[0].replace("[","").strip()
            note_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            note_date = datetime.today().date()
        text    = note.lower()
        ac      = sum(w in text for w in action_words)
        dc      = sum(w in text for w in delay_words)
        ss      = sum(w in text for w in stress_words)
        cog     = max(0, min(100, 50 + ac*5 - dc*4 - ss*3))
        records.append(NoteRecord(
            date=note_date, cognitive_score=cog,
            action_count=ac, delay_count=dc, stress_signals=ss,
            themes=[], raw_text=note,
        ))
    return records

def clean_notes(notes):
    return [n.split("]")[1].strip() if "]" in n else n for n in notes]

def get_today_notes(notes):
    today = datetime.now().strftime("%Y-%m-%d")
    return [n for n in notes if today in n]

def convert_notes_to_records(notes):
    return build_note_records(notes)


# ── Main AI pipeline ──────────────────────────────────────────────────────────

def run_behavioral_ai(notes):
    from types import SimpleNamespace
    records_obj = build_note_records(notes)
    records_ns  = [r if not isinstance(r,dict) else SimpleNamespace(**r) for r in records_obj]
    cognitive_score   = calculate_cognitive_score(notes)
    tracker           = BehavioralTrajectoryTracker()
    trajectory_result = tracker.analyse(records_ns)
    risk_model        = PredictiveRiskModel()
    risk_result       = risk_model.analyse(records_ns)
    strategy_engine   = AIStrategyEngine()
    strategy_result   = strategy_engine.select_mode(cognitive_score, trajectory_result, risk_result)
    try:    prediction = predict_next_score()
    except: prediction = "Prediction engine initializing"
    try:    anomaly = detect_behavior_anomaly()
    except: anomaly = "Behavior learning phase"
    dominant_risk = risk_result.get("dominant_risk","none")
    advice = "🔄 Maintain systems — consistency is working."
    if dominant_risk == "burnout":         advice = "Prioritize recovery before performance."
    elif dominant_risk == "procrastination": advice = "Reduce distractions and enforce focus blocks."
    if cognitive_score > 70:
        focus_state, focus_advice = "deep",       "🚀 Strong focus signals. Continue current workflow."
    elif cognitive_score > 40:
        focus_state, focus_advice = "focused",    "Good momentum. Maintain structured work."
    else:
        focus_state, focus_advice = "distracted", "⚠ Distraction risk rising. Reduce interruptions."
    return {
        "score": cognitive_score, "trajectory": trajectory_result.get("trajectory","stable"),
        "strategy": strategy_result.get("mode","observe"), "risks": dominant_risk,
        "strategy_reason": strategy_result.get("rationale","No reasoning available"),
        "weekly": "Stable weekly direction.", "advice": advice,
        "focus_state": focus_state, "focus_advice": focus_advice,
        "prediction": prediction, "anomaly": anomaly,
    }


# ── Analysis helpers ──────────────────────────────────────────────────────────

def extract_people(text):
    if not isinstance(text, str): text = str(text)
    return _extract_people(text)

def summarize_notes(notes):
    if not notes: return "No notes found."
    urgency, tasks, all_people = [], [], []
    time_kw = ["today","tomorrow","monday","tuesday","wednesday","thursday",
                "friday","saturday","sunday","morning","evening","night",
                "week","month","deadline","hour","minutes"]
    for note in notes:
        all_people.extend(extract_people(note))
        nl = note.lower()
        for kw in time_kw:
            if kw in nl: urgency.append(kw)
        for word in _tokenize(note):
            lemma = _lemma(word)
            if lemma in _VERB_ROOTS and word not in _STOPWORDS:
                tasks.append(lemma)
    insight = ["People involved: " + (", ".join(set(all_people)) if all_people else "None detected")]
    if urgency: insight.append("Upcoming timelines: " + ", ".join(set(urgency)))
    if tasks:   insight.append("Action words detected: " + ", ".join(set(tasks)))
    if len(insight)==1 and "None detected" in insight[0]: return "Nothing meaningful detected."
    return "Context Insight:\n" + "\n".join(insight)

def generate_priority(insight_text):
    kw = ["deadline","tomorrow","urgent","meeting","submit","exam","health"]
    found = [w for w in kw if w in insight_text.lower()]
    return ("⚠ Priority Alert: " + ", ".join(set(found))) if found else "No urgent priorities detected."

def detect_patterns(notes):
    wc = {}
    for note in notes:
        for word in _tokenize(note):
            if word not in _STOPWORDS and len(word) > 2:
                l = _lemma(word); wc[l] = wc.get(l,0)+1
    repeated = [w for w,c in sorted(wc.items(), key=lambda x:-x[1]) if c>2]
    return ("🔁 Repeated focus areas: " + ", ".join(repeated[:5])) if repeated else "No strong patterns yet."

def recent_focus(notes):
    kw = ["deadline","meeting","exam","submit"]
    detected = [w for note in notes[-5:] for w in kw if w in note.lower()]
    return ("🕒 Recent focus: " + ", ".join(set(detected))) if detected else "No recent urgency."

def predict_pressure(notes):
    pw = ["deadline","exam","submit","urgent","tomorrow"]
    rs = sum(1 for n in notes[-7:] for w in pw if w in n.lower())
    if rs>=5: return "🚨 Rising pressure detected — upcoming commitments may stack."
    if rs>=3: return "⚡ Moderate pressure building."
    return "Low upcoming pressure."

def detect_intent(notes):
    intents = {"plan":0,"delay":0,"decide":0,"start":0,"finish":0}
    variants = {
        "plan":   ["plan","plans","planned","planning"],
        "delay":  ["delay","delays","delayed","delaying","postpone","postponed"],
        "decide": ["decide","decided","deciding","decision"],
        "start":  ["start","starts","started","starting","begin","began","beginning"],
        "finish": ["finish","finished","finishing","complete","completed","done"],
    }
    for note in notes:
        words = _tokenize(note)
        for intent, vs in variants.items():
            intents[intent] += sum(1 for w in words if w in vs)
    active = [k for k,v in intents.items() if v>0]
    return ("🧭 Intent signals: " + ", ".join(active)) if active else "No strong intent signals."

def detect_stall(notes):
    dw = ["tomorrow","later","soon","after","next"]
    count = sum(1 for n in notes for w in dw if w in n.lower())
    return "⚠️ Repeated postponement detected." if count>=3 else "No delay pattern."

def detect_semantic_stall(notes):
    if not AI_AVAILABLE or model is None: return "AI semantic analysis unavailable."
    if len(notes)<3: return "Not enough data for semantic analysis."
    try:
        emb = model.encode(notes)
        if (emb@emb.T).mean()>0.8: return "High repetition in thinking patterns detected."
        return "Healthy thought variation."
    except Exception: return "AI semantic analysis failed."

def analyze_behavior_state(notes):
    dw = ["later","tomorrow","evening","after","not now","soon"]
    aw = ["start","finish","submit","complete","prepare","study"]
    sw = ["pressure","urgent","deadline","exam","meeting"]
    d=a=s=0
    for n in notes:
        t=n.lower(); d+=sum(w in t for w in dw); a+=sum(w in t for w in aw); s+=sum(w in t for w in sw)
    return d,a,s

def generate_ai_insight(notes):
    d,a,s = analyze_behavior_state(notes)
    beh   = "⚠️ Delay tendency increasing." if d>a else "✅ Execution momentum detected." if a>d else "Balanced cognitive state."
    stress= "⚡ High stress signals detected." if s>3 else "Moderate pressure building." if s>0 else "Low stress signals."
    return beh+"\n"+stress

def get_behavior_profile():
    try:
        with open("feedback.txt","r") as f: fb=[l.strip().lower() for l in f]
        y=sum(1 for f in fb if "yes" in f); n=sum(1 for f in fb if "no" in f)
        return "low_follow_through" if n>y else "high_follow_through" if y>n else "neutral"
    except: return "neutral"

def generate_suggestion(summary, priority, patterns, behavior):
    adv=[]
    if "deadline" in priority.lower(): adv.append("Plan your work immediately.")
    if "health"   in summary.lower():  adv.append("Protect time for wellbeing.")
    if "exam"     in summary.lower():  adv.append("Allocate dedicated study time.")
    if "meeting"  in summary.lower():  adv.append("Prepare agenda beforehand.")
    if "project"  in patterns.lower(): adv.append("This is recurring — consistency matters.")
    base = " ".join(adv) if adv else "No strong suggestion right now."
    if behavior=="low_follow_through":  return "💡 Strong Suggestion: "+base+" Consider scheduling it now."
    if behavior=="high_follow_through": return "💡 Suggestion: "+base+" You're doing well — stay consistent."
    return "💡 Suggestion: "+base

def generate_daily_report(summary,priority,patterns,recent,pressure,suggestion,intent,stall,semantic_stall,ai_insight):
    r ="\n📊 DAILY CONTEXT REPORT\n-------------------------\n"
    r+=summary+"\n\n"+priority+"\n"+patterns+"\n"+recent+"\n"+pressure+"\n\n"
    r+=intent+"\n\n"+suggestion+"\n"+stall+"\n"+semantic_stall+"\n"
    r+="\n🧠 AI Behavior Analysis:\n"+ai_insight+"\n-------------------------\n"
    return r

def generate_weekly_insight():
    try:
        with open("behavior_history.json","r") as f: history=json.load(f)
        if len(history)<2: return "Not enough history for weekly insight."
        lw=history[-7:]; scores=[e["score"] for e in lw]; avg=sum(scores)/len(scores)
        trend=scores[-1]-scores[0]
        tt="Improving" if trend>5 else "Declining" if trend<-5 else "Stable"
        risks=[e["risk"] for e in lw]; dom=max(set(risks),key=risks.count)
        return f"📊 Weekly Behavioral Insight\nAverage Score: {round(avg,1)}\nTrend: {tt}\nDominant Risk: {dom}"
    except: return "Weekly insight unavailable."

def detect_focus_drift(notes):
    dw=["youtube","instagram","reels","scrolling","wasting time","procrastination","gaming"]
    sw=["study","exam","project","coding","lecture","assignment","learning"]
    d=sum(1 for n in notes for w in dw if w in n.lower())
    s=sum(1 for n in notes for w in sw if w in n.lower())
    return "focus_drift" if d>s else "focused" if s>d else "neutral"

def focus_recovery_strategy(state):
    if state=="focus_drift": return "⚠ Focus drifting. Try a 25-minute deep work session."
    if state=="focused":     return "🚀 Strong focus signals. Continue current workflow."
    return "Maintain balanced attention."

def record_feedback():
    fb=input("Did you follow today's suggestion? (yes/no): ").lower()
    with open("feedback.txt","a") as f: f.write(fb+"\n")
    print("Feedback recorded.")

def take_input(): return input("Enter your note for today: ")

def predict_behavior_risk(notes):
    dw=["later","tomorrow","soon","after"]
    c=sum(1 for n in notes for w in dw if w in n.lower())
    return "⚠️ Repeating delay pattern detected." if c>=3 else "No strong behavioral risk."

def select_ai_strategy(trajectory, cognitive_score):
    if trajectory=="declining" or cognitive_score<40:  return {"mode":"Recovery","advice":"Reduce workload."}
    if trajectory=="improving" and cognitive_score>60:  return {"mode":"Growth","advice":"Increase challenge."}
    if trajectory=="stable":                             return {"mode":"Stability","advice":"Maintain systems."}
    return {"mode":"Structure","advice":"Build routine."}

def detect_weekly_drift():
    try:
        with open("behavior_memory.json","r") as f: mem=json.load(f)
    except: return "No long-term data yet."
    if len(mem)<7: return "Not enough weekly data."
    lw=mem[-7:]
    imp=sum(1 for d in lw if d["trajectory"]=="improving")
    dec=sum(1 for d in lw if d["trajectory"]=="declining")
    if dec>imp: return "⚠ Behavioral drift trending downward this week."
    if imp>dec: return "📈 Positive weekly behavioral shift detected."
    return "Stable weekly direction."

def adaptive_ai_advice(ai):
    t=ai.get("trajectory","stable"); r=ai.get("risks","low")
    if t=="declining":  return "⚠ Reduce workload. Focus only on critical tasks."
    if r=="burnout":    return "🧘 Prioritize recovery before performance."
    if t=="improving":  return "🚀 Increase challenge — growth phase detected."
    if t=="stable":     return "🔄 Maintain systems — consistency is working."
    return "Observe patterns — no strong shift yet."


def main():
    print("1. Add Note\n2. View Summary\n3. Quick Add (one-line)")
    choice=input("Choose option: ")
    if choice in ("1","3"):
        note = take_input() if choice=="1" else input("Quick note: ")
        save_to_file(note); confirm_saved()
        if input("Generate report now? (y/n): ").lower()=="y":
            notes=read_notes(); tn=get_today_notes(notes)
            ai=generate_ai_insight(tn); s=summarize_notes(tn); p=generate_priority(s)
            pat=detect_patterns(tn); r=recent_focus(tn); pr=predict_pressure(tn)
            i=detect_intent(tn); st=detect_stall(notes); sem=detect_semantic_stall(notes)
            prof=get_behavior_profile(); sug=generate_suggestion(s,p,pat,prof)
            print(generate_daily_report(s,p,pat,r,pr,sug,i,st,sem,ai)); record_feedback()
    elif choice=="2":
        notes=read_notes(); tn=get_today_notes(notes)
        s=summarize_notes(tn); p=generate_priority(s); pat=detect_patterns(tn)
        r=recent_focus(tn); pr=predict_pressure(tn); i=detect_intent(tn)
        st=detect_stall(notes); sem=detect_semantic_stall(notes)
        prof=get_behavior_profile(); sug=generate_suggestion(s,p,pat,prof)
        print(generate_daily_report(s,p,pat,r,pr,sug,i,st,sem,generate_ai_insight(tn))); record_feedback()

if __name__=="__main__":
    main()