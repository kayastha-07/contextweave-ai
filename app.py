"""
ContextWeave — app.py  v2.0
============================
FastAPI application.
Runs with: uvicorn app:app --reload

Endpoints (all existing endpoints preserved):
  GET  /                    — health check
  POST /signup              — create account
  POST /login               — get JWT token
  GET  /login-page          — login HTML
  GET  /signup-page         — signup HTML
  GET  /dashboard           — main dashboard HTML
  POST /add-note            — save note (protected)
  GET  /report              — full AI report (protected)
  GET  /behavior-data       — chart data (protected)
  POST /simulate            — NEW: behavior simulation (protected)
"""
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
import json, os

# ── Database + Auth ────────────────────────────────────────────────────────────
from database import init_db, get_db, User, Note
from auth import hash_password, verify_password, create_token, get_current_user

# ── Existing main.py pipeline (no changes to any function) ────────────────────
from main import (
    detect_intent,
    detect_semantic_stall,
    detect_stall,
    get_today_notes,
    run_behavioral_ai,
    save_to_file,
    read_notes,
    summarize_notes,
    generate_priority,
    detect_patterns,
    recent_focus,
    predict_pressure,
    get_behavior_profile,
    generate_suggestion,
    generate_daily_report,
    generate_ai_insight,
    build_note_records,
    calculate_cognitive_score,
    detect_trajectory,
    select_strategy,
    BehavioralTrajectoryTracker,
    clean_notes,
    generate_weekly_insight,
)

# ── New v2 modules ─────────────────────────────────────────────────────────────
from behavior_simulation_engine import simulate_scenario
from insight_engine import generate_insights
from behavior_pattern_engine import discover_behavior_patterns

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="ContextWeave", version="2.0")
templates = Jinja2Templates(directory="templates")

# Mount static directory if it exists
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup():
    init_db()


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class NoteInput(BaseModel):
    text: str

class SignupInput(BaseModel):
    email: str
    password: str

class LoginInput(BaseModel):
    email: str
    password: str

class SimulateInput(BaseModel):
    scenario: str


# ══════════════════════════════════════════════════════════════════════════════
# Public routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def home():
    return {"message": "ContextWeave v2.0 — Behavioral Intelligence Platform"}


@app.post("/signup")
def signup(data: SignupInput, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Account created successfully. Please log in."}


@app.post("/login")
def login(data: LoginInput, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user.id)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/login-page", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup-page", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


# ══════════════════════════════════════════════════════════════════════════════
# Protected routes
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/add-note")
def add_note(
    note: NoteInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db.add(Note(
        user_id=current_user.id,
        text=note.text,
        created_at=datetime.utcnow(),
    ))
    db.commit()
    return {"status": "Note saved"}


@app.post("/simulate")
def simulate(
    data: SimulateInput,
    current_user: User = Depends(get_current_user),
):
    """
    Behavior Simulation Engine.
    Send any free-text scenario and get:
      predicted_score, risk, recommendation, signals_detected
    """
    if not data.scenario or not data.scenario.strip():
        raise HTTPException(status_code=400, detail="Scenario text is required")
    return simulate_scenario(data.scenario)


@app.get("/report")
def get_report(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # ── Load this user's notes from DB ────────────────────────────────────────
    db_notes = (
        db.query(Note)
        .filter(Note.user_id == current_user.id)
        .order_by(Note.created_at)
        .all()
    )

    # Format as [YYYY-MM-DD] text  (same format main.py functions expect)
    notes = [
        f"[{n.created_at.strftime('%Y-%m-%d')}] {n.text}"
        for n in db_notes
    ]

    # ── Run existing AI pipeline (no changes) ─────────────────────────────────
    today_notes    = clean_notes(get_today_notes(notes))
    ai             = run_behavioral_ai(today_notes)
    all_records    = build_note_records(notes)       # full history for pattern/insight engines

    stall          = detect_stall(notes)
    semantic_stall = detect_semantic_stall(notes)
    summary        = summarize_notes(today_notes)
    priority       = generate_priority(summary)
    patterns_text  = detect_patterns(today_notes)
    recent         = recent_focus(today_notes)
    pressure       = predict_pressure(today_notes)
    intent         = detect_intent(today_notes)
    behavior_profile = get_behavior_profile()
    suggestion     = generate_suggestion(summary, priority, patterns_text, behavior_profile)
    weekly         = generate_weekly_insight()

    # ── New v2 engines ────────────────────────────────────────────────────────
    behavior_pats = discover_behavior_patterns(all_records)
    ai_insights   = generate_insights(all_records)

    # Save behavioral state for history chart (per-user file)
    _save_user_behavior_state(current_user.id, ai)

    # ══════════════════════════════════════════════════════════════════════════
    # Build clean structured report
    # ══════════════════════════════════════════════════════════════════════════
    r  = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    r += "       CONTEXTWEAVE — BEHAVIORAL INTELLIGENCE\n"
    r += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    # Core metrics block
    r += f"🎯 Cognitive Score       : {ai['score']}\n"
    r += f"📈 Behavior Direction    : {ai['trajectory'].capitalize()}\n"
    r += f"⚠  Dominant Risk         : {ai['risks'].capitalize()}\n"
    r += f"🧭 Strategy Mode         : {ai['strategy']}\n"
    r += f"🔮 Predicted Next Score  : {ai.get('prediction', 'Not enough data yet')}\n"
    r += f"🎯 Focus State           : {str(ai.get('focus_state', '—')).capitalize()}\n"
    r += f"💡 Focus Advice          : {ai.get('focus_advice', '—')}\n"
    r += f"💬 AI Guidance           : {ai.get('advice', '—')}\n"

    # Weekly summary
    r += f"\n{weekly}\n"

    # Context detail
    r += f"\n📋 Context Summary\n{'─'*40}\n"
    r += summary   + "\n"
    r += priority  + "\n"
    r += recent    + "\n"
    r += pressure  + "\n"
    r += intent    + "\n"
    r += stall     + "\n"
    r += semantic_stall + "\n"

    # Behavioral patterns (numbered list, parseable by frontend)
    r += f"\n🔁 Behavioral Patterns Detected\n{'─'*40}\n"
    if behavior_pats:
        for i, p in enumerate(behavior_pats, 1):
            r += f"  {i}. {p}\n"
    else:
        r += "  Keep logging — patterns will surface with more data.\n"

    # AI insights (numbered list, parseable by frontend)
    r += f"\n🧠 AI Insights\n{'─'*40}\n"
    if ai_insights:
        for i, ins in enumerate(ai_insights, 1):
            r += f"  {i}. {ins}\n"
    else:
        r += "  Insights will appear after more behavioral data is logged.\n"

    # Anomaly
    r += f"\n🚨 Anomaly Check         : {ai.get('anomaly', 'No anomaly analysis yet')}\n"

    # Recommendations
    r += f"\n✅ Recommendations\n{'─'*40}\n"
    r += f"  {suggestion}\n"

    # Optional forecast (if behavior_forecaster.py is present)
    forecast = ai.get("forecast")
    if forecast:
        try:
            from behavior_forecaster import format_forecast_for_report
            r += "\n" + format_forecast_for_report(forecast)
        except Exception:
            pass

    r += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    return {"report": r}


@app.get("/behavior-data")
def behavior_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return per-user behavioral history for the chart."""
    history_file = f"behavior_history_{current_user.id}.json"
    if not os.path.exists(history_file):
        return {"dates": [], "scores": []}

    with open(history_file, "r") as f:
        history = json.load(f)

    return {
        "dates":  [h["date"]  for h in history],
        "scores": [h["score"] for h in history],
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_user_behavior_state(user_id: int, ai_output: dict):
    """Save today's AI output to a per-user JSON history file."""
    file_path = f"behavior_history_{user_id}.json"
    state = {
        "date":       datetime.now().strftime("%Y-%m-%d"),
        "score":      ai_output.get("score", 0),
        "trajectory": ai_output.get("trajectory", "stable"),
        "strategy":   ai_output.get("strategy", "observe"),
        "risk":       ai_output.get("risks", "none"),
    }
    history = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    # Avoid duplicate entries for the same date
    today = state["date"]
    history = [h for h in history if h.get("date") != today]
    history.append(state)

    with open(file_path, "w") as f:
        json.dump(history, f, indent=2)