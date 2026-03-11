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

# ══════════════════════════════════════════════════════════════════════════════
# Public Profile routes
# ══════════════════════════════════════════════════════════════════════════════

from profile_engine import build_behavioral_dna

@app.get("/profile/{user_id}", response_class=HTMLResponse)
def public_profile_page(user_id: int, request: Request):
    """Serve the public profile HTML page."""
    return templates.TemplateResponse("profile.html", {"request": request})


@app.get("/api/profile/{user_id}")
def get_profile_data(user_id: int, db: Session = Depends(get_db)):
    """Return public profile JSON data."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.is_public:
        return {"is_public": False}

    # Load notes
    db_notes = (
        db.query(Note)
        .filter(Note.user_id == user_id)
        .order_by(Note.created_at)
        .all()
    )
    notes = [f"[{n.created_at.strftime('%Y-%m-%d')}] {n.text}" for n in db_notes]

    dna = build_behavioral_dna(user_id, notes)

    return {
        "is_public":      True,
        "email":          user.email,
        "behavioral_dna": dna,
    }


@app.get("/me")
def get_me(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return current user's id and public status."""
    return {
        "id":        current_user.id,
        "email":     current_user.email,
        "is_public": current_user.is_public,
    }


@app.post("/toggle-public")
def toggle_public(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Toggle the user's profile between public and private."""
    current_user.is_public = not current_user.is_public
    db.commit()
    return {
        "is_public":   current_user.is_public,
        "profile_url": f"/profile/{current_user.id}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# PDF Export
# ══════════════════════════════════════════════════════════════════════════════

from fastapi.responses import Response as FastAPIResponse
from pdf_engine import generate_report_pdf
from profile_engine import build_behavioral_dna

@app.get("/export-pdf")
def export_pdf(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate and return a behavioral report PDF for download."""

    # Load notes
    db_notes = (
        db.query(Note)
        .filter(Note.user_id == current_user.id)
        .order_by(Note.created_at)
        .all()
    )
    notes = [f"[{n.created_at.strftime('%Y-%m-%d')}] {n.text}" for n in db_notes]

    # Run report pipeline (reuse existing logic)
    from main import (
        clean_notes, get_today_notes, run_behavioral_ai,
        build_note_records, detect_stall, detect_semantic_stall,
        summarize_notes, generate_priority, detect_patterns,
        recent_focus, predict_pressure, detect_intent,
        get_behavior_profile, generate_suggestion,
        generate_weekly_insight, discover_behavior_patterns,
    )
    from insight_engine import generate_insights
    from behavior_pattern_engine import discover_behavior_patterns as dbp

    today_notes   = clean_notes(get_today_notes(notes))
    ai            = run_behavioral_ai(today_notes)
    all_records   = build_note_records(notes)
    summary       = summarize_notes(today_notes)
    priority      = generate_priority(summary)
    patterns_text = detect_patterns(today_notes)
    recent        = recent_focus(today_notes)
    pressure      = predict_pressure(today_notes)
    intent        = detect_intent(today_notes)
    stall         = detect_stall(notes)
    sem_stall     = detect_semantic_stall(notes)
    profile       = get_behavior_profile()
    suggestion    = generate_suggestion(summary, priority, patterns_text, profile)
    weekly        = generate_weekly_insight()
    behavior_pats = dbp(all_records)
    ai_insights   = generate_insights(all_records)

    # Build report text
    r  = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    r += "       CONTEXTWEAVE — BEHAVIORAL INTELLIGENCE\n"
    r += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    r += f"🎯 Cognitive Score       : {ai['score']}\n"
    r += f"📈 Behavior Direction    : {ai['trajectory'].capitalize()}\n"
    r += f"⚠  Dominant Risk         : {ai['risks'].capitalize()}\n"
    r += f"🧭 Strategy Mode         : {ai['strategy']}\n"
    r += f"🎯 Focus State           : {str(ai.get('focus_state','—')).capitalize()}\n"
    r += f"💡 Focus Advice          : {ai.get('focus_advice','—')}\n"
    r += f"💬 AI Guidance           : {ai.get('advice','—')}\n"
    r += f"\n{weekly}\n"
    r += f"\n📋 Context Summary\n{'─'*40}\n{summary}\n{priority}\n{recent}\n{pressure}\n{intent}\n{stall}\n{sem_stall}\n"
    r += f"\n🔁 Behavioral Patterns Detected\n{'─'*40}\n"
    for i, p in enumerate(behavior_pats, 1): r += f"  {i}. {p}\n"
    r += f"\n🧠 AI Insights\n{'─'*40}\n"
    for i, ins in enumerate(ai_insights, 1): r += f"  {i}. {ins}\n"
    r += f"\n✅ Recommendations\n{'─'*40}\n  {suggestion}\n"
    r += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    # Build DNA
    dna = build_behavioral_dna(current_user.id, notes)

    # Generate PDF
    pdf_bytes = generate_report_pdf(current_user.email, r, dna, len(db_notes))

    filename = f"contextweave_report_{datetime.now().strftime('%Y%m%d')}.pdf"
    return FastAPIResponse(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Rate Limiting + Email OTP + Password Reset + Admin + Forecasting
# ══════════════════════════════════════════════════════════════════════════════

import hashlib, secrets
from rate_limiter import login_limiter, signup_limiter, otp_limiter, note_limiter, report_limiter
from email_service import send_otp, verify_otp
from forecasting_engine import forecast_scores
from database import User, Note  # already imported above, fine

# Admin password read at request time - not here at import time


# ── Schemas ───────────────────────────────────────────────────────────────────
class OTPRequest(BaseModel):
    email: str

class OTPVerify(BaseModel):
    email: str
    otp:   str

class ResetPassword(BaseModel):
    email:        str
    otp:          str
    new_password: str

class AdminLogin(BaseModel):
    password: str


# ── Email Verification ────────────────────────────────────────────────────────

@app.post("/send-verification")
def send_verification(data: OTPRequest, request: Request, db: Session = Depends(get_db)):
    otp_limiter.check(request)
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")
    if user.is_verified:
        return {"message": "Email already verified"}
    ok = send_otp(data.email, purpose="verify")
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to send OTP. Check GMAIL_USER and GMAIL_APP_PASS env vars.")
    return {"message": "OTP sent to your email"}


@app.post("/verify-email")
def verify_email(data: OTPVerify, db: Session = Depends(get_db)):
    if not verify_otp(data.email, data.otp):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_verified = True
    db.commit()
    return {"message": "Email verified successfully"}


# ── Password Reset ────────────────────────────────────────────────────────────

@app.post("/forgot-password")
def forgot_password(data: OTPRequest, request: Request, db: Session = Depends(get_db)):
    otp_limiter.check(request)
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        # Don't reveal if email exists
        return {"message": "If this email is registered, an OTP has been sent"}
    ok = send_otp(data.email, purpose="reset")
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to send OTP")
    return {"message": "If this email is registered, an OTP has been sent"}


@app.post("/reset-password")
def reset_password(data: ResetPassword, db: Session = Depends(get_db)):
    if not verify_otp(data.email, data.otp):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    from auth import hash_password
    user.password_hash = hash_password(data.new_password)
    db.commit()
    return {"message": "Password reset successfully. Please log in."}


# ── Rate-limited login/signup overrides ───────────────────────────────────────

@app.post("/login/safe")
def login_safe(data: LoginInput, request: Request, db: Session = Depends(get_db)):
    login_limiter.check(request)
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user.id)
    return {"access_token": token, "token_type": "bearer", "is_verified": user.is_verified}


# ── Behavioral Forecasting ────────────────────────────────────────────────────

@app.get("/forecast")
def get_forecast(
    current_user: User = Depends(get_current_user),
):
    return forecast_scores(current_user.id, days_ahead=3)


# ── Admin routes ──────────────────────────────────────────────────────────────

def _check_admin(request: Request):
    admin_pwd = os.getenv("ADMIN_PASSWORD", "contextweave-admin-2026")
    admin_tok = hashlib.sha256(admin_pwd.encode()).hexdigest()
    token = request.headers.get("X-Admin-Token", "")
    if token != admin_tok:
        raise HTTPException(status_code=403, detail="Forbidden")

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

@app.post("/admin/login")
def admin_login(data: AdminLogin):
    admin_pwd = os.getenv("ADMIN_PASSWORD", "contextweave-admin-2026")
    admin_tok = hashlib.sha256(admin_pwd.encode()).hexdigest()
    if data.password != admin_pwd:
        raise HTTPException(status_code=403, detail="Wrong password")
    return {"token": admin_tok}

@app.get("/admin/stats")
def admin_stats(request: Request, db: Session = Depends(get_db)):
    _check_admin(request)
    from sqlalchemy import func
    from datetime import date
    total_users  = db.query(func.count(User.id)).scalar()
    total_notes  = db.query(func.count(Note.id)).scalar()
    today_str    = date.today().isoformat()
    notes_today  = db.query(func.count(Note.id)).filter(
        func.date(Note.created_at) == today_str
    ).scalar()

    # avg score across all users from their history files
    scores = []
    for uid in [u.id for u in db.query(User.id).all()]:
        hf = f"behavior_history_{uid}.json"
        if os.path.exists(hf):
            try:
                with open(hf) as f:
                    h = json.load(f)
                scores += [e["score"] for e in h if "score" in e]
            except Exception:
                pass
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    return {
        "total_users": total_users,
        "total_notes": total_notes,
        "notes_today": notes_today,
        "avg_score":   avg_score,
    }

@app.get("/admin/users")
def admin_users(request: Request, db: Session = Depends(get_db)):
    _check_admin(request)
    users = db.query(User).all()
    result = []
    for u in users:
        notes_count = db.query(Note).filter(Note.user_id == u.id).count()
        avg_score   = None
        hf = f"behavior_history_{u.id}.json"
        if os.path.exists(hf):
            try:
                with open(hf) as f:
                    h = json.load(f)
                sc = [e["score"] for e in h if "score" in e]
                if sc: avg_score = round(sum(sc)/len(sc), 1)
            except Exception:
                pass
        result.append({
            "id":          u.id,
            "email":       u.email,
            "notes_count": notes_count,
            "avg_score":   avg_score,
            "is_verified": getattr(u, "is_verified", False),
            "is_public":   getattr(u, "is_public", False),
            "joined":      "—",
        })
    return result


# ── Debug email (REMOVE AFTER TESTING) ───────────────────────────────────────
@app.get("/debug-email")
def debug_email(request: Request):
    import smtplib
    gmail_user     = os.getenv("GMAIL_USER", "NOT_SET")
    gmail_app_pass = os.getenv("GMAIL_APP_PASS", "NOT_SET")
    admin_pwd      = os.getenv("ADMIN_PASSWORD", "NOT_SET")

    result = {
        "GMAIL_USER":     gmail_user,
        "GMAIL_APP_PASS": "SET" if gmail_app_pass != "NOT_SET" else "NOT_SET",
        "GMAIL_APP_PASS_LENGTH": len(gmail_app_pass),
        "ADMIN_PASSWORD": "SET" if admin_pwd != "NOT_SET" else "NOT_SET",
    }

    # Try actual SMTP connection
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(gmail_user, gmail_app_pass)
            result["smtp_connection"] = "SUCCESS"
    except smtplib.SMTPAuthenticationError as e:
        result["smtp_connection"] = "FAILED - AUTH ERROR"
        result["smtp_error"] = str(e)
    except Exception as e:
        result["smtp_connection"] = "FAILED - OTHER"
        result["smtp_error"] = str(e)

    return result

@app.get("/admin/notes")
def admin_notes(request: Request, db: Session = Depends(get_db)):
    _check_admin(request)
    notes = (
        db.query(Note, User.email)
        .join(User, Note.user_id == User.id)
        .order_by(Note.created_at.desc())
        .limit(20)
        .all()
    )
    return [
        {
            "email": email,
            "text":  note.text[:120] + ("…" if len(note.text) > 120 else ""),
            "date":  note.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for note, email in notes
    ]