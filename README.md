# 🧠 ContextWeave — Behavioral Intelligence Platform

> **Transform your daily notes into deep behavioral insights, cognitive scores, and 3-day performance forecasts.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-contextweave--production.up.railway.app-3b9eff?style=for-the-badge)](https://contextweave-production.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-8B5CF6?style=for-the-badge)](https://railway.app)


---

## 🚀 What is ContextWeave?

ContextWeave is a full-stack behavioral intelligence SaaS that analyzes your daily notes to reveal hidden cognitive patterns, predict your mental performance, and generate a shareable Behavioral DNA profile.

Write a note. Get intelligence.

---

## ✨ Features

### 🧠 Core Intelligence
- **Cognitive Scoring** — Every note is analyzed in real-time to produce a behavioral score from 0–100
- **47+ Pattern Detection** — Identifies procrastination cycles, stress triggers, focus states, collaboration signals, and more
- **3-Day Behavioral Forecasting** — Linear regression on behavioral history predicts your cognitive score 3 days ahead
- **Anomaly Detection** — Flags behavioral stalls, burnout signals, and semantic shifts automatically

### 👤 Behavioral DNA
- **Archetype System** — Computes your unique behavioral archetype (e.g. "The Momentum Builder", "The Pressure Converter")
- **Public Profile** — Shareable profile page showing your DNA, archetype, streak, and dominant patterns
- **PDF Export** — Download your full behavioral report as a styled PDF

### 🔐 Auth & Security
- **JWT Authentication** — Secure token-based auth with bcrypt password hashing
- **Email OTP Verification** — Resend API integration for email verification and password reset
- **Rate Limiting** — Custom in-memory sliding window rate limiter protecting all endpoints
- **Password Reset** — Full OTP-based forgot password flow

### 📊 Admin & Analytics
- **Admin Dashboard** — Password-protected dashboard showing total users, notes, daily activity, and per-user stats
- **Behavioral History** — Per-user JSON-based history tracking with trend analysis

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI (Python 3.11) |
| **Database** | SQLite + SQLAlchemy ORM |
| **Authentication** | JWT (python-jose) + bcrypt |
| **NLP Engine** | Custom pure-Python behavioral signal extraction (47+ patterns) |
| **Forecasting** | NumPy linear regression with weighted sliding window |
| **Email** | Resend API |
| **PDF Generation** | ReportLab |
| **Frontend** | Jinja2 templates + vanilla JS |
| **Deployment** | Railway (auto-deploy from GitHub) |

---

## 🧬 How It Works

```
User writes a note
        ↓
Pure-Python NLP extracts behavioral signals
(focus, stress, procrastination, collaboration, momentum...)
        ↓
Behavioral Engine computes cognitive score (0–100)
        ↓
Pattern Engine detects 47+ behavioral patterns
        ↓
Forecasting Engine predicts next 3 days via linear regression
        ↓
DNA Engine computes archetype + behavioral profile
        ↓
User gets insights, recommendations, and forecast
```

---

## 📁 Project Structure

```
contextweave/
├── app.py                      # FastAPI routes (27 endpoints)
├── main.py                     # Core NLP + behavioral analysis engine
├── behavioral_engine.py        # Behavioral signal processing
├── prediction_engine.py        # Score prediction logic
├── anomaly_engine.py           # Anomaly + stall detection
├── habit_engine.py             # Habit pattern recognition
├── profile_engine.py           # Behavioral DNA computation
├── forecasting_engine.py       # 3-day score forecasting (NumPy)
├── insight_engine.py           # AI insight generation
├── behavior_pattern_engine.py  # Pattern discovery
├── pdf_engine.py               # PDF report generation (ReportLab)
├── email_service.py            # OTP email via Resend API
├── rate_limiter.py             # Custom sliding window rate limiter
├── auth.py                     # JWT + bcrypt authentication
├── database.py                 # SQLAlchemy models + migrations
├── templates/
│   ├── landing.html            # Marketing landing page
│   ├── index.html              # Main dashboard
│   ├── login.html              # Login + forgot password
│   ├── signup.html             # Registration
│   ├── profile.html            # Public behavioral profile
│   └── admin_dashboard.html   # Admin panel
├── Procfile                    # Railway deployment config
└── requirements.txt
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page |
| `POST` | `/signup` | Register new user |
| `POST` | `/login` | Authenticate + get JWT |
| `POST` | `/add-note` | Add behavioral note |
| `GET` | `/report` | Get full behavioral report |
| `GET` | `/behavior-data` | Get score history |
| `GET` | `/forecast` | Get 3-day score forecast |
| `GET` | `/export-pdf` | Download behavioral report PDF |
| `GET` | `/profile/{user_id}` | Public behavioral profile |
| `POST` | `/toggle-public` | Toggle profile visibility |
| `POST` | `/forgot-password` | Request password reset OTP |
| `POST` | `/reset-password` | Reset password with OTP |
| `POST` | `/send-verification` | Send email verification OTP |
| `POST` | `/verify-email` | Verify email with OTP |
| `GET` | `/admin` | Admin dashboard |
| `GET` | `/admin/stats` | Platform-wide statistics |
| `GET` | `/admin/users` | All users with metrics |
| `GET` | `/admin/notes` | Recent notes feed |

---

## ⚙️ Local Setup

```bash
# Clone
git clone https://github.com/kayastha-07/contextweave-ai
cd contextweave-ai

# Install dependencies
pip install -r requirements.txt

# Create .env
RESEND_API_KEY=re_your_key
SECRET_KEY=your_secret_key
ADMIN_PASSWORD=your_admin_password
USE_GEMINI=false

# Run
uvicorn app:app --reload --port 8080
```

Visit `http://localhost:8080`

---

## 🚢 Deployment

Deployed on **Railway** with automatic deploys from the `main` branch.

**Environment Variables required:**
```
SECRET_KEY         JWT signing secret
RESEND_API_KEY     Resend API key for emails
ADMIN_PASSWORD     Admin dashboard password
USE_GEMINI         false (or true with GEMINI_API_KEY)
```

---

## 🧪 Behavioral Archetypes

ContextWeave computes one of 7 archetypes based on your behavioral history:

| Archetype | Trigger Condition |
|-----------|------------------|
| 🚀 The Momentum Builder | High score + improving trend + executor pattern |
| ⚡ The Pressure Converter | High stress signals + score ≥ 55 |
| 🔄 The Recovering Strategist | Procrastinator pattern + improving trend |
| 🌐 The Network Thinker | Dominant collaborator signals |
| ⚙️ The Consistent Operator | Active streak ≥ 5 days |
| 🔨 The Rebuilder | Score < 40 |
| ⚖️ The Balanced Thinker | Default balanced profile |

---

## 📄 License

MIT License — feel free to fork and build on this.

---

## 👨‍💻 Author

**Anchal Kumar Shrivastav**
B.Tech CSE (AI & ML) · SDGI Global University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/anchal-shrivastav-647141276?utm_source=share_via&utm_content=profile&utm_medium=member_android)
[![GitHub](https://img.shields.io/badge/GitHub-kayastha--07-181717?style=flat&logo=github)](https://github.com/kayastha-07)

---

*Built with FastAPI, SQLAlchemy, NumPy, ReportLab, and a lot of behavioral science.*