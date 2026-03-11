"""
ContextWeave — email_service.py
Uses official resend Python SDK.
"""
import os
import random
import string
from datetime import datetime, timedelta
from typing import Dict, Tuple

_otp_store: Dict[str, Tuple[str, datetime]] = {}
OTP_EXPIRY_MIN = 5


def _generate_otp(length=6) -> str:
    return "".join(random.choices(string.digits, k=length))


def send_otp(to_email: str, purpose: str = "verify") -> bool:
    try:
        import resend
    except ImportError:
        print("[email_service] resend package not installed")
        return False

    resend.api_key = os.getenv("RESEND_API_KEY", "")
    if not resend.api_key:
        print("[email_service] RESEND_API_KEY not set")
        return False

    otp     = _generate_otp()
    expires = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MIN)
    _otp_store[to_email] = (otp, expires)

    subject_map = {
        "verify": "ContextWeave — Verify your email",
        "reset":  "ContextWeave — Password reset OTP",
    }

    try:
        params = {
            "from":    "ContextWeave <onboarding@resend.dev>",
            "to":      [to_email],
            "subject": subject_map.get(purpose, "ContextWeave OTP"),
            "html":    _build_email_html(otp, purpose),
        }
        resend.Emails.send(params)
        print(f"[email_service] OTP sent to {to_email}")
        return True
    except Exception as e:
        print(f"[email_service] Resend error: {e}")
        return False


def verify_otp(email: str, otp: str) -> bool:
    entry = _otp_store.get(email)
    if not entry:
        return False
    stored_otp, expires_at = entry
    if datetime.utcnow() > expires_at:
        del _otp_store[email]
        return False
    if stored_otp != otp.strip():
        return False
    del _otp_store[email]
    return True


def _build_email_html(otp: str, purpose: str) -> str:
    action = "verify your email" if purpose == "verify" else "reset your password"
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: 'Helvetica Neue', sans-serif; background:#f4f7fb; margin:0; padding:0; }}
  .wrap {{ max-width:480px; margin:40px auto; background:#fff;
           border-radius:16px; overflow:hidden;
           box-shadow:0 4px 24px rgba(0,0,0,0.08); }}
  .header {{ background:linear-gradient(135deg,#050a12,#0d1a2e); padding:32px; text-align:center; }}
  .brand {{ color:#3b9eff; font-size:20px; font-weight:700; letter-spacing:1px; font-family:monospace; }}
  .body {{ padding:36px 32px; }}
  .greeting {{ font-size:16px; color:#1a1a2e; margin-bottom:12px; font-weight:600; }}
  .desc {{ font-size:14px; color:#5a6a80; margin-bottom:28px; line-height:1.6; }}
  .otp-box {{ background:#f0f4ff; border:2px dashed #3b9eff;
              border-radius:12px; padding:24px; text-align:center; margin-bottom:28px; }}
  .otp {{ font-size:44px; font-weight:700; color:#1a1a2e;
          letter-spacing:12px; font-family:monospace; }}
  .expiry {{ font-size:12px; color:#8a9ab0; margin-top:8px; }}
  .warning {{ font-size:12px; color:#8a9ab0; line-height:1.6; }}
  .footer {{ background:#f8fafc; padding:20px 32px; text-align:center;
             font-size:11px; color:#aab; border-top:1px solid #eee; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="header"><div class="brand">🧠 ContextWeave</div></div>
  <div class="body">
    <div class="greeting">Your one-time password</div>
    <div class="desc">
      Use the OTP below to {action}.
      This code expires in <strong>{OTP_EXPIRY_MIN} minutes</strong>.
    </div>
    <div class="otp-box">
      <div class="otp">{otp}</div>
      <div class="expiry">Valid for {OTP_EXPIRY_MIN} minutes only</div>
    </div>
    <div class="warning">
      If you didn't request this, you can safely ignore this email.<br>
      Never share this OTP with anyone.
    </div>
  </div>
  <div class="footer">
    ContextWeave Behavioral Intelligence Platform<br>
    Built by Abhishek Shrivastav · SDGI Global University
  </div>
</div>
</body>
</html>
"""