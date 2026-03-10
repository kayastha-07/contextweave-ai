"""
ContextWeave — rate_limiter.py
Simple in-memory rate limiter (no Redis needed).
Works on Railway without any extra dependencies.
"""
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import HTTPException, Request


class RateLimiter:
    """
    Sliding window rate limiter.
    Usage: limiter = RateLimiter(max_calls=5, window_seconds=60)
    """
    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls      = max_calls
        self.window_seconds = window_seconds
        self._store: dict   = defaultdict(list)

    def _get_key(self, request: Request) -> str:
        # Use IP + path as key
        ip = request.client.host if request.client else "unknown"
        return f"{ip}:{request.url.path}"

    def check(self, request: Request, key: str = None):
        """Raise 429 if rate limit exceeded."""
        k      = key or self._get_key(request)
        now    = datetime.utcnow()
        window = now - timedelta(seconds=self.window_seconds)

        # Clean old entries
        self._store[k] = [t for t in self._store[k] if t > window]

        if len(self._store[k]) >= self.max_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests. Try again in {self.window_seconds} seconds."
            )
        self._store[k].append(now)


# ── Pre-configured limiters ───────────────────────────────────────────────────
login_limiter    = RateLimiter(max_calls=5,  window_seconds=60)   # 5 login attempts/min
signup_limiter   = RateLimiter(max_calls=3,  window_seconds=60)   # 3 signups/min
otp_limiter      = RateLimiter(max_calls=3,  window_seconds=300)  # 3 OTP requests/5min
note_limiter     = RateLimiter(max_calls=30, window_seconds=60)   # 30 notes/min
report_limiter   = RateLimiter(max_calls=10, window_seconds=60)   # 10 reports/min