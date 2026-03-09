"""
ContextWeave — auth.py
JWT creation/verification + bcrypt password helpers.
Uses bcrypt directly (no passlib) to avoid version compatibility errors.
"""

import os
import bcrypt
from datetime import datetime, timedelta

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database import get_db, User

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY         = os.getenv("CW_SECRET_KEY", "contextweave-secret-change-in-production")
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 72

# ── Password hashing (bcrypt direct — no passlib) ─────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

# ── JWT ───────────────────────────────────────────────────────────────────────
def create_token(user_id: int) -> str:
    expire  = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> int:
    """Returns user_id (int) or raises HTTPException 401."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise ValueError
        return int(user_id)
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ── FastAPI dependency ────────────────────────────────────────────────────────
bearer_scheme = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency that validates the Bearer token and returns the User row.
    Inject with:  current_user: User = Depends(get_current_user)
    """
    user_id = decode_token(credentials.credentials)
    user    = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user