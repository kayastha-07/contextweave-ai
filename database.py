"""
ContextWeave — database.py  v3
Added: is_public, is_verified, otp fields
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./contextweave.db"
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    is_public     = Column(Boolean, default=False, nullable=False)
    is_verified   = Column(Boolean, default=False, nullable=False)  # email verification
    notes         = relationship("Note", back_populates="owner", cascade="all, delete-orphan")


class Note(Base):
    __tablename__ = "notes"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    text       = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner      = relationship("User", back_populates="notes")


def init_db():
    Base.metadata.create_all(bind=engine)
    # Safe migration — add new columns if missing
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    existing  = [c["name"] for c in inspector.get_columns("users")]
    with engine.connect() as conn:
        if "is_public" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN is_public BOOLEAN DEFAULT 0"))
        if "is_verified" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT 0"))
        conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()