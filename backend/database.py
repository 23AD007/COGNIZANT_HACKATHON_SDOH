"""Database configuration following the Amulya SQLAlchemy pattern."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def database_url() -> str:
    direct_url = os.getenv("DATABASE_URL")
    if direct_url:
        return direct_url
    required = {name: os.getenv(name) for name in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")}
    if not all(required.values()):
        environment = os.getenv("APP_ENV", os.getenv("FLASK_ENV", "")).lower()
        testing = os.getenv("TESTING", "").lower() in {"1", "true", "yes"}
        if environment in {"development", "testing"} or testing:
            local_path = PROJECT_ROOT / "data" / "processed" / "healthlens_dev.sqlite"
            return f"sqlite:///{local_path.as_posix()}"
        raise ValueError("Database configuration is incomplete. Set DATABASE_URL or DB_HOST, DB_NAME, DB_USER and DB_PASSWORD. SQLite is available only with FLASK_ENV=development/testing or TESTING=true.")
    return str(URL.create("postgresql+psycopg2", username=required["DB_USER"], password=required["DB_PASSWORD"], host=required["DB_HOST"], port=int(os.getenv("DB_PORT", "5432")), database=required["DB_NAME"]))


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=1)
def get_engine(url: str | None = None):
    resolved_url = url or database_url()
    options = {"pool_pre_ping": True}
    if resolved_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(resolved_url, **options)


def session_factory(url: str | None = None):
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine(url))


def get_db():
    """Yield a request-scoped SQLAlchemy session for FastAPI dependencies."""
    session: Session = session_factory()()
    try:
        yield session
    finally:
        session.close()
