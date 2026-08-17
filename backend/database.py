"""Database configuration following the Amulya SQLAlchemy pattern."""
from __future__ import annotations

import os
from pathlib import Path
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def database_url() -> str:
    direct_url = os.getenv("DATABASE_URL")
    if direct_url:
        return direct_url
    required = {name: os.getenv(name) for name in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")}
    if not all(required.values()):
        environment = os.getenv("FLASK_ENV", "").lower()
        testing = os.getenv("TESTING", "").lower() in {"1", "true", "yes"}
        if environment in {"development", "testing"} or testing:
            local_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "healthlens_dev.sqlite"
            return f"sqlite:///{local_path}"
        raise ValueError("Database configuration is incomplete. Set DATABASE_URL or DB_HOST, DB_NAME, DB_USER and DB_PASSWORD. SQLite is available only with FLASK_ENV=development/testing or TESTING=true.")
    return str(URL.create("postgresql+psycopg2", username=required["DB_USER"], password=required["DB_PASSWORD"], host=required["DB_HOST"], port=int(os.getenv("DB_PORT", "5432")), database=required["DB_NAME"]))


class Base(DeclarativeBase):
    pass


def get_engine(url: str | None = None):
    return create_engine(url or database_url(), pool_pre_ping=True)


def session_factory(url: str | None = None):
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine(url))
