from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def _record(row: Any) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def list_members(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(text("""
        SELECT member_id, age, gender, race, ethnicity, city, state
        FROM member_model_features
        ORDER BY member_id
    """)).mappings()
    return [dict(row) for row in rows]


def get_member(db: Session, member_id: str) -> dict[str, Any] | None:
    return _record(db.execute(text("""
        SELECT member_id, age, gender, race, ethnicity, city, state
        FROM member_model_features
        WHERE member_id = :member_id
    """), {"member_id": member_id}).mappings().first())


def get_sdoh(db: Session, member_id: str) -> dict[str, Any] | None:
    return _record(db.execute(text("""
        SELECT * FROM member_sdoh_features WHERE member_id = :member_id
    """), {"member_id": member_id}).mappings().first())


def get_clinical(db: Session, member_id: str) -> dict[str, Any] | None:
    return _record(db.execute(text("""
        SELECT * FROM member_clinical_features WHERE member_id = :member_id
    """), {"member_id": member_id}).mappings().first())


def get_location(db: Session, member_id: str) -> dict[str, Any] | None:
    return _record(db.execute(text("""
        SELECT member_id, lat, lon, county, county_fips, city, state
        FROM member_model_features WHERE member_id = :member_id
    """), {"member_id": member_id}).mappings().first())
