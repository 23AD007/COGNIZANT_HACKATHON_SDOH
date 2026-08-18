from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.member import MemberResponse


from backend.schemas.risk import RiskResponse
from backend.services.risk_service import (
    predict_member_risk,
    get_risk_band
)

router = APIRouter(
    prefix="/members",
    tags=["Members"]
)



@router.get("", response_model=list[MemberResponse])
def get_members(db: Session = Depends(get_db)):

    query = text("""
        SELECT
            member_id,
            age,
            gender,
            race,
            city,
            state,
            lat,
            lon
        FROM sdoh.member_sdoh_features
        LIMIT 100
    """)

    result = db.execute(query)

    return [
        {
            "member_id": row.member_id,
            "age": row.age,
            "gender": row.gender,
            "race": row.race,
            "city": row.city,
            "state": row.state,
            "lat": row.lat,
            "lon": row.lon,
        }
        for row in result
    ]


@router.get("/{member_id}/sdoh")
def get_member_sdoh(
    member_id: str,
    db: Session = Depends(get_db)
):

    query = text("""
        SELECT *
        FROM sdoh.member_sdoh_features
        WHERE member_id = :member_id
    """)

    result = db.execute(
        query,
        {"member_id": member_id}
    ).mappings().first()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Member not found"
        )

    return dict(result)


@router.get("/{member_id}/clinical")
def get_member_clinical(
    member_id: str,
    db: Session = Depends(get_db)
):

    query = text("""
        SELECT
            patient_id,
            encounter_count,
            condition_count,
            medication_count,
            procedure_count
        FROM sdoh.clinical_patient_coverage
        WHERE patient_id = :member_id
    """)

    result = db.execute(
        query,
        {"member_id": member_id}
    ).mappings().first()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Clinical information not found"
        )

    return dict(result)


@router.get("/{member_id}", response_model=MemberResponse)
def get_member(
    member_id: str,
    db: Session = Depends(get_db)
):

    query = text("""
        SELECT
            member_id,
            age,
            gender,
            race,
            city,
            state,
            lat,
            lon
        FROM sdoh.member_sdoh_features
        WHERE member_id = :member_id
    """)

    result = db.execute(
        query,
        {"member_id": member_id}
    ).mappings().first()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Member not found"
        )

    return result