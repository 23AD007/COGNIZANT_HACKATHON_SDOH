from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.member import MemberResponse
from fastapi import APIRouter, Depends, HTTPException, Query

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

@router.get("/{member_id}/location")
def get_member_location(
    member_id: str,
    db: Session = Depends(get_db)
):

    query = text("""
        SELECT
            m.member_id,
            m.city,
            m.state,

            c.geoid,
            c.name AS county_name,

            s.unemployment_pct,
            s.uninsured_pct,
            s.public_assistance_pct,

            ST_X(m.geom) AS longitude,
            ST_Y(m.geom) AS latitude

        FROM sdoh.member_sdoh_features AS m

        JOIN sdoh.county_boundaries AS c
            ON ST_Within(m.geom, c.geom)

        LEFT JOIN sdoh.county_sdoh_features AS s
            ON LPAD(s.county_fips::text, 5, '0') = c.geoid

        WHERE m.member_id = :member_id
          AND m.geom IS NOT NULL
    """)

    result = db.execute(
        query,
        {"member_id": member_id}
    ).mappings().first()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Member location or county information not found"
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

@router.get("/{member_id}/nearby")
def get_nearby_members(
    member_id: str,
    radius_km: float = Query(default=10, gt=0, le=100),
    db: Session = Depends(get_db)
):

    query = text("""
        SELECT
            m2.member_id,
            m2.age,
            m2.gender,
            m2.city,
            m2.state,
            ST_Distance(
                m1.geom::geography,
                m2.geom::geography
            ) / 1000 AS distance_km
        FROM sdoh.member_sdoh_features m1
        JOIN sdoh.member_sdoh_features m2
            ON m1.member_id <> m2.member_id
        WHERE m1.member_id = :member_id
          AND m1.geom IS NOT NULL
          AND m2.geom IS NOT NULL
          AND ST_DWithin(
                m1.geom::geography,
                m2.geom::geography,
                :radius_meters
              )
        ORDER BY distance_km
        LIMIT 20
    """)

    result = db.execute(
        query,
        {
            "member_id": member_id,
            "radius_meters": radius_km * 1000
        }
    ).mappings().all()

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Member not found or no nearby members found"
        )

    return {
        "member_id": member_id,
        "radius_km": radius_km,
        "nearby_members": [dict(row) for row in result]
    }
    
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