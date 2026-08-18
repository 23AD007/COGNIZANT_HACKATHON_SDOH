from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.risk import RiskResponse
from backend.services.risk_service import (
    predict_member_risk,
    get_risk_band,
    FEATURE_COLUMNS,
)


router = APIRouter(
    prefix="/members",
    tags=["Risk"]
)


@router.get(
    "/{member_id}/risk",
    response_model=RiskResponse
)
def get_member_risk(
    member_id: str,
    db: Session = Depends(get_db)
):

    # Get member SDOH data
    sdoh_query = text("""
        SELECT *
        FROM sdoh.member_sdoh_features
        WHERE member_id = :member_id
    """)

    sdoh = (
        db.execute(
            sdoh_query,
            {"member_id": member_id}
        )
        .mappings()
        .first()
    )

    if sdoh is None:
        raise HTTPException(
            status_code=404,
            detail="Member not found"
        )

    # Get clinical summary
    clinical_query = text("""
        SELECT
            patient_id,
            encounter_count,
            condition_count,
            medication_count,
            procedure_count
        FROM sdoh.clinical_patient_coverage
        WHERE patient_id = :member_id
    """)

    clinical = (
        db.execute(
            clinical_query,
            {"member_id": member_id}
        )
        .mappings()
        .first()
    )

    features = dict(sdoh)

    if clinical:
        features["clinical_encounter_count"] = (
            clinical["encounter_count"] or 0
        )

        features["clinical_condition_count"] = (
            clinical["condition_count"] or 0
        )

        features["clinical_medication_count"] = (
            clinical["medication_count"] or 0
        )

        features["clinical_procedure_count"] = (
            clinical["procedure_count"] or 0
        )

    # Check whether all 68 model features are available.
    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in features
    ]

    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Risk model input is incomplete",
                "missing_features": missing,
            }
        )

    try:
        probability = predict_member_risk(features)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Risk prediction failed: {str(e)}"
        )

    return {
        "member_id": member_id,
        "risk_probability": probability,
        "risk_percentage": round(
            probability * 100,
            2
        ),
        "risk_band": get_risk_band(probability),
    }