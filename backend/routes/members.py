from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.member import LocationResponse, MemberResponse
from backend.schemas.risk import RiskResponse
from backend.services import member_service, risk_service


router = APIRouter(prefix="/members", tags=["members"])


def _required(value: dict | None, detail: str) -> dict:
    if value is None:
        raise HTTPException(status_code=404, detail=detail)
    return value


@router.get("", response_model=list[MemberResponse])
def members(db: Session = Depends(get_db)):
    return member_service.list_members(db)


@router.get("/{member_id}", response_model=MemberResponse)
def member(member_id: str, db: Session = Depends(get_db)):
    return _required(member_service.get_member(db, member_id), "Member not found")


@router.get("/{member_id}/sdoh")
def sdoh(member_id: str, db: Session = Depends(get_db)):
    return _required(member_service.get_sdoh(db, member_id), "SDOH information not found")


@router.get("/{member_id}/clinical")
def clinical(member_id: str, db: Session = Depends(get_db)):
    return _required(member_service.get_clinical(db, member_id), "Clinical information not found")


@router.get("/{member_id}/risk", response_model=RiskResponse)
def risk(member_id: str, db: Session = Depends(get_db)):
    return _required(risk_service.get_risk(db, member_id), "Risk output not found")


@router.get("/{member_id}/recommendations")
def recommendations(member_id: str, db: Session = Depends(get_db)):
    return _required(risk_service.get_recommendation(db, member_id), "Recommendation not found")


@router.get("/{member_id}/location", response_model=LocationResponse)
def location(member_id: str, db: Session = Depends(get_db)):
    return _required(member_service.get_location(db, member_id), "Location not found")
