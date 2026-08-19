from fastapi import APIRouter, HTTPException
from backend.schemas.member import LocationResponse, MemberResponse
from backend.schemas.risk import RiskResponse
from backend.services import artifact_service


router = APIRouter(prefix="/members", tags=["members"])


def _required(value: dict | None, detail: str) -> dict:
    if value is None:
        raise HTTPException(status_code=404, detail=detail)
    return value


@router.get("", response_model=list[MemberResponse])
def members():
    return artifact_service.list_members()


@router.get("/{member_id}", response_model=MemberResponse)
def member(member_id: str):
    return _required(artifact_service.get_member(member_id), "Member not found")


@router.get("/{member_id}/sdoh")
def sdoh(member_id: str):
    return _required(artifact_service.get_sdoh(member_id), "SDOH information not found")


@router.get("/{member_id}/clinical")
def clinical(member_id: str):
    return _required(artifact_service.get_clinical(member_id), "Clinical information not found")


@router.get("/{member_id}/risk", response_model=RiskResponse)
def risk(member_id: str):
    return _required(artifact_service.get_risk(member_id), "Risk output not found")


@router.get("/{member_id}/recommendations")
def recommendations(member_id: str):
    return _required(artifact_service.get_recommendation(member_id), "Recommendation could not be generated")


@router.get("/{member_id}/location", response_model=LocationResponse)
def location(member_id: str):
    return _required(artifact_service.get_location(member_id), "Location not found")
