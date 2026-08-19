from fastapi import APIRouter

from backend.services.artifact_service import dashboard_summary


router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary")
def summary():
    return dashboard_summary()
