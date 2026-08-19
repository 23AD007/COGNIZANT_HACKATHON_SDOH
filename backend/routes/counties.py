from fastapi import APIRouter, HTTPException

from backend.schemas.county import CountyDetail, CountySummary
from backend.services import county_service

router = APIRouter(prefix="/counties", tags=["counties"])


def _county_or_404(county_fips: str) -> dict:
    try:
        county = county_service.get_county(county_fips)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="County not found") from exc
    if county is None:
        raise HTTPException(status_code=404, detail="County not found")
    return county


@router.get("", response_model=list[CountySummary])
def counties():
    return county_service.list_counties()


@router.get("/locations")
def locations():
    return county_service.get_county_locations()


@router.get("/{county_fips}", response_model=CountyDetail)
def county(county_fips: str):
    return _county_or_404(county_fips)


@router.get("/{county_fips}/recommendations")
def recommendations(county_fips: str):
    _county_or_404(county_fips)
    result = county_service.get_county_recommendations(county_fips)
    if result is None:
        raise HTTPException(status_code=404, detail="County not found")
    if not result["recommendations"]:
        raise HTTPException(status_code=422, detail="County has insufficient SDOH data for population prioritization")
    return result
