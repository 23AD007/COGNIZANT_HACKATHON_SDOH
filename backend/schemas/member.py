from pydantic import BaseModel


class MemberResponse(BaseModel):
    member_id: str
    age: float | None = None
    gender: str | None = None
    race: str | None = None
    ethnicity: str | None = None
    city: str | None = None
    state: str | None = None


class LocationResponse(BaseModel):
    member_id: str
    lat: float | None = None
    lon: float | None = None
    county: str | None = None
    county_fips: str | None = None
    city: str | None = None
    state: str | None = None
