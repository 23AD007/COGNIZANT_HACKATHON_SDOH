from pydantic import BaseModel


class MemberResponse(BaseModel):
    member_id: str
    age: float | None = None
    gender: str | None = None
    race: str | None = None
    ethnicity: str | None = None
    city: str | None = None
    state: str | None = None
    lat: float | None = None
    lon: float | None = None