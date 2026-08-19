from pydantic import BaseModel


class RiskResponse(BaseModel):
    member_id: str
    risk_probability: float
    risk_band: str
    risk_rank: int | None = None
    risk_percentile: float | None = None
