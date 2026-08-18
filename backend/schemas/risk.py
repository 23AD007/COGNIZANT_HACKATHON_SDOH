from pydantic import BaseModel


class RiskResponse(BaseModel):
    member_id: str
    risk_probability: float
    risk_percentage: float
    risk_band: str