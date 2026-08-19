from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class CountySummary(BaseModel):
    county_fips: str
    county_name: str
    state_abbr: str
    state_name: str
    member_count: int
    risk_member_count: int
    risk_coverage: float
    recommendation_available: bool
    data_sources: list[str]


class CountyDetail(CountySummary):
    mean_risk: float | None
    median_risk: float | None
    high_risk_count: int
    very_high_risk_count: int
    high_or_very_high_count: int
    high_or_very_high_percentage: float | None
    risk_band: str | None
    risk_distribution: dict[str, int]
    county_features: dict[str, Any] | None = None
    sdoh_features: dict[str, Any] | None = None
    sram_features: dict[str, Any] | None = None
