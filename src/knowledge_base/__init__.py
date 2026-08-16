"""
HealthLens Knowledge Base.

Static domain and factor definitions used by the
Knowledge Graph and intervention reasoning layers.
"""

from .sdoh_domains import (
    SDOHDomain,
    SDOH_DOMAINS,
    get_sdoh_domain,
    get_all_sdoh_domains,
)

from .sdoh_factors import (
    SDOHFactor,
    SDOH_FACTORS,
    get_sdoh_factor,
    get_all_sdoh_factors,
    get_factors_by_domain,
)


__all__ = [
    "SDOHDomain",
    "SDOH_DOMAINS",
    "get_sdoh_domain",
    "get_all_sdoh_domains",
    "SDOHFactor",
    "SDOH_FACTORS",
    "get_sdoh_factor",
    "get_all_sdoh_factors",
    "get_factors_by_domain",
]