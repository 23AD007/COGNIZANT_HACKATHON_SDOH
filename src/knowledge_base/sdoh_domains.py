"""
HealthLens SDOH domain knowledge base.

This module contains the canonical SDOH domain definitions.

The knowledge base is intentionally static.
Member-specific observations belong in the Knowledge Graph,
not here.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SDOHDomain:
    """
    Canonical SDOH domain.
    """

    key: str
    name: str
    description: str


SDOH_DOMAINS: Dict[str, SDOHDomain] = {
    "economic_stability": SDOHDomain(
        key="economic_stability",
        name="Economic Stability",
        description=(
            "Economic conditions affecting a member's ability "
            "to obtain healthcare, food, housing, and other necessities."
        ),
    ),

    "education_access": SDOHDomain(
        key="education_access",
        name="Education Access",
        description=(
            "Educational attainment and access to educational "
            "resources that may influence health literacy and opportunity."
        ),
    ),

    "healthcare_access": SDOHDomain(
        key="healthcare_access",
        name="Healthcare Access",
        description=(
            "Access to healthcare services, preventive care, "
            "screening, insurance coverage, and routine care."
        ),
    ),

    "neighborhood_built_environment": SDOHDomain(
        key="neighborhood_built_environment",
        name="Neighborhood and Built Environment",
        description=(
            "Physical and environmental characteristics of the "
            "communities in which members live."
        ),
    ),

    "social_community_context": SDOHDomain(
        key="social_community_context",
        name="Social and Community Context",
        description=(
            "Social relationships, community conditions, and "
            "social circumstances that may influence health."
        ),
    ),

    "housing": SDOHDomain(
        key="housing",
        name="Housing",
        description=(
            "Housing stability, affordability, crowding, vacancy, "
            "and housing-related transportation constraints."
        ),
    ),

    "transportation": SDOHDomain(
        key="transportation",
        name="Transportation",
        description=(
            "Transportation availability and geographic access "
            "to healthcare and essential services."
        ),
    ),

    "digital_access": SDOHDomain(
        key="digital_access",
        name="Digital Access",
        description=(
            "Access to computers, broadband, and digital resources "
            "that can enable telehealth and digital health services."
        ),
    ),
}


def get_sdoh_domain(key: str) -> SDOHDomain:
    """
    Return a domain by canonical key.
    """

    normalized = str(key).strip().lower()

    if normalized not in SDOH_DOMAINS:
        raise KeyError(f"Unknown SDOH domain: {key}")

    return SDOH_DOMAINS[normalized]


def get_all_sdoh_domains() -> List[SDOHDomain]:
    """
    Return all canonical SDOH domains.
    """

    return list(SDOH_DOMAINS.values())


def validate_domains() -> None:
    """
    Validate the domain knowledge base.
    """

    expected = {
        "economic_stability",
        "education_access",
        "healthcare_access",
        "neighborhood_built_environment",
        "social_community_context",
        "housing",
        "transportation",
        "digital_access",
    }

    actual = set(SDOH_DOMAINS.keys())

    if actual != expected:
        raise ValueError(
            "SDOH domain definition mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Actual:   {sorted(actual)}"
        )

    for key, domain in SDOH_DOMAINS.items():
        if domain.key != key:
            raise ValueError(
                f"Domain key mismatch: {key} != {domain.key}"
            )

        if not domain.name.strip():
            raise ValueError(
                f"Domain {key} has no display name."
            )


def _self_test() -> None:
    print("=" * 70)
    print("HEALTHLENS KNOWLEDGE BASE — SDOH DOMAINS")
    print("=" * 70)

    print("Knowledge base version: 1.0.0")
    print(f"SDOH domains:           {len(SDOH_DOMAINS)}")

    validate_domains()

    print()
    print("DOMAINS")
    print("-" * 70)

    for domain in get_all_sdoh_domains():
        print(
            f"{domain.key:<35} {domain.name}"
        )

    print()
    print("SDOH DOMAIN SELF-TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()