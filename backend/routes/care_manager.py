from fastapi import APIRouter, HTTPException

from src.reasoning.contextual_reasoner import (
    ContextualReasoner,
    serialize_reasoning_result,
)

from src.prioritization.intervention_prioritizer import (
    prioritize_member,
)

from src.recommendations.recommendation_engine import (
    RecommendationEngine,
)


router = APIRouter(
    prefix="/members",
    tags=["Care Manager"]
)


# ------------------------------------------------------------
# Reasoning
# ------------------------------------------------------------

@router.get("/{member_id}/reasoning")
def get_member_reasoning(member_id: str):

    try:
        reasoner = ContextualReasoner()

        result = reasoner.reason(member_id)

        return serialize_reasoning_result(result)

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Unable to generate reasoning: {str(e)}"
        )


# ------------------------------------------------------------
# Intervention priorities
# ------------------------------------------------------------

@router.get("/{member_id}/priorities")
def get_member_priorities(member_id: str):

    try:
        reasoner = ContextualReasoner()

        reasoning_result = reasoner.reason(member_id)

        prioritization_result = prioritize_member(
            reasoning_result
        )

        return prioritization_result.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Unable to generate priorities: {str(e)}"
        )


# ------------------------------------------------------------
# Recommendations
# ------------------------------------------------------------

@router.get("/{member_id}/recommendations")
def get_member_recommendations(member_id: str):

    try:
        reasoner = ContextualReasoner()

        reasoning_result = reasoner.reason(member_id)

        prioritization_result = prioritize_member(
            reasoning_result
        )

        engine = RecommendationEngine()

        recommendation_result = engine.recommend(
            reasoning_result,
            prioritization_result,
        )

        return recommendation_result.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Unable to generate recommendations: {str(e)}"
        )


# ------------------------------------------------------------
# Complete Care Manager Summary
# ------------------------------------------------------------

@router.get("/{member_id}/summary")
def get_member_summary(member_id: str):

    try:
        # --------------------------------------------------------
        # 1. Reasoning
        # --------------------------------------------------------

        reasoner = ContextualReasoner()

        reasoning_result = reasoner.reason(
            member_id
        )

        # --------------------------------------------------------
        # 2. Prioritization
        # --------------------------------------------------------

        prioritization_result = prioritize_member(
            reasoning_result
        )

        # --------------------------------------------------------
        # 3. Recommendations
        # --------------------------------------------------------

        engine = RecommendationEngine()

        recommendation_result = engine.recommend(
            reasoning_result,
            prioritization_result,
        )

        # --------------------------------------------------------
        # 4. Aggregate existing results
        # --------------------------------------------------------

        return {
            "member_id": member_id,

            "risk": {
                "probability": (
                    reasoning_result.risk_probability
                ),
                "band": (
                    reasoning_result.risk_band
                ),
            },

            "reasoning": {
                "sdoh_factors": (
                    reasoning_result.sdoh_factors
                ),
                "sdoh_domains": (
                    reasoning_result.sdoh_domains
                ),
                "clinical_factors": (
                    reasoning_result.clinical_factors
                ),
                "evidence_records": (
                    reasoning_result.evidence_records
                ),
                "reasoning_trace": (
                    reasoning_result.reasoning_trace
                ),
            },

            "prioritization": (
                prioritization_result.to_dict()
            ),

            "recommendations": (
                recommendation_result.to_dict()
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to generate member summary: "
                f"{type(e).__name__}: {str(e)}"
            )
        )