from fastapi import APIRouter, HTTPException

from backend.schemas.knowledge import KnowledgeGraphResponse
from backend.services import knowledge_service


router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("", response_model=KnowledgeGraphResponse, response_model_exclude_none=True)
def knowledge() -> dict:
    try:
        return knowledge_service.get_knowledge_graph()
    except knowledge_service.KnowledgeGraphUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
