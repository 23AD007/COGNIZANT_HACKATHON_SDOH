from __future__ import annotations

from fastapi import FastAPI

from backend.routes.counties import router as counties_router
from backend.routes.dashboard import router as dashboard_router
from backend.routes.members import router as members_router
from backend.routes.knowledge import router as knowledge_router


app = FastAPI(
    title="HealthLens API",
    description="Current member-risk, SDOH, clinical, and intervention outputs.",
    version="1.0.0",
)
app.include_router(members_router)
app.include_router(dashboard_router)
app.include_router(counties_router)
app.include_router(knowledge_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
