from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.data_store import initialize_development_database
from backend.database import get_engine
from backend.routes.dashboard import router as dashboard_router
from backend.routes.members import router as members_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_development_database(get_engine())
    yield


app = FastAPI(
    title="HealthLens API",
    description="Current member-risk, SDOH, clinical, and intervention outputs.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(members_router)
app.include_router(dashboard_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
