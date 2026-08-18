from fastapi import FastAPI

from backend.routes.members import router as members_router
from backend.routes.risk import router as risk_router
from backend.routes.care_manager import router as care_manager_router

app = FastAPI(
    title="HealthLens Care Manager API",
    description="API for Care Manager access to member SDOH and clinical information",
    version="1.0.0"
)


# Include routers AFTER creating the FastAPI app
app.include_router(members_router)
app.include_router(risk_router)
app.include_router(care_manager_router)

@app.get("/")
def root():
    return {
        "message": "HealthLens Care Manager API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }