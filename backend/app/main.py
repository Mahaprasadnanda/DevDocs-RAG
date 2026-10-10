from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import health

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="DevDocs RAG Backend API"
)

app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
