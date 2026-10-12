from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import health, retrieve, ask, debug, registry

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="DevDocs RAG Backend API"
)

app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(retrieve.router, prefix="/api/v1/retrieve", tags=["retrieve"])
app.include_router(ask.router, prefix="/api/v1/ask", tags=["ask"])
app.include_router(debug.router, prefix="/api/v1/debug", tags=["debug"])
app.include_router(registry.router, prefix="/api/v1/registry", tags=["registry"])
