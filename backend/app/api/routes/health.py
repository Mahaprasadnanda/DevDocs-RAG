from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "API is running",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }
