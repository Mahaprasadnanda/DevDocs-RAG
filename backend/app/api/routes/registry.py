from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
from app.ingestion.registry import get_all_technologies

router = APIRouter()

class TechnologyItem(BaseModel):
    tech_id: str
    display_name: str
    root_url: str
    version: str
    enabled: bool

@router.get("", response_model=List[TechnologyItem])
@router.get("/", response_model=List[TechnologyItem])
def list_technologies():
    """
    Returns list of configured official documentation sources and versions.
    """
    configs = get_all_technologies(enabled_only=True)
    return [
        TechnologyItem(
            tech_id=c.tech_id,
            display_name=c.display_name,
            root_url=c.root_url,
            version=c.version,
            enabled=c.enabled
        )
        for c in configs
    ]
