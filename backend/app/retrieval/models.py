from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class RetrievalRequest(BaseModel):
    query: str = Field(..., description="The natural language query.")
    top_k: int = Field(8, gt=0, description="Maximum number of results to return.")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score threshold.")
    technology: Optional[str] = Field(None, description="Optional technology filter: e.g. 'fastapi', 'react', 'spring-boot', 'all', or None for auto-detect.")
    version: Optional[str] = Field(None, description="Optional documentation version filter.")

class RetrievalResult(BaseModel):
    chunk_id: str
    text: str
    source_url: HttpUrl
    document_id: str
    section_title: Optional[str] = None
    score: float
    technology: Optional[str] = None
    doc_version: Optional[str] = None

class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]
    total_results: int
    detected_technology: Optional[str] = None
