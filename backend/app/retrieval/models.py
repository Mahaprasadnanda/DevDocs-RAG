from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class RetrievalRequest(BaseModel):
    query: str = Field(..., description="The natural language query.")
    top_k: int = Field(8, gt=0, description="Maximum number of results to return.")

class RetrievalResult(BaseModel):
    chunk_id: str
    text: str
    source_url: HttpUrl
    document_id: str
    section_title: Optional[str] = None
    score: float

class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]
    total_results: int
