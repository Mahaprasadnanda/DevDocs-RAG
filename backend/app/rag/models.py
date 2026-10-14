from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

class AskRequest(BaseModel):
    query: str = Field(..., description="The user's technical documentation question.")
    top_k: int = Field(8, gt=0, le=20, description="Maximum number of context chunks to retrieve.")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score threshold.")
    technology: Optional[str] = Field(None, description="Optional technology filter: e.g. 'fastapi', 'react', 'spring-boot', 'all', or None for auto-detect.")
    version: Optional[str] = Field(None, description="Optional version filter.")

class SourceCitation(BaseModel):
    source_id: str = Field(..., description="Stable identifier used in citations, e.g. S1.")
    chunk_id: str = Field(..., description="ID of the underlying chunk.")
    document_id: str = Field(..., description="Source document identifier.")
    section_title: Optional[str] = Field(None, description="Section heading from documentation.")
    source_url: HttpUrl = Field(..., description="Canonical or source documentation URL.")
    score: float = Field(..., description="Vector similarity score.")
    technology: Optional[str] = Field(None, description="Technology identifier.")
    doc_version: Optional[str] = Field(None, description="Documentation version.")
    ecosystem_component: Optional[str] = Field(None, description="Ecosystem component identifier, e.g. 'spring-boot-core', 'spring-framework', 'spring-cloud-gateway', 'resilience4j'.")

class AskResponse(BaseModel):
    query: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="The generated grounded answer.")
    sources: List[SourceCitation] = Field(default_factory=list, description="Retrieved source citations supporting the answer.")
    total_sources: int = Field(..., description="Count of sources provided in the response.")
    model: str = Field(..., description="The LLM model identifier used for generation.")
    detected_technology: Optional[str] = Field(None, description="Selected or detected technology identifier.")
