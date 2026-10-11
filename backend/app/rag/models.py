from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

class AskRequest(BaseModel):
    query: str = Field(..., description="The user's technical documentation question.")
    top_k: int = Field(8, gt=0, le=20, description="Maximum number of context chunks to retrieve.")

class SourceCitation(BaseModel):
    source_id: str = Field(..., description="Stable identifier used in citations, e.g. S1.")
    chunk_id: str = Field(..., description="ID of the underlying chunk.")
    document_id: str = Field(..., description="Source document identifier.")
    section_title: Optional[str] = Field(None, description="Section heading from documentation.")
    source_url: HttpUrl = Field(..., description="Canonical or source documentation URL.")
    score: float = Field(..., description="Vector similarity score.")

class AskResponse(BaseModel):
    query: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="The generated grounded answer.")
    sources: List[SourceCitation] = Field(default_factory=list, description="Retrieved source citations supporting the answer.")
    total_sources: int = Field(..., description="Count of sources provided in the response.")
    model: str = Field(..., description="The LLM model identifier used for generation.")
