from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class ChunkRecord(BaseModel):
    chunk_id: str = Field(..., description="Stable identifier for this chunk.")
    document_id: str = Field(..., description="The ID of the source document.")
    title: str = Field(..., description="The title of the source document.")
    source_url: HttpUrl = Field(..., description="The source URL of the document.")
    canonical_url: Optional[HttpUrl] = Field(None, description="The canonical URL if specified.")
    section_heading: Optional[str] = Field(None, description="The most relevant heading for this chunk, if any.")
    chunk_index: int = Field(..., description="The zero-based index of this chunk within the document.")
    text: str = Field(..., description="The text content of the chunk.")
