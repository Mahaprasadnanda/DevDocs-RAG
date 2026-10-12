from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
from datetime import datetime, timezone

class DocumentRecord(BaseModel):
    document_id: str = Field(..., description="Stable source identifier derived from the source URL.")
    title: str = Field(..., description="Page title.")
    source_url: HttpUrl = Field(..., description="The URL the document was fetched from.")
    canonical_url: Optional[HttpUrl] = Field(None, description="The canonical URL if specified in the page.")
    headings: List[str] = Field(default_factory=list, description="Ordered section headings found in the document.")
    text: str = Field(..., description="Cleaned main text of the documentation page.")
    technology: str = Field("fastapi", description="Technology identifier, e.g. 'fastapi', 'react', 'spring-boot'.")
    doc_version: Optional[str] = Field(None, description="Documentation version label, e.g. '3.4', '19.0'.")
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when the document was fetched.")
