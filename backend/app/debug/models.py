from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from app.rag.models import SourceCitation

class DebugRequest(BaseModel):
    error_message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The error message or stack trace to diagnose."
    )
    framework: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional framework or technology, e.g. FastAPI, Starlette, Pydantic."
    )
    language: Optional[str] = Field(
        None,
        max_length=50,
        description="Optional programming language, e.g. Python, TypeScript."
    )
    context: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional description of developer action, environment, or intent when error occurred."
    )
    top_k: int = Field(
        8,
        gt=0,
        le=20,
        description="Maximum number of context chunks to retrieve."
    )
    min_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold."
    )

    @field_validator("error_message")
    @classmethod
    def validate_error_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Error message cannot be empty or whitespace only.")
        return v.strip()


class LikelyCause(BaseModel):
    cause: str = Field(..., description="Short title or description of the cause.")
    explanation: str = Field(..., description="Detailed explanation of why this error may occur.")
    citations: List[str] = Field(default_factory=list, description="Citation identifiers supporting this cause, e.g. ['S1'].")


class SuggestedFix(BaseModel):
    fix: str = Field(..., description="Short title or description of the fix.")
    explanation: str = Field(..., description="Concrete corrective action or code guidance.")
    citations: List[str] = Field(default_factory=list, description="Citation identifiers supporting this fix, e.g. ['S1'].")


class DebugResponse(BaseModel):
    error_summary: str = Field(..., description="Concise summary of the reported error.")
    likely_causes: List[LikelyCause] = Field(default_factory=list, description="Plausible causes with explanations and citations.")
    diagnostic_checks: List[str] = Field(default_factory=list, description="Ordered steps the developer can perform to investigate the problem.")
    suggested_fixes: List[SuggestedFix] = Field(default_factory=list, description="Possible corrective actions with citations where supported.")
    sources: List[SourceCitation] = Field(default_factory=list, description="Validated sources actually referenced in the response.")
    total_sources: int = Field(..., description="Count of sources provided in the response.")
    uncertainty_notes: List[str] = Field(default_factory=list, description="What cannot be determined from the error or available documentation.")
    model: str = Field(..., description="The LLM model identifier used for generation.")
