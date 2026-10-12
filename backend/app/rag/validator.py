import re
import logging
from typing import List, Set, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from app.rag.models import SourceCitation

logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(r'\[S(\d+)\]')
MALFORMED_CITATION_REGEX = re.compile(
    r'\[(?:Source|Doc|Ref|Section)\s*(\d+)\]',
    re.IGNORECASE
)
URL_REGEX = re.compile(r'https?://[^\s\)\]\>\"\']+')

REFUSAL_PHRASES = (
    "does not contain enough information",
    "does not contain sufficient information",
    "not contain enough relevant information",
    "cannot be answered based on the provided",
    "documentation does not provide",
    "documentation does not contain",
    "not enough information in the provided",
    "no information provided",
)

LOCAL_EXAMPLE_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "example.com", "json-schema.org")


class CitationValidationResult(BaseModel):
    is_valid: bool = Field(..., description="Whether the answer passed all citation and grounding checks.")
    is_refusal: bool = Field(..., description="Whether the answer is a valid refusal due to insufficient info.")
    cited_source_ids: List[str] = Field(default_factory=list, description="Unique valid citation identifiers in text.")
    invalid_source_ids: List[str] = Field(default_factory=list, description="Unauthorized citation identifiers in text.")
    fabricated_urls: List[str] = Field(default_factory=list, description="URLs in text not present in retrieved context.")
    error_message: Optional[str] = Field(None, description="Diagnostic error message explaining validation failure.")
    sanitized_answer: str = Field(..., description="Normalized answer text.")


def is_refusal_response(text: str) -> bool:
    """
    Check if the generated response is a truthful refusal to answer due to insufficient context.
    """
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def extract_citations(text: str) -> List[str]:
    """
    Extract all [S{n}] citation identifiers from the text in order of appearance.
    """
    matches = CITATION_REGEX.findall(text)
    return [f"S{num}" for num in matches]


def extract_external_urls(text: str) -> List[str]:
    """
    Extract documentation and external URLs from text, excluding local test/mock hosts.
    """
    raw_urls = URL_REGEX.findall(text)
    external_urls = []
    for u in raw_urls:
        cleaned = u.rstrip(".,;:)[]")
        parsed = urlparse(cleaned)
        host = (parsed.hostname or "").lower()
        if host and host not in LOCAL_EXAMPLE_HOSTS:
            external_urls.append(cleaned)
    return external_urls


def normalize_citation_syntax(text: str) -> str:
    """
    Normalize minor syntax variations such as lowercase [s1] to canonical [S1].
    """
    return re.sub(r'\[s(\d+)\]', r'[S\1]', text)


def validate_citations(
    answer: str,
    allowed_citations: List[SourceCitation]
) -> CitationValidationResult:
    """
    Validate that:
    1. Every citation [S{n}] in answer belongs to the allowed_citations retrieved for this request.
    2. No fabricated or unknown source identifiers exist.
    3. No fabricated documentation URLs exist.
    4. An informative grounded answer includes at least one valid citation.
    5. A truthful refusal requires no citations and is accepted.
    """
    if not answer or not answer.strip():
        return CitationValidationResult(
            is_valid=False,
            is_refusal=False,
            cited_source_ids=[],
            invalid_source_ids=[],
            fabricated_urls=[],
            error_message="Answer is empty or whitespace only.",
            sanitized_answer=""
        )

    sanitized = normalize_citation_syntax(answer.strip())

    # Check if this is an explicit refusal
    if is_refusal_response(sanitized):
        return CitationValidationResult(
            is_valid=True,
            is_refusal=True,
            cited_source_ids=[],
            invalid_source_ids=[],
            fabricated_urls=[],
            error_message=None,
            sanitized_answer=sanitized
        )

    # Allowed set of citation IDs for the current request
    allowed_id_set: Set[str] = {c.source_id for c in allowed_citations}

    # Extract all citations
    found_citations = extract_citations(sanitized)
    unique_cited = sorted(list(dict.fromkeys(found_citations)))

    # Detect unauthorized source IDs
    invalid_ids = [sid for sid in unique_cited if sid not in allowed_id_set]

    # Detect malformed citations (e.g. [Source 1], [Doc 2])
    malformed_matches = MALFORMED_CITATION_REGEX.findall(sanitized)
    if malformed_matches:
        for m in malformed_matches:
            invalid_ids.append(f"Source {m}")

    # Detect fabricated URLs
    allowed_urls = {str(c.source_url).rstrip("/") for c in allowed_citations}
    found_urls = extract_external_urls(sanitized)
    fabricated_urls = []
    for u in found_urls:
        u_clean = u.rstrip("/")
        # URL must match or be a fragment/sub-path of an allowed source URL
        matched = any(
            u_clean == au or u_clean.startswith(au + "#") or u_clean.startswith(au + "/")
            for au in allowed_urls
        )
        if not matched:
            fabricated_urls.append(u)

    # Check validation rules
    if invalid_ids:
        error = f"Answer contains unauthorized source identifier(s): {', '.join(invalid_ids)}."
        logger.warning(f"Citation validation error: {error}")
        return CitationValidationResult(
            is_valid=False,
            is_refusal=False,
            cited_source_ids=unique_cited,
            invalid_source_ids=invalid_ids,
            fabricated_urls=fabricated_urls,
            error_message=error,
            sanitized_answer=sanitized
        )

    if fabricated_urls:
        error = f"Answer contains fabricated documentation URL(s): {', '.join(fabricated_urls)}."
        logger.warning(f"Citation validation error: {error}")
        return CitationValidationResult(
            is_valid=False,
            is_refusal=False,
            cited_source_ids=unique_cited,
            invalid_source_ids=[],
            fabricated_urls=fabricated_urls,
            error_message=error,
            sanitized_answer=sanitized
        )

    if not unique_cited:
        error = "Answer contains no inline citations [S1], [S2] to retrieved documentation."
        logger.warning(f"Citation validation error: {error}")
        return CitationValidationResult(
            is_valid=False,
            is_refusal=False,
            cited_source_ids=[],
            invalid_source_ids=[],
            fabricated_urls=[],
            error_message=error,
            sanitized_answer=sanitized
        )

    return CitationValidationResult(
        is_valid=True,
        is_refusal=False,
        cited_source_ids=unique_cited,
        invalid_source_ids=[],
        fabricated_urls=[],
        error_message=None,
        sanitized_answer=sanitized
    )


def validate_debug_response_citations(
    debug_dict: dict,
    raw_json: str,
    allowed_citations: List[SourceCitation]
) -> CitationValidationResult:
    """
    Validate citations and documentation URLs across all fields of a structured debugging response.
    """
    allowed_id_set: Set[str] = {c.source_id for c in allowed_citations}
    allowed_urls = {str(c.source_url).rstrip("/") for c in allowed_citations}

    all_cited: List[str] = []

    # 1. Extract from likely_causes
    for cause in debug_dict.get("likely_causes", []):
        if isinstance(cause, dict):
            for c_id in cause.get("citations", []):
                cleaned_id = re.sub(r'[\[\]]', '', str(c_id)).strip().upper()
                if cleaned_id:
                    all_cited.append(cleaned_id)
            all_cited.extend(extract_citations(str(cause.get("explanation", ""))))

    # 2. Extract from suggested_fixes
    for fix in debug_dict.get("suggested_fixes", []):
        if isinstance(fix, dict):
            for c_id in fix.get("citations", []):
                cleaned_id = re.sub(r'[\[\]]', '', str(c_id)).strip().upper()
                if cleaned_id:
                    all_cited.append(cleaned_id)
            all_cited.extend(extract_citations(str(fix.get("explanation", ""))))

    # 3. Extract from raw_json overall
    all_cited.extend(extract_citations(raw_json))

    unique_cited = sorted(list(dict.fromkeys(all_cited)))

    # Detect unauthorized citations
    invalid_ids = [sid for sid in unique_cited if sid not in allowed_id_set]

    # Detect malformed citations (e.g. [Source 1])
    malformed = MALFORMED_CITATION_REGEX.findall(raw_json)
    for m in malformed:
        invalid_ids.append(f"Source {m}")

    # Detect fabricated URLs
    found_urls = extract_external_urls(raw_json)
    fabricated_urls = []
    for u in found_urls:
        u_clean = u.rstrip("/")
        matched = any(
            u_clean == au or u_clean.startswith(au + "#") or u_clean.startswith(au + "/")
            for au in allowed_urls
        )
        if not matched:
            fabricated_urls.append(u)

    if invalid_ids:
        error = f"Debugging response contains unauthorized source identifier(s): {', '.join(invalid_ids)}."
        logger.warning(f"Debug citation validation error: {error}")
        return CitationValidationResult(
            is_valid=False,
            is_refusal=False,
            cited_source_ids=unique_cited,
            invalid_source_ids=invalid_ids,
            fabricated_urls=fabricated_urls,
            error_message=error,
            sanitized_answer=raw_json
        )

    if fabricated_urls:
        error = f"Debugging response contains fabricated documentation URL(s): {', '.join(fabricated_urls)}."
        logger.warning(f"Debug citation validation error: {error}")
        return CitationValidationResult(
            is_valid=False,
            is_refusal=False,
            cited_source_ids=unique_cited,
            invalid_source_ids=[],
            fabricated_urls=fabricated_urls,
            error_message=error,
            sanitized_answer=raw_json
        )

    # Check for empty citations: allowed if it's an insufficient documentation diagnosis
    summary_text = str(debug_dict.get("error_summary", ""))
    notes_text = " ".join([str(n) for n in debug_dict.get("uncertainty_notes", [])])
    is_refusal = is_refusal_response(summary_text) or is_refusal_response(notes_text) or not debug_dict.get("likely_causes")

    if not unique_cited and not is_refusal:
        error = "Debugging response contains no source citations supporting the diagnosis."
        logger.warning(f"Debug citation validation error: {error}")
        return CitationValidationResult(
            is_valid=False,
            is_refusal=False,
            cited_source_ids=[],
            invalid_source_ids=[],
            fabricated_urls=[],
            error_message=error,
            sanitized_answer=raw_json
        )

    return CitationValidationResult(
        is_valid=True,
        is_refusal=is_refusal,
        cited_source_ids=unique_cited,
        invalid_source_ids=[],
        fabricated_urls=[],
        error_message=None,
        sanitized_answer=raw_json
    )
