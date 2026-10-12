import re
import logging
from typing import List, Set, Optional, Dict, Any
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


def is_url_allowed(candidate_url: str, allowed_source_urls: Set[str]) -> bool:
    """
    Validate that candidate_url strictly matches an allowed source URL or is an
    exact subpath/fragment of it, enforcing strict hostname boundaries, scheme, and port checks.
    Rejects lookalike domains (e.g. fastapi.tiangolo.com.evil.com, fastapi.tiangolo.com@evil.com).
    """
    candidate = urlparse(candidate_url)
    if candidate.scheme.lower() not in ("http", "https"):
        return False

    cand_host = (candidate.hostname or "").lower()
    if not cand_host or cand_host in LOCAL_EXAMPLE_HOSTS:
        return True

    cand_port = candidate.port

    for allowed_str in allowed_source_urls:
        allowed = urlparse(allowed_str)
        allowed_host = (allowed.hostname or "").lower()
        if not allowed_host:
            continue

        # Strict exact hostname match - forbids prefix, suffix, and lookalike domain spoofing
        if cand_host != allowed_host:
            continue

        # Strict scheme check
        if candidate.scheme.lower() != allowed.scheme.lower():
            continue

        # Port match
        if cand_port != allowed.port:
            continue

        # Path match with strict directory boundary
        cand_path = candidate.path.rstrip("/")
        allow_path = allowed.path.rstrip("/")

        if cand_path == allow_path or cand_path.startswith(allow_path + "/"):
            return True

    return False


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
    3. No fabricated documentation URLs exist (with strict boundary checking).
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
    allowed_urls: Set[str] = {str(c.source_url).rstrip("/") for c in allowed_citations}

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

    # Detect fabricated URLs using strict boundary check
    found_urls = extract_external_urls(sanitized)
    fabricated_urls = [u for u in found_urls if not is_url_allowed(u, allowed_urls)]

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
    Specifically checks:
    - error_summary
    - likely_causes[].cause, likely_causes[].explanation, likely_causes[].citations
    - diagnostic_checks[]
    - suggested_fixes[].fix, suggested_fixes[].explanation, suggested_fixes[].citations
    - uncertainty_notes[]
    - raw generated JSON for external URLs and malformed citations
    """
    allowed_id_set: Set[str] = {c.source_id for c in allowed_citations}
    allowed_urls: Set[str] = {str(c.source_url).rstrip("/") for c in allowed_citations}

    all_cited: List[str] = []
    invalid_ids: List[str] = []
    fabricated_urls: List[str] = []

    def check_text_snippet(text: Any):
        if not text or not isinstance(text, str):
            return
        # 1. Valid citations
        all_cited.extend(extract_citations(text))
        # 2. Malformed citations (e.g. [Source 1])
        for m in MALFORMED_CITATION_REGEX.findall(text):
            invalid_ids.append(f"Source {m}")
        # 3. External URLs
        for u in extract_external_urls(text):
            if not is_url_allowed(u, allowed_urls):
                fabricated_urls.append(u)

    # 1. Validate error_summary
    check_text_snippet(debug_dict.get("error_summary"))

    # 2. Validate likely_causes
    for cause in debug_dict.get("likely_causes", []):
        if isinstance(cause, dict):
            check_text_snippet(cause.get("cause"))
            check_text_snippet(cause.get("explanation"))
            for c_id in cause.get("citations", []):
                cleaned_id = re.sub(r'[\[\]]', '', str(c_id)).strip().upper()
                if not cleaned_id or cleaned_id not in allowed_id_set:
                    invalid_ids.append(str(c_id))
                else:
                    all_cited.append(cleaned_id)
        else:
            invalid_ids.append(f"malformed_cause_{type(cause).__name__}")

    # 3. Validate diagnostic_checks
    for check in debug_dict.get("diagnostic_checks", []):
        check_text_snippet(check)

    # 4. Validate suggested_fixes
    for fix in debug_dict.get("suggested_fixes", []):
        if isinstance(fix, dict):
            check_text_snippet(fix.get("fix"))
            check_text_snippet(fix.get("explanation"))
            for c_id in fix.get("citations", []):
                cleaned_id = re.sub(r'[\[\]]', '', str(c_id)).strip().upper()
                if not cleaned_id or cleaned_id not in allowed_id_set:
                    invalid_ids.append(str(c_id))
                else:
                    all_cited.append(cleaned_id)
        else:
            invalid_ids.append(f"malformed_fix_{type(fix).__name__}")

    # 5. Validate uncertainty_notes
    for note in debug_dict.get("uncertainty_notes", []):
        check_text_snippet(note)

    # 6. Validate raw_json for any leaked external URLs or citations
    check_text_snippet(raw_json)

    # Detect unauthorized citation IDs among cited
    for cid in all_cited:
        if cid not in allowed_id_set:
            invalid_ids.append(cid)

    unique_cited = sorted(list(dict.fromkeys([c for c in all_cited if c in allowed_id_set])))
    invalid_ids = sorted(list(dict.fromkeys(invalid_ids)))
    fabricated_urls = sorted(list(dict.fromkeys(fabricated_urls)))

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
