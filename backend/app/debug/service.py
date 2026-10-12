import json
import logging
import re
from typing import Optional, List, Dict, Any
from app.core.config import settings
from app.retrieval.service import retrieval_service, RetrievalService
from app.llm.client import (
    groq_client,
    GroqLLMClient,
    GroqProviderError,
    GroqTimeoutError,
    GroqConfigurationError,
)
from app.rag.context import build_context
from app.rag.models import SourceCitation
from app.rag.validator import (
    validate_debug_response_citations,
    extract_citations,
    CitationValidationResult,
)
from app.debug.models import (
    DebugRequest,
    DebugResponse,
    LikelyCause,
    SuggestedFix,
    RawDebugPayload,
)
from app.debug.prompts import (
    DEBUG_SYSTEM_PROMPT,
    format_debug_user_prompt,
)

logger = logging.getLogger(__name__)

# Patterns for redacting sensitive secrets from diagnostic input
REDACTION_PATTERNS = [
    (re.compile(r'(?i)(bearer\s+[a-zA-Z0-9_\-\.]{8,})'), 'Bearer [REDACTED]'),
    (re.compile(r'(?i)(api[_-]?key\s*[:=]\s*["\']?)[a-zA-Z0-9_\-\.]{8,}'), r'\1[REDACTED]'),
    (re.compile(r'(?i)(password\s*[:=]\s*["\']?)[^"\'\s]{4,}'), r'\1[REDACTED]'),
    (re.compile(r'(?i)(secret\s*[:=]\s*["\']?)[a-zA-Z0-9_\-\.]{8,}'), r'\1[REDACTED]'),
    (re.compile(r'(?i)([a-z]+://[^:]+:)[^@\s]+(@[^\s]+)'), r'\1[REDACTED]\2'),
]


def sanitize_diagnostic_input(text: str) -> str:
    """
    Redact obvious credentials, tokens, and passwords from diagnostic inputs.
    """
    if not text:
        return ""
    sanitized = text
    for pattern, repl in REDACTION_PATTERNS:
        sanitized = pattern.sub(repl, sanitized)
    return sanitized


def build_debug_retrieval_query(
    error_message: str,
    framework: Optional[str] = None,
    language: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Construct a concise, semantically dense retrieval query from diagnostic inputs.
    Extracts the root error/exception line from stack traces and combines with tech stack keywords.
    """
    # Look for the last line or exception line of a traceback
    lines = [line.strip() for line in error_message.split("\n") if line.strip()]
    core_error = ""

    # Common exception formats: "SomeError: detail" or "Error: detail"
    for line in reversed(lines):
        if any(err_token in line for err_token in ("Error", "Exception", "Fault", "Failed", "detail", "ValidationError")):
            core_error = line
            break

    if not core_error and lines:
        core_error = lines[-1]

    # Strip local memory addresses (e.g. 0x00007ff...) and long file paths
    core_error = re.sub(r'0x[0-9a-fA-F]+', '', core_error)
    core_error = re.sub(r'File ".*?", line \d+, in ', '', core_error)
    core_error = core_error.strip()

    query_parts: List[str] = []
    if framework:
        query_parts.append(framework.strip())
    if language:
        query_parts.append(language.strip())
    if core_error:
        query_parts.append(core_error)
    elif error_message:
        query_parts.append(error_message[:150].strip())

    if context:
        # Include a short snippet of context if space permits
        ctx_clean = context.strip()[:100]
        query_parts.append(ctx_clean)

    combined_query = " ".join(query_parts)
    # Bound total query to ~250 characters for optimal embedding performance
    return combined_query[:250].strip()


def parse_debug_json(raw_text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM output, handling markdown code fences and surrounding prose.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    return json.loads(text)


class DebugService:
    def __init__(
        self,
        retrieval_svc: Optional[RetrievalService] = None,
        llm: Optional[GroqLLMClient] = None
    ):
        self.retrieval_svc = retrieval_svc or retrieval_service
        self.llm = llm or groq_client

    def diagnose(self, request: DebugRequest) -> DebugResponse:
        """
        Orchestrate documentation-grounded error diagnosis with citation validation.
        """
        # 1. Sanitize sensitive inputs
        clean_error = sanitize_diagnostic_input(request.error_message)
        clean_context = sanitize_diagnostic_input(request.context) if request.context else None

        logger.info(
            f"Diagnosing error for framework='{request.framework}' language='{request.language}': "
            f"'{clean_error[:60]}...'"
        )

        # 2. Construct focused retrieval query
        retrieval_query = build_debug_retrieval_query(
            clean_error,
            request.framework,
            request.language,
            clean_context
        )
        logger.info(f"Generated debug retrieval query: '{retrieval_query}'")

        # 3. Retrieve relevant documentation chunks using configured threshold
        results = self.retrieval_svc.retrieve(
            retrieval_query,
            top_k=request.top_k,
            min_score=request.min_score
        )

        # 4. Handle empty retrieval result (no relevant chunks above threshold)
        if not results:
            logger.info("No matching documentation chunks met relevance threshold for error.")
            return DebugResponse(
                error_summary="The provided error does not match any indexed documentation topics.",
                likely_causes=[
                    LikelyCause(
                        cause="Insufficient Documentation Evidence",
                        explanation=(
                            "The indexed documentation (FastAPI tutorials) does not contain "
                            "evidence or patterns matching this error."
                        ),
                        citations=[]
                    )
                ],
                diagnostic_checks=[
                    "Check if the error relates to indexed FastAPI documentation.",
                    "Verify the error against official documentation or logs for the relevant framework."
                ],
                suggested_fixes=[],
                sources=[],
                total_sources=0,
                uncertainty_notes=[
                    "No relevant documentation excerpts were retrieved above the similarity threshold."
                ],
                model=self.llm.model
            )

        # 5. Build bounded context
        context_text, citations = build_context(results, max_chars=settings.RAG_CONTEXT_MAX_CHARS)
        if not context_text:
            return DebugResponse(
                error_summary="The provided error does not match any indexed documentation topics.",
                likely_causes=[],
                diagnostic_checks=[],
                suggested_fixes=[],
                sources=[],
                total_sources=0,
                uncertainty_notes=["Retrieved documentation context was empty."],
                model=self.llm.model
            )

        # 6. Construct prompt
        user_prompt = format_debug_user_prompt(
            clean_error,
            request.framework,
            request.language,
            clean_context,
            context_text
        )
        messages = [
            {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # 7. Call LLM (Attempt 1)
        logger.info(f"Calling Groq LLM model '{self.llm.model}' for structured diagnosis.")
        raw_response = self.llm.generate(messages)

        parsed_data: Optional[Dict[str, Any]] = None
        validated_payload: Optional[RawDebugPayload] = None
        validation: Optional[CitationValidationResult] = None

        try:
            parsed_data = parse_debug_json(raw_response)
            validated_payload = RawDebugPayload.model_validate(parsed_data)
            validation = validate_debug_response_citations(parsed_data, raw_response, citations)
        except Exception as e:
            logger.warning(f"Initial debug JSON parsing, schema, or citation validation failed: {e}")
            validation = CitationValidationResult(
                is_valid=False,
                is_refusal=False,
                cited_source_ids=[],
                invalid_source_ids=[],
                fabricated_urls=[],
                error_message=f"JSON parsing or structure error: {str(e)}",
                sanitized_answer=raw_response
            )

        # 8. Single corrective retry if validation failed
        if not validation.is_valid or not validated_payload:
            logger.warning(
                f"Debug citation validation failed on first attempt: {validation.error_message}. "
                "Attempting single corrective retry."
            )
            valid_ids_str = ", ".join([f"[{c.source_id}]" for c in citations])
            corrective_instruction = (
                f"Your previous response had a formatting or citation validation issue: {validation.error_message}\n"
                f"The ONLY valid source identifiers for this request are: {valid_ids_str}.\n"
                "Please output strictly valid JSON matching the schema.\n"
                "Ensure every cause and suggested fix cites ONLY the valid source identifiers above. "
                "Do not include any fabricated URLs or unlisted source identifiers."
            )
            retry_messages = [
                {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": raw_response},
                {"role": "user", "content": corrective_instruction}
            ]
            try:
                retry_raw = self.llm.generate(retry_messages)
                parsed_data = parse_debug_json(retry_raw)
                validated_payload = RawDebugPayload.model_validate(parsed_data)
                validation = validate_debug_response_citations(parsed_data, retry_raw, citations)
            except (GroqProviderError, GroqTimeoutError, GroqConfigurationError):
                raise
            except Exception as e:
                logger.error(f"Corrective retry parsing, schema, or validation failed: {e}")
                parsed_data = None
                validated_payload = None
                validation = CitationValidationResult(
                    is_valid=False,
                    is_refusal=False,
                    cited_source_ids=[],
                    invalid_source_ids=[],
                    fabricated_urls=[],
                    error_message=str(e),
                    sanitized_answer=""
                )

        # 9. Fallback if validation still failed after retry
        if not validation.is_valid or not validated_payload:
            logger.warning(
                f"Debug diagnosis failed citation validation after retry: {validation.error_message}. "
                "Returning controlled insufficient-evidence diagnosis."
            )
            return DebugResponse(
                error_summary="Insufficient documentation evidence to safely diagnose this error.",
                likely_causes=[
                    LikelyCause(
                        cause="Insufficient Documentation Evidence",
                        explanation=(
                            "The available documentation does not contain enough verified evidence "
                            "to diagnose this error without risking speculation."
                        ),
                        citations=[]
                    )
                ],
                diagnostic_checks=[
                    "Check the stack trace for relevant function or class names.",
                    "Review official documentation for the specific library reporting the error."
                ],
                suggested_fixes=[],
                sources=[],
                total_sources=0,
                uncertainty_notes=[
                    f"Generated diagnosis could not be verified against retrieved documentation: {validation.error_message}"
                ],
                model=self.llm.model
            )

        # 10. Construct structured response and filter sources to only cited ones
        likely_causes_list: List[LikelyCause] = []
        for cause in validated_payload.likely_causes:
            raw_cits = cause.citations
            clean_cits = [re.sub(r'[\[\]]', '', str(c)).strip().upper() for c in raw_cits if c]
            likely_causes_list.append(
                LikelyCause(
                    cause=cause.cause,
                    explanation=cause.explanation,
                    citations=clean_cits
                )
            )

        suggested_fixes_list: List[SuggestedFix] = []
        for fix in validated_payload.suggested_fixes:
            raw_cits = fix.citations
            clean_cits = [re.sub(r'[\[\]]', '', str(c)).strip().upper() for c in raw_cits if c]
            suggested_fixes_list.append(
                SuggestedFix(
                    fix=fix.fix,
                    explanation=fix.explanation,
                    citations=clean_cits
                )
            )

        diagnostic_checks = validated_payload.diagnostic_checks
        uncertainty_notes = validated_payload.uncertainty_notes
        error_summary = validated_payload.error_summary

        # Filter sources to only those cited in the validated response
        cited_set = set(validation.cited_source_ids)
        final_sources = [c for c in citations if c.source_id in cited_set]

        return DebugResponse(
            error_summary=error_summary,
            likely_causes=likely_causes_list,
            diagnostic_checks=diagnostic_checks,
            suggested_fixes=suggested_fixes_list,
            sources=final_sources,
            total_sources=len(final_sources),
            uncertainty_notes=uncertainty_notes,
            model=self.llm.model
        )


debug_service = DebugService()
