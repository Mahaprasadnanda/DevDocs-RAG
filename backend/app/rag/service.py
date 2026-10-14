import logging
from typing import Optional
from app.core.config import settings
from app.retrieval.service import retrieval_service, RetrievalService
from app.llm.client import (
    groq_client,
    GroqLLMClient,
    GroqProviderError,
    GroqTimeoutError,
    GroqConfigurationError,
)
from app.rag.models import AskResponse
from app.rag.context import build_context
from app.rag.prompts import SYSTEM_PROMPT, format_user_prompt
from app.rag.validator import validate_citations, CitationValidationResult

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(
        self,
        retrieval_svc: Optional[RetrievalService] = None,
        llm: Optional[GroqLLMClient] = None
    ):
        self.retrieval_svc = retrieval_svc or retrieval_service
        self.llm = llm or groq_client

    def answer(
        self,
        query: str,
        top_k: int = 8,
        min_score: Optional[float] = None,
        technology: Optional[str] = None,
        version: Optional[str] = None
    ) -> AskResponse:
        """
        Orchestrate retrieval, grounded answer generation, and citation validation.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        clean_query = query.strip()
        logger.info(f"Retrieving context for query: '{clean_query[:50]}...' with top_k={top_k}, tech={technology}")

        # 1. Retrieve top-k chunks (with relevance thresholding and technology filtering)
        if technology is not None or version is not None:
            results = self.retrieval_svc.retrieve(
                clean_query,
                top_k=top_k,
                min_score=min_score,
                technology=technology,
                version=version,
            )
        else:
            results = self.retrieval_svc.retrieve(
                clean_query,
                top_k=top_k,
                min_score=min_score,
            )

        detected_tech = None
        if technology and technology.strip().lower() not in ["auto", "all"]:
            detected_tech = technology.strip()
        elif results and results[0].technology:
            detected_tech = results[0].technology

        # 2. Handle empty retrieval result (out-of-domain or below relevance threshold)
        if not results:
            logger.info("No matching chunks retrieved above relevance threshold.")
            return AskResponse(
                query=clean_query,
                answer="The indexed documentation does not contain enough relevant information to answer this question.",
                sources=[],
                total_sources=0,
                model=self.llm.model,
                detected_technology=detected_tech
            )

        # 3. Construct bounded context and structured citations for THIS request
        context_text, citations = build_context(results, max_chars=settings.RAG_CONTEXT_MAX_CHARS)

        if not context_text:
            return AskResponse(
                query=clean_query,
                answer="The indexed documentation does not contain enough relevant information to answer this question.",
                sources=[],
                total_sources=0,
                model=self.llm.model,
                detected_technology=detected_tech
            )

        # 4. Construct grounded prompt
        user_prompt = format_user_prompt(clean_query, context_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # 5. Call Groq LLM (Attempt 1)
        logger.info(f"Calling Groq LLM model '{self.llm.model}' for grounded completion.")
        raw_answer = self.llm.generate(messages)

        # 6. Validate citations against retrieved sources allowlist
        validation = validate_citations(raw_answer, citations)

        # 7. If validation fails, attempt exactly ONE deterministic corrective retry
        if not validation.is_valid:
            logger.warning(
                f"Citation validation failed on first attempt: {validation.error_message}. "
                "Attempting single corrective retry."
            )
            valid_ids_str = ", ".join([f"[{c.source_id}]" for c in citations])
            if not validation.cited_source_ids:
                corrective_instruction = (
                    f"Your previous response had a citation issue: {validation.error_message}\n"
                    f"The valid source identifiers for this question are: {valid_ids_str}.\n"
                    "Please rewrite the answer, keeping your explanations but adding inline bracketed citations [S1], [S2] to the supporting sources.\n"
                    "Do not include any fabricated URLs or unlisted source identifiers."
                )
            else:
                corrective_instruction = (
                    f"Your previous response had a citation issue: {validation.error_message}\n"
                    f"The ONLY valid source identifiers for this question are: {valid_ids_str}.\n"
                    "Please rewrite the answer ensuring:\n"
                    "1. Every factual statement cites ONLY the valid source identifiers above using [S1], [S2], etc.\n"
                    "2. Do not include any fabricated URLs, external links, or unlisted source identifiers.\n"
                    "3. If the provided sources do not contain enough information to support an answer, state: "
                    "'The provided documentation does not contain enough information to answer this question.'"
                )
            retry_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": raw_answer},
                {"role": "user", "content": corrective_instruction}
            ]
            try:
                retry_answer = self.llm.generate(retry_messages)
                validation = validate_citations(retry_answer, citations)
            except (GroqProviderError, GroqTimeoutError, GroqConfigurationError):
                raise
            except Exception as e:
                logger.error(f"Unexpected error during corrective retry: {e}")
                validation = CitationValidationResult(
                    is_valid=False,
                    is_refusal=False,
                    cited_source_ids=[],
                    invalid_source_ids=[],
                    fabricated_urls=[],
                    error_message=str(e),
                    sanitized_answer=""
                )

        # 8. If still invalid after retry, return controlled insufficient-evidence fallback
        if not validation.is_valid:
            logger.warning(
                f"Answer failed citation validation after retry: {validation.error_message}. "
                "Returning controlled insufficient-evidence response."
            )
            return AskResponse(
                query=clean_query,
                answer="The indexed documentation does not contain enough relevant information to answer this question.",
                sources=[],
                total_sources=0,
                model=self.llm.model,
                detected_technology=detected_tech
            )

        # 9. Handle explicit refusal response
        if validation.is_refusal:
            logger.info("Model returned valid refusal due to insufficient context.")
            return AskResponse(
                query=clean_query,
                answer=validation.sanitized_answer,
                sources=[],
                total_sources=0,
                model=self.llm.model,
                detected_technology=detected_tech
            )

        # 10. Filter returned sources to ONLY those actually cited in the verified answer
        cited_set = set(validation.cited_source_ids)
        final_sources = [c for c in citations if c.source_id in cited_set]

        return AskResponse(
            query=clean_query,
            answer=validation.sanitized_answer,
            sources=final_sources,
            total_sources=len(final_sources),
            model=self.llm.model,
            detected_technology=detected_tech
        )

rag_service = RAGService()
