import logging
from typing import Optional
from app.core.config import settings
from app.retrieval.service import retrieval_service, RetrievalService
from app.llm.client import groq_client, GroqLLMClient
from app.rag.models import AskResponse
from app.rag.context import build_context
from app.rag.prompts import SYSTEM_PROMPT, format_user_prompt

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(
        self,
        retrieval_svc: Optional[RetrievalService] = None,
        llm: Optional[GroqLLMClient] = None
    ):
        self.retrieval_svc = retrieval_svc or retrieval_service
        self.llm = llm or groq_client

    def answer(self, query: str, top_k: int = 8) -> AskResponse:
        """
        Orchestrate retrieval and grounded answer generation.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        clean_query = query.strip()
        logger.info(f"Retrieving context for query: '{clean_query[:50]}...' with top_k={top_k}")

        # 1. Retrieve top-k chunks
        results = self.retrieval_svc.retrieve(clean_query, top_k=top_k)

        # 2. Handle empty retrieval result
        if not results:
            logger.info("No matching chunks retrieved from index.")
            return AskResponse(
                query=clean_query,
                answer="The indexed documentation does not contain enough relevant information to answer this question.",
                sources=[],
                total_sources=0,
                model=self.llm.model
            )

        # 3. Construct bounded context and structured citations
        context_text, citations = build_context(results, max_chars=settings.RAG_CONTEXT_MAX_CHARS)

        if not context_text:
            return AskResponse(
                query=clean_query,
                answer="The indexed documentation does not contain enough relevant information to answer this question.",
                sources=[],
                total_sources=0,
                model=self.llm.model
            )

        # 4. Construct grounded prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(clean_query, context_text)}
        ]

        # 5. Call Groq LLM
        logger.info(f"Calling Groq LLM model '{self.llm.model}' for grounded completion.")
        answer_text = self.llm.generate(messages)

        return AskResponse(
            query=clean_query,
            answer=answer_text,
            sources=citations,
            total_sources=len(citations),
            model=self.llm.model
        )

rag_service = RAGService()
