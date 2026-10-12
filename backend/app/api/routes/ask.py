import logging
from fastapi import APIRouter, HTTPException
from app.rag.models import AskRequest, AskResponse
from app.rag.service import rag_service
from app.retrieval.service import IndexUnavailableError
from app.llm.client import GroqConfigurationError, GroqTimeoutError, GroqProviderError

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("", response_model=AskResponse)
@router.post("/", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Accepts a technical documentation question, retrieves semantically relevant context chunks,
    and returns a grounded answer with citations from Groq LLM.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty or whitespace only."
        )

    try:
        if request.technology is not None or request.version is not None:
            response = rag_service.answer(
                query=request.query,
                top_k=request.top_k,
                min_score=request.min_score,
                technology=request.technology,
                version=request.version,
            )
        elif request.min_score is not None:
            response = rag_service.answer(
                query=request.query,
                top_k=request.top_k,
                min_score=request.min_score,
            )
        else:
            response = rag_service.answer(
                query=request.query,
                top_k=request.top_k,
            )
        return response
    except IndexUnavailableError as e:
        logger.error(f"Vector index unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Search index unavailable: {str(e)}"
        )
    except GroqConfigurationError as e:
        logger.error(f"Groq configuration error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service is not configured: {str(e)}"
        )
    except GroqTimeoutError as e:
        logger.error(f"Groq API timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="Upstream LLM provider request timed out. Please try again."
        )
    except GroqProviderError as e:
        logger.error(f"Groq API provider error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM provider error: {str(e)}"
        )
    except ValueError as e:
        logger.warning(f"Validation error in ask endpoint: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in ask endpoint: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal server error occurred while generating the answer."
        )
