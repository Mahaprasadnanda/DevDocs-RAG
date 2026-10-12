import logging
from fastapi import APIRouter, HTTPException
from app.debug.models import DebugRequest, DebugResponse
from app.debug.service import debug_service
from app.retrieval.service import IndexUnavailableError
from app.llm.client import GroqConfigurationError, GroqTimeoutError, GroqProviderError

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("", response_model=DebugResponse)
@router.post("/", response_model=DebugResponse)
def debug_error(request: DebugRequest):
    """
    Accepts an error message or stack trace, retrieves semantically relevant documentation chunks,
    and returns a structured diagnosis with verified citations.
    """
    if not request.error_message or not request.error_message.strip():
        raise HTTPException(
            status_code=422,
            detail="Error message cannot be empty or whitespace only."
        )

    try:
        response = debug_service.diagnose(request)
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
        logger.warning(f"Validation error in debug endpoint: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in debug endpoint: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal server error occurred while diagnosing the error."
        )
