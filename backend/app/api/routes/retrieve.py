from fastapi import APIRouter, HTTPException
from app.retrieval.models import RetrievalRequest, RetrievalResponse
from app.retrieval.service import retrieval_service, IndexUnavailableError

router = APIRouter()

@router.post("/", response_model=RetrievalResponse)
def retrieve_documents(request: RetrievalRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty or whitespace only.")
        
    try:
        if request.min_score is not None:
            results = retrieval_service.retrieve(request.query, request.top_k, min_score=request.min_score)
        else:
            results = retrieval_service.retrieve(request.query, request.top_k)
        return RetrievalResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
    except IndexUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
