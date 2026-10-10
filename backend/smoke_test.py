
import sys
from app.retrieval.service import retrieval_service
try:
    results = retrieval_service.retrieve('How do I define a POST endpoint in FastAPI?', top_k=3)
    print(f'Retrieved {len(results)} results.')
    for i, r in enumerate(results):
        print(f'Result {i+1}: Score: {r.score:.4f}, Chunk: {r.chunk_id}, Title: {r.section_title}')
        print(f'Text preview: {r.text[:60]}...')
except Exception as e:
    print(f'Error: {e}')
