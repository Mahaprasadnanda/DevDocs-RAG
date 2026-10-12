from typing import List, Tuple
from app.retrieval.models import RetrievalResult
from app.rag.models import SourceCitation

def build_context(
    results: List[RetrievalResult],
    max_chars: int = 4000
) -> Tuple[str, List[SourceCitation]]:
    """
    Build a bounded textual context from retrieval results and map to structured citations.
    Respects max_chars budget.
    """
    if not results:
        return "", []

    context_blocks: List[str] = []
    citations: List[SourceCitation] = []
    current_length = 0

    for idx, r in enumerate(results, start=1):
        source_id = f"S{idx}"
        title = r.section_title or "Documentation Section"
        url_str = str(r.source_url)
        header = f"--- Source [{source_id}] ---\nTitle: {title}\nURL: {url_str}\nContent:\n"
        content = r.text.strip()
        block = header + content + "\n"

        block_len = len(block)
        # Check budget
        if current_length + block_len > max_chars and citations:
            # Reached budget and already have at least 1 citation, stop adding more chunks
            break

        context_blocks.append(block)
        current_length += block_len

        citation = SourceCitation(
            source_id=source_id,
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            section_title=r.section_title,
            source_url=r.source_url,
            score=r.score,
            technology=r.technology,
            doc_version=r.doc_version
        )
        citations.append(citation)

    combined_context = "\n".join(context_blocks).strip()
    return combined_context, citations
