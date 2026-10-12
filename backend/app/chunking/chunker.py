import hashlib
from typing import List, Optional
from app.chunking.models import ChunkRecord
from app.ingestion.models import DocumentRecord

def generate_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Generate a stable chunk ID from document ID, index, and content."""
    payload = f"{document_id}:{chunk_index}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def find_split_point(text: str, max_size: int) -> int:
    """
    Finds the best index to split the text, prioritizing:
    1. Double newline (paragraph)
    2. Single newline
    3. Period/Sentence end
    4. Space
    5. Hard split if no boundaries found
    """
    if len(text) <= max_size:
        return len(text)

    # Look for boundaries within the max_size window (backwards)
    window = text[:max_size]
    
    idx = window.rfind('\n\n')
    if idx != -1: return idx + 2
    
    idx = window.rfind('\n')
    if idx != -1: return idx + 1
    
    idx = window.rfind('. ')
    if idx != -1: return idx + 2
    
    idx = window.rfind(' ')
    if idx != -1: return idx + 1
    
    # Priority 5: Hard split
    return max_size

def find_overlap_start(text: str, current_end: int, overlap_size: int) -> int:
    """
    Finds a safe starting index for the next chunk, going backwards
    from current_end by approx overlap_size characters, finding a natural boundary.
    """
    if current_end <= overlap_size:
        return 0
        
    target_start = current_end - overlap_size
    window = text[target_start:current_end]
    
    idx = window.find('\n\n')
    if idx != -1 and target_start + idx + 2 < current_end: return target_start + idx + 2
    
    idx = window.find('\n')
    if idx != -1 and target_start + idx + 1 < current_end: return target_start + idx + 1
    
    idx = window.find('. ')
    if idx != -1 and target_start + idx + 2 < current_end: return target_start + idx + 2
    
    idx = window.find(' ')
    if idx != -1 and target_start + idx + 1 < current_end: return target_start + idx + 1
    
    # Hard fallback
    return target_start


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        
    if not text or not text.strip():
        return []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        remaining = text[start:]
        if not remaining.strip():
            break
            
        if len(remaining) <= chunk_size:
            chunks.append(remaining.strip())
            break
            
        split_point = find_split_point(remaining, chunk_size)
        chunk_text = remaining[:split_point].strip()
        
        if chunk_text:
            chunks.append(chunk_text)
            
        # Failsafe for forward progress
        if split_point == 0:
            split_point = chunk_size
            
        actual_end = start + split_point
        
        if actual_end >= text_len:
            break
            
        next_start = find_overlap_start(text, actual_end, chunk_overlap)
        
        # Ensure we always move forward to avoid infinite loops
        if next_start <= start:
            next_start = start + 1
            
        start = next_start
            
    return chunks

def process_document(doc: DocumentRecord, chunk_size: int, chunk_overlap: int) -> List[ChunkRecord]:
    """Converts a DocumentRecord into overlapping ChunkRecords."""
    raw_chunks = chunk_text(doc.text, chunk_size, chunk_overlap)
    
    records = []
    current_section = doc.headings[0] if doc.headings else None
    
    for i, c_text in enumerate(raw_chunks):
        # Update current section if a known heading appears in this chunk's text
        for h in doc.headings:
            if h in c_text:
                current_section = h
        
        chunk_id = generate_chunk_id(doc.document_id, i, c_text)
        
        record = ChunkRecord(
            chunk_id=chunk_id,
            document_id=doc.document_id,
            title=doc.title,
            source_url=doc.source_url,
            canonical_url=doc.canonical_url,
            section_heading=current_section,
            chunk_index=i,
            text=c_text,
            technology=getattr(doc, "technology", "fastapi"),
            doc_version=getattr(doc, "doc_version", None)
        )
        records.append(record)
        
    return records
