import hashlib
from bs4 import BeautifulSoup, Tag
from pydantic import HttpUrl
from app.ingestion.models import DocumentRecord

def generate_document_id(url: str) -> str:
    """Generate a stable identifier based on the normalized URL."""
    # Normalize by stripping trailing slash
    normalized = url.rstrip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure where possible."""
    lines = [line.strip() for line in text.splitlines()]
    # Join non-empty lines with a single space, or use newlines for gaps
    cleaned = "\n".join(line for line in lines if line)
    return cleaned

def extract_document(html_content: str, source_url: str) -> DocumentRecord:
    """
    Parses HTML and extracts the main documentation content.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"
    
    # 2. Canonical URL
    canonical_link = soup.find("link", rel="canonical")
    canonical_url = canonical_link["href"] if canonical_link and canonical_link.has_attr("href") else None
    
    # 3. Identify main content container
    # For MkDocs Material (which FastAPI uses), the main content is usually in <article class="md-content__inner">
    # Fallback to <main> or <body>
    main_content = soup.find("article", class_="md-content__inner")
    if not main_content:
        main_content = soup.find("main")
    if not main_content:
        main_content = soup.body
        
    if not main_content:
        raise ValueError("Could not find a body or main content container in the HTML.")
        
    # 4. Remove irrelevant elements
    for element in main_content.find_all(["script", "style", "noscript", "nav", "footer", "header"]):
        element.decompose()
        
    # Remove table of contents and other non-article sidebars if they exist
    for element in main_content.find_all(class_=["md-sidebar", "toc"]):
        element.decompose()
        
    # Remove headerlink anchors which cause trailing artifact text (e.g., '¶', 'A ')
    for element in main_content.find_all("a", class_="headerlink"):
        element.decompose()
        
    # 5. Extract headings
    headings = []
    for heading in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        heading_text = heading.get_text(strip=True)
        if heading_text:
            headings.append(heading_text)
            
    # 6. Extract and clean text
    # We use separator='\n' to maintain some structural boundary
    raw_text = main_content.get_text(separator="\n")
    cleaned_text = clean_text(raw_text)
    
    if not cleaned_text:
        raise ValueError("Main content container was found, but yielded no extractable text.")
        
    doc_id = generate_document_id(source_url)
    
    return DocumentRecord(
        document_id=doc_id,
        title=title,
        source_url=source_url,  # type: ignore
        canonical_url=canonical_url, # type: ignore
        headings=headings,
        text=cleaned_text
    )
