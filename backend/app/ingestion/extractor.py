from typing import Optional, List
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

def find_content_container(soup: BeautifulSoup, technology: str, strategy: Optional[str] = None) -> Optional[Tag]:
    """Finds the main content element according to technology-specific strategy."""
    strat = (strategy or technology or "").lower()

    if "nodejs" in strat or "node" in strat:
        container = soup.find("div", id="apicontent") or soup.find("div", role="main")
        if container:
            return container

    if "spring" in strat:
        container = (
            soup.find("article", class_="doc")
            or soup.find("div", class_="doc")
            or soup.find("article", class_="rm-Article")
            or soup.find("div", class_="rm-Article")
            or soup.find("div", id="content")
            or soup.find("main")
            or soup.find("article")
        )
        if container:
            return container

    if "django" in strat:
        container = soup.find("div", id="docs-content") or soup.find("div", role="main") or soup.find("article")
        if container:
            return container

    if "flask" in strat:
        container = soup.find("div", role="main") or soup.find("section", class_="body") or soup.find("section")
        if container:
            return container

    if "fastapi" in strat or "pydantic" in strat:
        container = soup.find("article", class_="md-content__inner")
        if container:
            return container

    if "go" in strat or "golang" in strat:
        container = soup.find("article") or soup.find("div", class_="Article")
        if container:
            return container

    if "react" in strat:
        container = soup.find("article")
        if container:
            return container

    # General fallback chain across any technical documentation site
    return (
        soup.find("article", class_="md-content__inner")
        or soup.find("div", id="apicontent")
        or soup.find("article", class_="doc")
        or soup.find("article", class_="rm-Article")
        or soup.find("div", id="docs-content")
        or soup.find("article")
        or soup.find("main")
        or soup.find("div", role="main")
        or soup.find("section")
        or soup.body
    )

def extract_document(
    html_content: str,
    source_url: str,
    technology: str = "fastapi",
    doc_version: Optional[str] = None,
    strategy: Optional[str] = None,
    ecosystem_component: Optional[str] = None
) -> DocumentRecord:
    """
    Parses HTML and extracts the main documentation content across supported technologies.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"
    
    # 2. Canonical URL
    canonical_link = soup.find("link", rel="canonical")
    canonical_url = canonical_link["href"] if canonical_link and canonical_link.has_attr("href") else None
    
    # 3. Identify main content container
    main_content = find_content_container(soup, technology, strategy)
    if not main_content:
        raise ValueError("Could not find a body or main content container in the HTML.")
        
    # 4. Remove irrelevant elements
    for element in main_content.find_all(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        element.decompose()
        
    # Remove table of contents and other non-article sidebars if they exist
    for element in main_content.find_all(class_=["md-sidebar", "toc", "sidebar", "menu", "navbar"]):
        element.decompose()
        
    # Remove headerlink anchors which cause trailing artifact text (e.g., '¶', 'A ')
    for element in main_content.find_all("a", class_=["headerlink", "anchor", "header-anchor"]):
        element.decompose()
        
    # 5. Extract headings
    headings = []
    for heading in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        heading_text = heading.get_text(strip=True)
        if heading_text:
            headings.append(heading_text)
            
    # 6. Extract and clean text
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
        text=cleaned_text,
        technology=technology,
        doc_version=doc_version,
        ecosystem_component=ecosystem_component
    )
