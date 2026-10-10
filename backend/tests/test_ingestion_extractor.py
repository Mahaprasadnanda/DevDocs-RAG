import pytest
from app.ingestion.extractor import extract_document, generate_document_id

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>First Steps - FastAPI</title>
    <link rel="canonical" href="https://fastapi.tiangolo.com/tutorial/first-steps/">
</head>
<body>
    <header>Site Header</header>
    <nav class="md-sidebar">Sidebar content</nav>
    <article class="md-content__inner">
        <h1>First Steps</h1>
        <p>Here are the first steps to use FastAPI.</p>
        <h2>Create it</h2>
        <p>Create a file <code>main.py</code> with:</p>
        <pre><code>from fastapi import FastAPI</code></pre>
        <script>alert("test");</script>
        <style>.test { color: red; }</style>
    </article>
    <footer>Site Footer</footer>
</body>
</html>
"""

NO_ARTICLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>No Article</title></head>
<body>
    <main>
        <h1>Main fallback</h1>
        <p>Fallback content</p>
    </main>
</body>
</html>
"""

EMPTY_BODY_HTML = """
<!DOCTYPE html>
<html>
<head><title>Empty</title></head>
<body>
</body>
</html>
"""

def test_extract_document_success():
    doc = extract_document(SAMPLE_HTML, "https://fastapi.tiangolo.com/tutorial/first-steps/")
    
    assert doc.title == "First Steps - FastAPI"
    assert str(doc.canonical_url) == "https://fastapi.tiangolo.com/tutorial/first-steps/"
    assert str(doc.source_url) == "https://fastapi.tiangolo.com/tutorial/first-steps/"
    assert doc.headings == ["First Steps", "Create it"]
    
    # Check text content and cleaning
    assert "First Steps" in doc.text
    assert "Here are the first steps" in doc.text
    assert "main.py" in doc.text
    assert "from fastapi import FastAPI" in doc.text
    
    # Check exclusion
    assert "Site Header" not in doc.text
    assert "Site Footer" not in doc.text
    assert "Sidebar content" not in doc.text
    assert "alert" not in doc.text
    assert ".test" not in doc.text

def test_extract_document_fallback_main():
    doc = extract_document(NO_ARTICLE_HTML, "https://fastapi.tiangolo.com/fallback")
    assert doc.title == "No Article"
    assert doc.headings == ["Main fallback"]
    assert "Fallback content" in doc.text

def test_extract_document_empty():
    with pytest.raises(ValueError, match="yielded no extractable text"):
        extract_document(EMPTY_BODY_HTML, "https://fastapi.tiangolo.com/empty")

def test_generate_document_id():
    id1 = generate_document_id("https://fastapi.tiangolo.com/path/")
    id2 = generate_document_id("https://fastapi.tiangolo.com/path")
    assert id1 == id2
    assert len(id1) == 64  # sha256 hex length
