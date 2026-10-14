import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.ingestion.registry import (
    DOCUMENTATION_REGISTRY,
    get_technology_config,
    get_all_technologies,
    validate_url_against_registry,
)
from app.ingestion.extractor import extract_document
from app.chunking.chunker import process_document
from app.chunking.models import ChunkRecord
from app.retrieval.tech_detector import detect_technology, TECH_PATTERNS
from app.retrieval.models import RetrievalResult
from app.retrieval.service import RetrievalService
from app.rag.service import RAGService
from app.debug.service import DebugService
from app.debug.models import DebugRequest

client = TestClient(app)

# ---------------------------------------------------------------------------
# 1. Official Documentation Registry Tests
# ---------------------------------------------------------------------------

def test_registry_contains_eight_technologies():
    expected_techs = {"fastapi", "react", "django", "flask", "pydantic", "spring-boot", "nodejs", "go"}
    registered_keys = set(DOCUMENTATION_REGISTRY.keys())
    assert expected_techs.issubset(registered_keys)
    assert len(registered_keys) >= 8

def test_registry_configs_have_required_fields():
    for tech_id, cfg in DOCUMENTATION_REGISTRY.items():
        assert cfg.tech_id == tech_id
        assert cfg.display_name
        assert cfg.root_url.startswith("https://")
        assert cfg.version
        assert len(cfg.allowed_hosts) > 0
        assert len(cfg.urls) > 0
        assert cfg.extraction_strategy

def test_validate_url_against_registry():
    assert validate_url_against_registry("https://react.dev/reference/react/useState", tech_id="react") is True
    assert validate_url_against_registry("https://docs.spring.io/spring-boot/reference/web/servlet.html", tech_id="spring-boot") is True
    assert validate_url_against_registry("https://go.dev/doc/effective_go", tech_id="go") is True
    assert validate_url_against_registry("https://nodejs.org/docs/latest/api/http.html", tech_id="nodejs") is True
    assert validate_url_against_registry("https://docs.spring.io/spring-boot/reference/web/servlet.html") is True
    assert validate_url_against_registry("https://malicious-site.com/steal-data") is False

def test_get_all_technologies():
    techs = get_all_technologies(enabled_only=True)
    assert len(techs) >= 8
    tech_ids = [t.tech_id for t in techs]
    assert "spring-boot" in tech_ids
    assert "nodejs" in tech_ids
    assert "go" in tech_ids

# ---------------------------------------------------------------------------
# 2. Technology Extraction Strategy Tests
# ---------------------------------------------------------------------------

def test_extract_document_react_strategy():
    html = """
    <html>
      <head><title>State: A Component's Memory – React</title></head>
      <body>
        <nav><a href="/">Home</a></nav>
        <article>
          <h1>State: A Component's Memory</h1>
          <p>Components often need to change what's on the screen as a result of an interaction.</p>
        </article>
      </body>
    </html>
    """
    doc = extract_document(html, "https://react.dev/learn/state-a-components-memory", technology="react", doc_version="19.0", strategy="react_dev")
    assert doc.technology == "react"
    assert doc.doc_version == "19.0"
    assert "Components often need to change" in doc.text
    assert "Home" not in doc.text

def test_extract_document_spring_strategy():
    html = """
    <html>
      <head><title>Spring Beans and Dependency Injection</title></head>
      <body>
        <div class="doc">
          <h1>Spring Beans and Dependency Injection</h1>
          <p>You are free to use any of the standard Spring Framework techniques to define your beans and their injected dependencies.</p>
        </div>
      </body>
    </html>
    """
    doc = extract_document(html, "https://docs.spring.io/spring-boot/reference/using/spring-beans-and-dependency-injection.html", technology="spring-boot", doc_version="3.4", strategy="spring_docs")
    assert doc.technology == "spring-boot"
    assert doc.doc_version == "3.4"
    assert "standard Spring Framework techniques" in doc.text

def test_extract_document_nodejs_strategy():
    html = """
    <html>
      <head><title>HTTP | Node.js v22.x Manual &amp; Documentation</title></head>
      <body>
        <header>Node.js Nav</header>
        <article id="apicontent">
          <h1>HTTP</h1>
          <p>To use the HTTP server and client one must require('node:http').</p>
        </article>
      </body>
    </html>
    """
    doc = extract_document(html, "https://nodejs.org/docs/latest/api/http.html", technology="nodejs", doc_version="22.x", strategy="nodejs_api")
    assert doc.technology == "nodejs"
    assert doc.doc_version == "22.x"
    assert "require('node:http')" in doc.text
    assert "Node.js Nav" not in doc.text

def test_extract_document_go_strategy():
    html = """
    <html>
      <head><title>Effective Go - The Go Programming Language</title></head>
      <body>
        <nav>Go navigation</nav>
        <article class="Doc">
          <h1>Effective Go</h1>
          <p>Go is a new language. Although it borrows ideas from existing languages, it has unusual properties.</p>
        </article>
      </body>
    </html>
    """
    doc = extract_document(html, "https://go.dev/doc/effective_go", technology="go", doc_version="1.24", strategy="go_pkg")
    assert doc.technology == "go"
    assert doc.doc_version == "1.24"
    assert "unusual properties" in doc.text
    assert "Go navigation" not in doc.text

# ---------------------------------------------------------------------------
# 3. Metadata Preservation in Chunking
# ---------------------------------------------------------------------------

def test_chunking_preserves_technology_and_version():
    html = """
    <html>
      <head><title>Go Concurrency</title></head>
      <body>
        <main>
          <h1>Goroutines and Channels</h1>
          <p>Goroutines run concurrently in the same address space. Channels are typed conduits through which you can send and receive values.</p>
        </main>
      </body>
    </html>
    """
    doc = extract_document(html, "https://go.dev/doc/effective_go", technology="go", doc_version="1.24")
    chunks = process_document(doc, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.technology == "go"
        assert c.doc_version == "1.24"
        assert c.document_id == doc.document_id

# ---------------------------------------------------------------------------
# 4. Technology Detection
# ---------------------------------------------------------------------------

def test_detect_technology_single_technologies():
    assert detect_technology("How do I use useState and useEffect?") == ["react"]
    assert detect_technology("Configuring @SpringBootApplication and application.properties") == ["spring-boot"]
    assert detect_technology("How to start goroutines and read from channel in Go?") == ["go"]
    assert detect_technology("Using require('node:fs') and promises in Node.js") == ["nodejs"]
    assert detect_technology("Defining ForeignKey relations in Django models.py") == ["django"]
    assert detect_technology("Creating a Flask Blueprint with render_template") == ["flask"]
    assert detect_technology("Field validation with Pydantic BaseModel and Field") == ["pydantic"]
    assert detect_technology("FastAPI APIRouter and Depends dependency injection") == ["fastapi"]

def test_detect_technology_cross_technology():
    res = detect_technology("Connecting a React frontend with a Spring Boot REST API")
    assert "react" in res
    assert "spring-boot" in res

def test_detect_technology_unrecognized():
    assert detect_technology("What is the speed of light in vacuum?") == []

# ---------------------------------------------------------------------------
# 5. Technology-Aware Filtered Retrieval
# ---------------------------------------------------------------------------

def test_retrieval_filtering_by_technology():
    mock_store = MagicMock()
    mock_store.index.ntotal = 4
    
    mock_metadata = [
        ChunkRecord(
            chunk_id="chunk_react",
            document_id="doc_react",
            title="React useState",
            section_heading="useState",
            text="useState is a React Hook that lets you add a state variable to your component.",
            source_url="https://react.dev/reference/react/useState",
            start_char=0,
            end_char=80,
            char_count=80,
            chunk_index=0,
            technology="react",
            doc_version="19.0"
        ),
        ChunkRecord(
            chunk_id="chunk_spring",
            document_id="doc_spring",
            title="Spring Boot Beans",
            section_heading="Spring Beans",
            text="Use @Autowired to inject dependencies in Spring Boot components.",
            source_url="https://docs.spring.io/spring-boot/reference/using/spring-beans-and-dependency-injection.html",
            start_char=0,
            end_char=75,
            char_count=75,
            chunk_index=0,
            technology="spring-boot",
            doc_version="3.4"
        ),
        ChunkRecord(
            chunk_id="chunk_go",
            document_id="doc_go",
            title="Go Goroutines",
            section_heading="Goroutines",
            text="A goroutine has a simple model: it is a function executing concurrently with other goroutines.",
            source_url="https://go.dev/doc/effective_go",
            start_char=0,
            end_char=95,
            char_count=95,
            chunk_index=0,
            technology="go",
            doc_version="1.24"
        ),
        ChunkRecord(
            chunk_id="chunk_node",
            document_id="doc_node",
            title="Node HTTP Server",
            section_heading="HTTP",
            text="http.createServer creates an HTTP server instance in Node.js.",
            source_url="https://nodejs.org/docs/latest/api/http.html",
            start_char=0,
            end_char=60,
            char_count=60,
            chunk_index=0,
            technology="nodejs",
            doc_version="22.x"
        ),
    ]
    mock_store.metadata = mock_metadata
    # Mock FAISS search returning all 4 with high scores
    mock_store.index.search.return_value = (
        [[0.95, 0.90, 0.85, 0.80]],
        [[0, 1, 2, 3]]
    )

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [[0.1] * 384]

    service = RetrievalService()
    service.store = mock_store
    service.embedder = mock_embedder
    service._is_loaded = True

    # 1. Filter explicitly for React
    results = service.retrieve("state management", top_k=5, min_score=0.30, technology="react")
    assert len(results) == 1
    assert results[0].technology == "react"
    assert results[0].chunk_id == "chunk_react"

    # 2. Filter explicitly for Spring Boot
    results = service.retrieve("dependency injection", top_k=5, min_score=0.30, technology="spring-boot")
    assert len(results) == 1
    assert results[0].technology == "spring-boot"
    assert results[0].chunk_id == "chunk_spring"

    # 3. Filter explicitly for Go
    results = service.retrieve("concurrency", top_k=5, min_score=0.30, technology="go")
    assert len(results) == 1
    assert results[0].technology == "go"
    assert results[0].chunk_id == "chunk_go"

    # 4. Filter explicitly for Node.js
    results = service.retrieve("http server", top_k=5, min_score=0.30, technology="nodejs")
    assert len(results) == 1
    assert results[0].technology == "nodejs"
    assert results[0].chunk_id == "chunk_node"

    # 5. Unconstrained retrieval returns all
    results = service.retrieve("general query", top_k=5, min_score=0.30)
    assert len(results) == 4

# ---------------------------------------------------------------------------
# 6. Debug Assistant Multi-Technology Tests
# ---------------------------------------------------------------------------

def test_debug_service_ecosystem_detection():
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = []
    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)

    # Spring Boot error
    req_spring = DebugRequest(
        error_message="org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type 'UserService'",
    )
    resp_spring = service.diagnose(req_spring)
    assert resp_spring.detected_technology == "spring-boot"

    # Go error
    req_go = DebugRequest(
        error_message="panic: runtime error: invalid memory address or nil pointer dereference [signal SIGSEGV]",
    )
    resp_go = service.diagnose(req_go)
    assert resp_go.detected_technology == "go"

    # Node.js error
    req_node = DebugRequest(
        error_message="Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'express' imported from /app/index.js",
    )
    resp_node = service.diagnose(req_node)
    assert resp_node.detected_technology == "nodejs"

    # React error
    req_react = DebugRequest(
        error_message="Error: Invalid hook call. Hooks can only be called inside the body of a function component.",
    )
    resp_react = service.diagnose(req_react)
    assert resp_react.detected_technology == "react"

def test_debug_service_unknown_technology_safe_fallback():
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = []
    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(
        error_message="UnknownSystemError: arbitrary unknown failure in system x",
    )
    resp = service.diagnose(req)
    assert resp.total_sources == 0
    assert any("technology ecosystem could not be identified" in note.lower() for note in resp.uncertainty_notes)
    mock_llm.generate.assert_not_called()

# ---------------------------------------------------------------------------
# 7. Registry Endpoint API Test
# ---------------------------------------------------------------------------

def test_registry_endpoint():
    response = client.get("/api/v1/registry/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 8
    tech_ids = [t["tech_id"] for t in data]
    for expected in ["fastapi", "react", "django", "flask", "pydantic", "spring-boot", "nodejs", "go"]:
        assert expected in tech_ids

# ---------------------------------------------------------------------------
# 8. API Backward Compatibility Tests
# ---------------------------------------------------------------------------

@patch("app.api.routes.retrieve.retrieval_service")
def test_retrieve_api_with_technology(mock_service):
    mock_service.retrieve.return_value = [
        RetrievalResult(
            chunk_id="c_react",
            text="React text",
            source_url="https://react.dev/learn",
            document_id="doc_react",
            section_title="Overview",
            score=0.92,
            technology="react",
            doc_version="19.0"
        )
    ]
    response = client.post("/api/v1/retrieve/", json={
        "query": "hooks",
        "top_k": 3,
        "technology": "react"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["detected_technology"] == "react"
    assert data["results"][0]["technology"] == "react"

@patch("app.api.routes.ask.rag_service")
def test_ask_api_with_technology(mock_rag):
    from app.rag.models import AskResponse, SourceCitation
    mock_rag.answer.return_value = AskResponse(
        query="Explain useState",
        answer="useState lets you declare state in React [S1].",
        sources=[
            SourceCitation(
                source_id="S1",
                chunk_id="chunk_r1",
                document_id="doc_r1",
                section_title="useState",
                source_url="https://react.dev/reference/react/useState",
                score=0.95,
                technology="react",
                doc_version="19.0"
            )
        ],
        total_sources=1,
        model="openai/gpt-oss-120b",
        detected_technology="react"
    )
    response = client.post("/api/v1/ask/", json={
        "query": "Explain useState",
        "technology": "react"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["detected_technology"] == "react"
    assert data["sources"][0]["technology"] == "react"


# ---------------------------------------------------------------------------
# 9. Spring Boot Retrieval Coverage and Ecosystem Component Tests
# ---------------------------------------------------------------------------

def test_get_ecosystem_component_mappings():
    from app.ingestion.registry import get_ecosystem_component, get_technology_config
    cfg = get_technology_config("spring-boot")
    assert cfg is not None
    assert get_ecosystem_component("https://docs.spring.io/spring-boot/reference/web/servlet.html", cfg) == "spring-boot-core"
    assert get_ecosystem_component("https://docs.spring.io/spring-framework/reference/core/beans/dependencies/factory-collaborators.html", cfg) == "spring-framework"
    assert get_ecosystem_component("https://docs.spring.io/spring-cloud-gateway/reference/spring-cloud-gateway-server-webmvc/filters/ratelimiter.html", cfg) == "spring-cloud-gateway"
    assert get_ecosystem_component("https://resilience4j.readme.io/docs/ratelimiter", cfg) == "resilience4j (third-party library)"


def test_build_context_includes_ecosystem_component():
    from app.rag.context import build_context
    result = RetrievalResult(
        chunk_id="chunk_r4j",
        text="Resilience4j provides rate limiting annotations.",
        source_url="https://resilience4j.readme.io/docs/ratelimiter",
        document_id="doc_r4j",
        section_title="RateLimiter Internals",
        score=0.75,
        technology="spring-boot",
        doc_version="3.4",
        ecosystem_component="resilience4j (third-party library)"
    )
    context_text, citations = build_context([result])
    assert "Component: resilience4j (third-party library)" in context_text
    assert len(citations) == 1
    assert citations[0].ecosystem_component == "resilience4j (third-party library)"


def test_spring_boot_retrieval_live_coverage_q1_rest_api():
    from app.retrieval.service import retrieval_service
    q1 = "How do I create a simple REST API in Spring Boot using @RestController and @GetMapping?"
    results = retrieval_service.retrieve(q1, top_k=5, technology="spring-boot")
    assert len(results) > 0
    assert results[0].score >= 0.50
    assert any("RestController" in r.text or "GetMapping" in r.text or "Spring Web" in (r.section_title or "") for r in results)


def test_spring_boot_retrieval_live_coverage_q2_constructor_injection():
    from app.retrieval.service import retrieval_service
    q2 = "How do I inject a service into a Spring Boot REST controller using constructor injection?"
    results = retrieval_service.retrieve(q2, top_k=8, technology="spring-boot")
    assert len(results) > 0
    # Top score is well above threshold
    assert results[0].score >= 0.50
    # Verify constructor injection documentation is retrieved
    has_di_doc = any(
        "constructor" in r.text.lower() or "dependency injection" in r.text.lower()
        for r in results
    )
    assert has_di_doc


def test_spring_boot_retrieval_live_coverage_q3_rate_limiting():
    from app.retrieval.service import retrieval_service
    q3 = "How can I implement rate limiting for a Spring Boot REST API, and what approaches are documented?"
    results = retrieval_service.retrieve(q3, top_k=8, technology="spring-boot")
    assert len(results) > 0
    # Top score is well above 0.50 (previously failed with no rate limiting docs)
    assert results[0].score >= 0.55
    components = {r.ecosystem_component for r in results if r.ecosystem_component}
    # Rate limiting must retrieve Resilience4j or Spring Cloud Gateway
    assert any(c in components for c in ["resilience4j (third-party library)", "spring-cloud-gateway"])
