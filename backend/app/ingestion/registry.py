from typing import List, Dict, Optional
from pydantic import BaseModel, HttpUrl, Field


class TechnologyConfig(BaseModel):
    tech_id: str = Field(..., description="Stable technology identifier, e.g. 'fastapi', 'react'.")
    display_name: str = Field(..., description="Human-readable display name.")
    root_url: str = Field(..., description="Official documentation root URL.")
    version: str = Field(..., description="Documentation version label, e.g. '0.115', '19.0', 'stable'.")
    enabled: bool = Field(True, description="Whether this technology collection is enabled.")
    allowed_hosts: List[str] = Field(..., description="Allowed HTTP hostnames for redirection and security.")
    urls: List[str] = Field(..., description="Curated documentation URLs to ingest.")
    extraction_strategy: str = Field("auto", description="Source-specific extraction strategy identifier.")
    aliases: List[str] = Field(default_factory=list, description="Alternative names and keywords for auto-detection.")
    component_mappings: Dict[str, str] = Field(default_factory=dict, description="URL pattern to ecosystem component mapping.")


# Official documentation registry for the 8 supported technologies
DOCUMENTATION_REGISTRY: Dict[str, TechnologyConfig] = {
    "fastapi": TechnologyConfig(
        tech_id="fastapi",
        display_name="FastAPI",
        root_url="https://fastapi.tiangolo.com/",
        version="0.115",
        enabled=True,
        allowed_hosts=["fastapi.tiangolo.com"],
        urls=[
            "https://fastapi.tiangolo.com/",
            "https://fastapi.tiangolo.com/tutorial/first-steps/",
            "https://fastapi.tiangolo.com/tutorial/path-params/",
            "https://fastapi.tiangolo.com/tutorial/body/",
            "https://fastapi.tiangolo.com/tutorial/handling-errors/"
        ],
        extraction_strategy="fastapi_mkdocs",
        aliases=["fastapi", "uvicorn", "starlette"]
    ),
    "react": TechnologyConfig(
        tech_id="react",
        display_name="React",
        root_url="https://react.dev/",
        version="19.0",
        enabled=True,
        allowed_hosts=["react.dev"],
        urls=[
            "https://react.dev/learn",
            "https://react.dev/learn/describing-the-ui",
            "https://react.dev/learn/state-a-components-memory",
            "https://react.dev/reference/react/useState",
            "https://react.dev/reference/react/useEffect"
        ],
        extraction_strategy="react_dev",
        aliases=["react", "reactjs", "jsx", "tsx"]
    ),
    "django": TechnologyConfig(
        tech_id="django",
        display_name="Django",
        root_url="https://docs.djangoproject.com/en/stable/",
        version="6.1",
        enabled=True,
        allowed_hosts=["docs.djangoproject.com"],
        urls=[
            "https://docs.djangoproject.com/en/stable/intro/tutorial01/",
            "https://docs.djangoproject.com/en/stable/topics/http/urls/",
            "https://docs.djangoproject.com/en/stable/topics/http/views/",
            "https://docs.djangoproject.com/en/stable/topics/db/models/",
            "https://docs.djangoproject.com/en/stable/topics/forms/"
        ],
        extraction_strategy="django_sphinx",
        aliases=["django", "django-rest-framework", "drf"]
    ),
    "flask": TechnologyConfig(
        tech_id="flask",
        display_name="Flask",
        root_url="https://flask.palletsprojects.com/",
        version="3.1",
        enabled=True,
        allowed_hosts=["flask.palletsprojects.com"],
        urls=[
            "https://flask.palletsprojects.com/en/stable/quickstart/",
            "https://flask.palletsprojects.com/en/stable/errorhandling/",
            "https://flask.palletsprojects.com/en/stable/blueprints/",
            "https://flask.palletsprojects.com/en/stable/views/"
        ],
        extraction_strategy="flask_pallets",
        aliases=["flask", "werkzeug", "jinja", "jinja2"]
    ),
    "pydantic": TechnologyConfig(
        tech_id="pydantic",
        display_name="Pydantic",
        root_url="https://docs.pydantic.dev/latest/",
        version="2.10",
        enabled=True,
        allowed_hosts=["docs.pydantic.dev", "pydantic.dev"],
        urls=[
            "https://docs.pydantic.dev/latest/concepts/models/",
            "https://docs.pydantic.dev/latest/concepts/fields/",
            "https://docs.pydantic.dev/latest/concepts/validators/",
            "https://docs.pydantic.dev/latest/errors/errors/"
        ],
        extraction_strategy="pydantic_mkdocs",
        aliases=["pydantic", "pydantic-core"]
    ),
    "spring-boot": TechnologyConfig(
        tech_id="spring-boot",
        display_name="Spring Boot",
        root_url="https://docs.spring.io/spring-boot/index.html",
        version="3.4",
        enabled=True,
        allowed_hosts=["docs.spring.io", "resilience4j.readme.io"],
        urls=[
            "https://docs.spring.io/spring-boot/index.html",
            "https://docs.spring.io/spring-boot/reference/using/index.html",
            "https://docs.spring.io/spring-boot/reference/using/spring-beans-and-dependency-injection.html",
            "https://docs.spring.io/spring-boot/reference/web/servlet.html",
            "https://docs.spring.io/spring-boot/reference/features/external-config.html",
            "https://docs.spring.io/spring-boot/reference/features/profiles.html",
            "https://docs.spring.io/spring-framework/reference/core/beans/dependencies/factory-collaborators.html",
            "https://docs.spring.io/spring-framework/reference/core/beans/annotation-config/autowired.html",
            "https://docs.spring.io/spring-framework/reference/web/webmvc/mvc-controller/ann-requestmapping.html",
            "https://docs.spring.io/spring-cloud-gateway/reference/spring-cloud-gateway-server-webmvc/filters/ratelimiter.html",
            "https://docs.spring.io/spring-cloud-gateway/reference/spring-cloud-gateway-server-webflux/gatewayfilter-factories/requestratelimiter-factory.html",
            "https://resilience4j.readme.io/docs/ratelimiter",
            "https://resilience4j.readme.io/docs/getting-started-3"
        ],
        extraction_strategy="spring_docs",
        aliases=["spring-boot", "springboot", "spring", "spring-mvc", "spring-framework", "spring-cloud", "resilience4j"],
        component_mappings={
            "spring-boot": "spring-boot-core",
            "spring-framework": "spring-framework",
            "spring-cloud-gateway": "spring-cloud-gateway",
            "resilience4j": "resilience4j (third-party library)"
        }
    ),
    "nodejs": TechnologyConfig(
        tech_id="nodejs",
        display_name="Node.js",
        root_url="https://nodejs.org/docs/latest/api/",
        version="22.x",
        enabled=True,
        allowed_hosts=["nodejs.org"],
        urls=[
            "https://nodejs.org/docs/latest/api/modules.html",
            "https://nodejs.org/docs/latest/api/esm.html",
            "https://nodejs.org/docs/latest/api/http.html",
            "https://nodejs.org/docs/latest/api/fs.html",
            "https://nodejs.org/docs/latest/api/errors.html",
            "https://nodejs.org/docs/latest/api/process.html"
        ],
        extraction_strategy="nodejs_api",
        aliases=["nodejs", "node", "npm", "express"]
    ),
    "go": TechnologyConfig(
        tech_id="go",
        display_name="Go (Golang)",
        root_url="https://go.dev/doc/",
        version="1.24",
        enabled=True,
        allowed_hosts=["go.dev"],
        urls=[
            "https://go.dev/doc/effective_go",
            "https://go.dev/doc/tutorial/getting-started",
            "https://go.dev/doc/tutorial/handle-errors",
            "https://go.dev/doc/articles/wiki/"
        ],
        extraction_strategy="go_pkg",
        aliases=["go", "golang", "goroutine", "gin"]
    )
}


def get_technology_config(tech_id: str) -> Optional[TechnologyConfig]:
    """Retrieve technology configuration by ID or alias (case-insensitive)."""
    norm = tech_id.strip().lower()
    if norm in DOCUMENTATION_REGISTRY:
        return DOCUMENTATION_REGISTRY[norm]
    # Check aliases
    for config in DOCUMENTATION_REGISTRY.values():
        if norm in [a.lower() for a in config.aliases] or norm == config.tech_id:
            return config
    return None


def get_all_technologies(enabled_only: bool = True) -> List[TechnologyConfig]:
    """Return all configured technologies."""
    if enabled_only:
        return [c for c in DOCUMENTATION_REGISTRY.values() if c.enabled]
    return list(DOCUMENTATION_REGISTRY.values())


def validate_url_against_registry(url: str, tech_id: Optional[str] = None) -> bool:
    """
    Validates that a URL belongs to an approved host in the registry.
    If tech_id is supplied, validates specifically against that technology's allowed hosts.
    """
    import httpx
    try:
        parsed = httpx.URL(url)
        host = parsed.host.lower()
    except Exception:
        return False

    if tech_id:
        cfg = get_technology_config(tech_id)
        if not cfg or not cfg.enabled:
            return False
        return any(host == ah.lower() or host.endswith("." + ah.lower()) for ah in cfg.allowed_hosts)

    for cfg in DOCUMENTATION_REGISTRY.values():
        if cfg.enabled and any(host == ah.lower() or host.endswith("." + ah.lower()) for ah in cfg.allowed_hosts):
            return True
    return False


def get_ecosystem_component(url: str, tech_cfg: Optional[TechnologyConfig] = None) -> str:
    """Determine the ecosystem component for a given URL based on technology config mappings."""
    if tech_cfg and tech_cfg.component_mappings:
        for pattern, comp in tech_cfg.component_mappings.items():
            if pattern in url:
                return comp
    return tech_cfg.tech_id if tech_cfg else "unknown"
