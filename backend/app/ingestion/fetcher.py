import httpx
from typing import Tuple, Optional
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class FetchError(Exception):
    """Custom exception for fetcher errors."""
    pass

def fetch_html(url: str, allowed_hosts: Optional[list] = None) -> Tuple[str, httpx.Response]:
    """
    Fetches HTML content from a URL synchronously.
    
    A synchronous client is chosen here because this is an offline CLI pipeline 
    processing a small curated list of pages sequentially. It simplifies the 
    architecture without needing asyncio loops.
    
    Returns:
        A tuple of (final_url, response_object).
        
    Raises:
        FetchError if the fetch fails (timeout, bad status, etc.)
    """
    headers = {
        "User-Agent": settings.INGESTION_USER_AGENT
    }
    timeout = httpx.Timeout(settings.INGESTION_REQUEST_TIMEOUT)
    
    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            
            # Enforce content-type
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type.lower():
                raise FetchError(f"Unsupported content type: {content_type}")
                
            response.raise_for_status()
            
            # Check size if Content-Length is provided
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > settings.INGESTION_MAX_RESPONSE_BYTES:
                raise FetchError(f"Response too large: {content_length} bytes")
                
            # If length wasn't provided, check actual body size
            if len(response.content) > settings.INGESTION_MAX_RESPONSE_BYTES:
                raise FetchError(f"Response body too large: {len(response.content)} bytes")
                
            # Restrict redirects to approved hosts
            orig_host = httpx.URL(url).host
            resp_host = response.url.host
            if allowed_hosts:
                valid_hosts = [h.lower() for h in allowed_hosts] + [orig_host.lower()]
                if not any(resp_host.lower() == h or resp_host.lower().endswith("." + h) for h in valid_hosts):
                    raise FetchError(f"Redirected to disallowed domain: {resp_host}")
            elif resp_host != orig_host:
                raise FetchError(f"Redirected to disallowed domain: {resp_host}")
                
            return str(response.url), response
            
    except httpx.TimeoutException as e:
        raise FetchError(f"Request timed out: {e}")
    except httpx.HTTPStatusError as e:
        raise FetchError(f"HTTP error: {e.response.status_code}")
    except httpx.RequestError as e:
        raise FetchError(f"Request error: {e}")
