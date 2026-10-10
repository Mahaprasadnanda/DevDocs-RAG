import pytest
from unittest.mock import patch, MagicMock
import httpx
from app.ingestion.fetcher import fetch_html, FetchError
from app.core.config import settings

@pytest.fixture
def mock_httpx_client():
    with patch("httpx.Client") as mock_client:
        yield mock_client

def test_fetch_html_success(mock_httpx_client):
    mock_instance = mock_httpx_client.return_value.__enter__.return_value
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html; charset=utf-8", "content-length": "1000"}
    mock_response.content = b"<html></html>"
    mock_response.url = httpx.URL("https://fastapi.tiangolo.com/")
    
    mock_instance.get.return_value = mock_response
    
    url, response = fetch_html("https://fastapi.tiangolo.com/")
    
    assert url == "https://fastapi.tiangolo.com/"
    assert response == mock_response
    mock_instance.get.assert_called_once_with("https://fastapi.tiangolo.com/")

def test_fetch_html_timeout(mock_httpx_client):
    mock_instance = mock_httpx_client.return_value.__enter__.return_value
    mock_instance.get.side_effect = httpx.TimeoutException("Timeout")
    
    with pytest.raises(FetchError, match="Request timed out"):
        fetch_html("https://fastapi.tiangolo.com/")

def test_fetch_html_http_error(mock_httpx_client):
    mock_instance = mock_httpx_client.return_value.__enter__.return_value
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.headers = {"content-type": "text/html"}
    mock_response.url = httpx.URL("https://fastapi.tiangolo.com/")
    
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("404 Not Found", request=MagicMock(), response=mock_response)
    mock_instance.get.return_value = mock_response
    
    with pytest.raises(FetchError, match="HTTP error: 404"):
        fetch_html("https://fastapi.tiangolo.com/")

def test_fetch_html_unsupported_content_type(mock_httpx_client):
    mock_instance = mock_httpx_client.return_value.__enter__.return_value
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/json"}
    mock_instance.get.return_value = mock_response
    
    with pytest.raises(FetchError, match="Unsupported content type"):
        fetch_html("https://fastapi.tiangolo.com/")

def test_fetch_html_oversized_response(mock_httpx_client):
    mock_instance = mock_httpx_client.return_value.__enter__.return_value
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/html", "content-length": str(settings.INGESTION_MAX_RESPONSE_BYTES + 100)}
    mock_instance.get.return_value = mock_response
    
    with pytest.raises(FetchError, match="Response too large"):
        fetch_html("https://fastapi.tiangolo.com/")

def test_fetch_html_disallowed_redirect(mock_httpx_client):
    mock_instance = mock_httpx_client.return_value.__enter__.return_value
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/html"}
    mock_response.url = httpx.URL("https://evil.com/")
    mock_instance.get.return_value = mock_response
    
    with pytest.raises(FetchError, match="Redirected to disallowed domain"):
        fetch_html("https://fastapi.tiangolo.com/")
