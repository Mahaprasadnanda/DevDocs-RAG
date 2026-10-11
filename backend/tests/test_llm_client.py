import pytest
from unittest.mock import MagicMock, patch
from app.llm.client import (
    GroqLLMClient,
    GroqConfigurationError,
    GroqTimeoutError,
    GroqProviderError
)

def test_missing_api_key_raises_configuration_error(monkeypatch):
    client = GroqLLMClient(api_key="")
    assert not client.is_configured()
    with pytest.raises(GroqConfigurationError, match="GROQ_API_KEY is not configured"):
        client.generate([{"role": "user", "content": "hi"}])

def test_successful_generation():
    client = GroqLLMClient(api_key="gsk-testkey")
    mock_groq = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a grounded technical answer."
    mock_completion.choices = [mock_choice]
    mock_groq.chat.completions.create.return_value = mock_completion
    client._client = mock_groq

    res = client.generate([{"role": "user", "content": "How to create an endpoint?"}])
    assert res == "This is a grounded technical answer."
    mock_groq.chat.completions.create.assert_called_once()

def test_empty_response_choices_raises_provider_error():
    client = GroqLLMClient(api_key="gsk-testkey")
    mock_groq = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = []
    mock_groq.chat.completions.create.return_value = mock_completion
    client._client = mock_groq

    with pytest.raises(GroqProviderError, match="Empty response choice"):
        client.generate([{"role": "user", "content": "hi"}])

def test_null_content_raises_provider_error():
    client = GroqLLMClient(api_key="gsk-testkey")
    mock_groq = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_completion.choices = [mock_choice]
    mock_groq.chat.completions.create.return_value = mock_completion
    client._client = mock_groq

    with pytest.raises(GroqProviderError, match="Null content received"):
        client.generate([{"role": "user", "content": "hi"}])

def test_authentication_error_mapped_to_configuration_error():
    client = GroqLLMClient(api_key="gsk-invalid")
    mock_groq = MagicMock()

    class FakeAuthError(Exception):
        pass
    FakeAuthError.__name__ = "AuthenticationError"

    mock_groq.chat.completions.create.side_effect = FakeAuthError("Invalid API Key gsk-secret-12345")
    client._client = mock_groq

    with pytest.raises(GroqConfigurationError) as exc_info:
        client.generate([{"role": "user", "content": "hi"}])
    # Ensure raw secret is not leaked in our exception message
    assert "gsk-secret-12345" not in str(exc_info.value)

def test_timeout_error_mapped_to_timeout_error():
    client = GroqLLMClient(api_key="gsk-testkey")
    mock_groq = MagicMock()

    class FakeTimeoutError(Exception):
        pass
    FakeTimeoutError.__name__ = "APITimeoutError"

    mock_groq.chat.completions.create.side_effect = FakeTimeoutError("Request timed out")
    client._client = mock_groq

    with pytest.raises(GroqTimeoutError, match="timed out"):
        client.generate([{"role": "user", "content": "hi"}])

def test_rate_limit_mapped_to_provider_error():
    client = GroqLLMClient(api_key="gsk-testkey")
    mock_groq = MagicMock()

    class FakeRateLimitError(Exception):
        pass
    FakeRateLimitError.__name__ = "RateLimitError"

    mock_groq.chat.completions.create.side_effect = FakeRateLimitError("Rate limit reached")
    client._client = mock_groq

    with pytest.raises(GroqProviderError, match="RateLimitError"):
        client.generate([{"role": "user", "content": "hi"}])


def test_not_found_error_mapped_to_provider_error():
    client = GroqLLMClient(api_key="gsk-testkey")
    mock_groq = MagicMock()

    class FakeNotFoundError(Exception):
        pass
    FakeNotFoundError.__name__ = "NotFoundError"

    mock_groq.chat.completions.create.side_effect = FakeNotFoundError("Unknown request URL: POST /openai/v1/openai/v1/chat/completions")
    client._client = mock_groq

    with pytest.raises(GroqProviderError, match="NotFoundError"):
        client.generate([{"role": "user", "content": "hi"}])


def test_groq_client_default_base_url_url_construction():
    import httpx
    captured_urls = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Grounded answer"}}]}
        )

    client = GroqLLMClient(api_key="gsk-test", base_url="https://api.groq.com")
    raw_groq = client._get_client()
    raw_groq._client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))

    res = client.generate([{"role": "user", "content": "test query"}])
    assert res == "Grounded answer"
    assert len(captured_urls) == 1
    assert captured_urls[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert "/openai/v1/openai/v1" not in captured_urls[0]


def test_groq_client_normalizes_legacy_base_url_preventing_duplicate():
    import httpx
    captured_urls = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Grounded answer"}}]}
        )

    # When configured with legacy or stale https://api.groq.com/openai/v1
    client = GroqLLMClient(api_key="gsk-test", base_url="https://api.groq.com/openai/v1")
    raw_groq = client._get_client()
    raw_groq._client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))

    res = client.generate([{"role": "user", "content": "test query"}])
    assert res == "Grounded answer"
    assert len(captured_urls) == 1
    assert "/openai/v1/openai/v1" not in captured_urls[0]
    assert captured_urls[0] == "https://api.groq.com/openai/v1/chat/completions"


def test_groq_client_normalizes_trailing_slash_base_url():
    import httpx
    captured_urls = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Grounded answer"}}]}
        )

    # When configured with trailing slash
    client = GroqLLMClient(api_key="gsk-test", base_url="https://api.groq.com/openai/v1/")
    raw_groq = client._get_client()
    raw_groq._client = httpx.Client(transport=httpx.MockTransport(mock_transport_handler))

    res = client.generate([{"role": "user", "content": "test query"}])
    assert res == "Grounded answer"
    assert len(captured_urls) == 1
    assert "/openai/v1/openai/v1" not in captured_urls[0]
    assert captured_urls[0] == "https://api.groq.com/openai/v1/chat/completions"

