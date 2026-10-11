import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class GroqConfigurationError(Exception):
    """Raised when Groq API key or required configuration is missing."""
    pass

class GroqTimeoutError(Exception):
    """Raised when request to Groq API times out."""
    pass

class GroqProviderError(Exception):
    """Raised when upstream Groq API encounters an error."""
    pass

class GroqLLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = None

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key or settings.GROQ_API_KEY

    @property
    def model(self) -> str:
        return self._model or settings.GROQ_MODEL

    @property
    def base_url(self) -> str:
        return self._base_url or settings.GROQ_BASE_URL

    @property
    def timeout(self) -> float:
        return self._timeout if self._timeout is not None else settings.GROQ_TIMEOUT_SECONDS

    def is_configured(self) -> bool:
        key = self.api_key
        return bool(key and key.strip())

    def _normalize_base_url(self, raw_url: Optional[str]) -> Optional[str]:
        """
        Normalize the base URL for the official Groq SDK.
        The Groq SDK internally appends '/openai/v1/chat/completions' to its base_url.
        If a user or environment passes 'https://api.groq.com/openai/v1' or 'https://api.groq.com/openai/v1/',
        strip the trailing '/openai/v1' path to prevent duplication (/openai/v1/openai/v1).
        """
        if not raw_url:
            return None
        url = raw_url.strip().rstrip("/")
        if url.endswith("/openai/v1"):
            url = url[:-len("/openai/v1")].rstrip("/")
        return url or None

    def _get_client(self):
        key = self.api_key
        if not key or not key.strip():
            raise GroqConfigurationError(
                "GROQ_API_KEY is not configured. Please supply a valid Groq API key."
            )
        if self._client is None:
            from groq import Groq
            normalized_base = self._normalize_base_url(self.base_url)
            self._client = Groq(
                api_key=key.strip(),
                base_url=normalized_base,
                timeout=self.timeout
            )
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate completion from Groq API with robust error handling.
        """
        client = self._get_client()

        try:
            import groq
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not response.choices or not response.choices[0].message:
                raise GroqProviderError("Empty response choice received from Groq API.")

            content = response.choices[0].message.content
            if content is None:
                raise GroqProviderError("Null content received in Groq API response.")

            return content
        except GroqConfigurationError:
            raise
        except Exception as e:
            # Check groq specific exceptions safely
            err_type = type(e).__name__
            if "AuthenticationError" in err_type:
                logger.error("Groq authentication failure")
                raise GroqConfigurationError("Invalid or unauthorized Groq API key.") from e
            elif "APITimeoutError" in err_type or "Timeout" in err_type:
                logger.error("Groq API request timed out")
                raise GroqTimeoutError("Request to Groq API timed out.") from e
            elif any(t in err_type for t in ["RateLimitError", "APIConnectionError", "APIStatusError", "InternalServerError", "NotFoundError"]):
                logger.error(f"Groq upstream failure: {err_type}")
                raise GroqProviderError(f"Groq provider error: {err_type}") from e
            elif isinstance(e, GroqProviderError):
                raise
            else:
                logger.error(f"Unexpected error calling Groq: {err_type}")
                raise GroqProviderError(f"Unexpected error communicating with Groq: {str(e)}") from e

groq_client = GroqLLMClient()
