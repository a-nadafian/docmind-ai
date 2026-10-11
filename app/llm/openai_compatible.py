import os
from typing import Any

import httpx

from app.llm.base import LLMConfigurationError, LLMProviderError

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"


class OpenAICompatibleChatModel:
    """Chat model adapter for OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("API key must not be empty.")

        resolved_base_url = base_url.strip().rstrip("/")
        if not resolved_base_url.startswith(("https://", "http://")):
            raise LLMConfigurationError(
                "LLM_BASE_URL must be a valid HTTP or HTTPS URL."
            )

        if not model.strip():
            raise LLMConfigurationError("LLM_MODEL must not be empty.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        self._api_key = api_key.strip()
        self._base_url = resolved_base_url
        self._model = model.strip()
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
        )

    @classmethod
    def from_env(cls) -> "OpenAICompatibleChatModel":
        """Build the adapter from environment variables."""
        api_key = os.getenv("LLM_API_KEY", "").strip()

        if not api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is not configured. "
                "Set it in your environment before calling /ask."
            )

        return cls(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 800,
    ) -> str:
        """Send a chat completion request and return its text."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")

        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"LLM provider returned HTTP {exc.response.status_code}."
            ) from exc

        except httpx.RequestError as exc:
            raise LLMProviderError("Could not connect to the LLM provider.") from exc

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise LLMProviderError("LLM provider returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise LLMProviderError("LLM provider returned an unexpected response.")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMProviderError("LLM provider returned no completion choices.")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMProviderError("LLM provider returned an invalid choice.")

        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMProviderError("LLM provider returned an invalid message.")

        content = message.get("content")

        if isinstance(content, str):
            answer = content.strip()

        elif isinstance(content, list):
            parts = [
                part["text"]
                for part in content
                if (isinstance(part, dict) and isinstance(part.get("text"), str))
            ]
            answer = "".join(parts).strip()

        else:
            answer = ""

        if not answer:
            raise LLMProviderError("LLM provider returned an empty answer.")

        return answer
