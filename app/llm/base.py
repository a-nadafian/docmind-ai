from typing import Protocol


class LLMConfigurationError(RuntimeError):
    """Raised when the language model is not configured correctly."""


class LLMProviderError(RuntimeError):
    """Raised when the language model provider fails."""


class ChatModel(Protocol):
    """Common interface for chat-based language models."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 800,
    ) -> str:
        """Generate a text response."""
        ...
