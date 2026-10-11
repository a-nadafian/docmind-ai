from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Common interface for embedding providers."""

    @property
    def dimension(self) -> int:
        """Return the number of dimensions in each embedding."""
        ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Embed document passages."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...
