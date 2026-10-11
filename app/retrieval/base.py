from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.ingestion.models import DocumentChunk


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A document chunk retrieved by vector similarity."""

    source: str
    text: str
    chunk_index: int
    page_number: int | None
    score: float


class VectorStore(Protocol):
    """Common interface for vector database implementations."""

    def replace_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Replace indexed chunks for the supplied document sources."""
        ...

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """Search for the most relevant chunks."""
        ...
