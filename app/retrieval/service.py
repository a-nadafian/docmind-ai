from app.embeddings.base import EmbeddingProvider
from app.retrieval.base import RetrievedChunk, VectorStore


class SemanticSearchService:
    """Coordinate query embedding and vector retrieval."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def search(
        self,
        question: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a question."""
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Question must not be empty.")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero.")

        query_vector = self._embedder.embed_query(normalized_question)

        return self._vector_store.search(
            query_vector=query_vector,
            limit=limit,
        )
