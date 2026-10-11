from collections.abc import Sequence
from math import isfinite
from os import getenv
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.ingestion.models import DocumentChunk
from app.retrieval.base import RetrievedChunk

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "docmind_chunks"


class QdrantVectorStore:
    """Store and search document embeddings using Qdrant."""

    def __init__(
        self,
        vector_size: int,
        url: str | None = None,
        collection_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")

        resolved_collection = collection_name or getenv(
            "QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME
        )
        if not resolved_collection.strip():
            raise ValueError("collection_name must not be empty")

        self.vector_size = vector_size
        self.collection_name = resolved_collection

        self.client = (
            client
            if client is not None
            else QdrantClient(url=url or getenv("QDRANT_URL", DEFAULT_QDRANT_URL))
        )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection or validate its vector dimensions."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        collection = self.client.get_collection(collection_name=self.collection_name)
        configured_vectors = collection.config.params.vectors
        configured_size = getattr(configured_vectors, "size", None)

        if configured_size != self.vector_size:
            raise ValueError(
                f"Collection {self.collection_name!r} has vector size "
                f"{configured_size}, but this model requires "
                f"{self.vector_size}. Use a compatible collection or "
                "a separate collection for this embedding model."
            )

    def _validate_vector(
        self,
        vector: Sequence[float],
        role: str,
    ) -> list[float]:
        """Validate vector dimensions and numeric values."""
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{role} must contain numeric values") from exc

        if len(values) != self.vector_size:
            raise ValueError(
                f"{role} must contain {self.vector_size} values; received {len(values)}"
            )

        if not all(isfinite(value) for value in values):
            raise ValueError(f"{role} must contain only finite values")

        return values

    def _delete_source(self, source: str) -> None:
        """Delete previously indexed chunks for one source."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def replace_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Replace existing chunks for each source in the supplied batch."""
        if len(chunks) != len(vectors):
            raise ValueError("The number of chunks must match the number of vectors")

        if not chunks:
            return 0

        # Validate the complete batch before modifying stored data.
        validated_vectors = [
            self._validate_vector(vector, f"vector at index {index}")
            for index, vector in enumerate(vectors)
        ]

        points: list[models.PointStruct] = []

        for chunk, vector in zip(chunks, validated_vectors, strict=True):
            # Stable IDs allow the same source/chunk position to be
            # identified consistently between indexing runs.
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"docmind-ai:{chunk.source}:{chunk.chunk_index}",
                )
            )

            payload: dict[str, Any] = {
                "source": chunk.source,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
            }

            if chunk.page_number is not None:
                payload["page_number"] = chunk.page_number

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Remove older chunks for the affected sources first so that
        # re-indexing a shorter document does not leave stale chunks.
        sources = sorted({chunk.source for chunk in chunks})
        for source in sources:
            self._delete_source(source)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """Return the most similar stored document chunks."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        vector = self._validate_vector(query_vector, "query_vector")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        results: list[RetrievedChunk] = []

        for point in response.points:
            payload = point.payload
            if not payload:
                continue

            try:
                source = payload["source"]
                text = payload["text"]
                chunk_index = int(payload["chunk_index"])
                page_number = payload.get("page_number")

                if not isinstance(source, str) or not isinstance(text, str):
                    continue

                results.append(
                    RetrievedChunk(
                        source=source,
                        text=text,
                        chunk_index=chunk_index,
                        page_number=(
                            int(page_number) if page_number is not None else None
                        ),
                        score=float(point.score),
                    )
                )
            except (KeyError, TypeError, ValueError):
                # Ignore malformed points rather than failing the search.
                continue

        return results

    def close(self) -> None:
        """Release the Qdrant client resources."""
        self.client.close()
