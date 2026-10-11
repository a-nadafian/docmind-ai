from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

from app.ingestion.models import DocumentChunk
from app.retrieval.qdrant_store import QdrantVectorStore


class FakeQdrantClient:
    """Minimal Qdrant client substitute for unit tests."""

    def __init__(self) -> None:
        self.created_collection: dict[str, Any] | None = None
        self.deleted: list[dict[str, Any]] = []
        self.points: list[models.PointStruct] = []
        self.last_query: dict[str, Any] | None = None
        self.response_points: list[Any] = []

    def collection_exists(self, collection_name: str) -> bool:
        return False

    def create_collection(self, **kwargs: Any) -> None:
        self.created_collection = kwargs

    def delete(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs)

    def upsert(self, **kwargs: Any) -> None:
        self.points = kwargs["points"]

    def query_points(self, **kwargs: Any) -> Any:
        self.last_query = kwargs
        return SimpleNamespace(points=self.response_points)

    def close(self) -> None:
        pass


@pytest.fixture
def fake_client() -> FakeQdrantClient:
    return FakeQdrantClient()


@pytest.fixture
def store(fake_client: FakeQdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(vector_size=3, client=fake_client)


def test_creates_cosine_collection(
    fake_client: FakeQdrantClient,
    store: QdrantVectorStore,
) -> None:
    assert fake_client.created_collection is not None

    config = fake_client.created_collection["vectors_config"]
    assert config.size == 3
    assert config.distance == models.Distance.COSINE
    assert store.collection_name == "docmind_chunks"


def test_replace_chunks_saves_payload_and_replaces_source(
    fake_client: FakeQdrantClient,
    store: QdrantVectorStore,
) -> None:
    chunks = [
        DocumentChunk(
            source="guide.md",
            text="Semantic search uses vectors.",
            chunk_index=0,
        ),
        DocumentChunk(
            source="guide.md",
            text="Qdrant stores vectors.",
            chunk_index=1,
            page_number=2,
        ),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    count = store.replace_chunks(chunks, vectors)

    assert count == 2
    assert len(fake_client.deleted) == 1
    assert len(fake_client.points) == 2
    assert fake_client.points[0].payload == {
        "source": "guide.md",
        "text": "Semantic search uses vectors.",
        "chunk_index": 0,
    }
    assert fake_client.points[1].payload is not None
    assert fake_client.points[1].payload["page_number"] == 2


def test_invalid_vector_does_not_delete_existing_source(
    fake_client: FakeQdrantClient,
    store: QdrantVectorStore,
) -> None:
    chunk = DocumentChunk(
        source="guide.md",
        text="A test document.",
        chunk_index=0,
    )

    with pytest.raises(ValueError, match="must contain 3 values"):
        store.replace_chunks([chunk], [[1.0, 0.0]])

    assert fake_client.deleted == []
    assert fake_client.points == []


def test_search_returns_retrieved_chunks(
    fake_client: FakeQdrantClient,
    store: QdrantVectorStore,
) -> None:
    fake_client.response_points = [
        SimpleNamespace(
            payload={
                "source": "guide.pdf",
                "text": "Hybrid retrieval combines multiple search methods.",
                "chunk_index": 4,
                "page_number": 2,
            },
            score=0.91,
        )
    ]

    results = store.search([1.0, 0.0, 0.0], limit=3)

    assert len(results) == 1
    assert results[0].source == "guide.pdf"
    assert results[0].page_number == 2
    assert results[0].chunk_index == 4
    assert results[0].score == pytest.approx(0.91)
    assert fake_client.last_query is not None
    assert fake_client.last_query["limit"] == 3


def test_search_rejects_invalid_inputs(
    store: QdrantVectorStore,
) -> None:
    with pytest.raises(ValueError, match="limit must be greater"):
        store.search([1.0, 0.0, 0.0], limit=0)

    with pytest.raises(ValueError, match="must contain 3 values"):
        store.search([1.0, 0.0], limit=3)
