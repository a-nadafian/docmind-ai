from fastapi.testclient import TestClient

from app.api.search import get_search_service
from app.main import app
from app.retrieval.base import RetrievedChunk


class FakeSearchService:
    """Provide deterministic results without external services."""

    def __init__(self) -> None:
        self.last_question: str | None = None
        self.last_limit: int | None = None

    def search(
        self,
        question: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        self.last_question = question
        self.last_limit = limit

        return [
            RetrievedChunk(
                source="sample.md",
                text="Hybrid retrieval combines semantic search with keyword search.",
                chunk_index=0,
                page_number=None,
                score=0.91,
            )
        ]


def test_search_endpoint_returns_ranked_evidence() -> None:
    service = FakeSearchService()
    app.dependency_overrides[get_search_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "question": "  What is hybrid retrieval?  ",
                    "top_k": 3,
                },
            )
    finally:
        app.dependency_overrides.pop(get_search_service, None)

    assert response.status_code == 200
    assert response.json() == {
        "question": "What is hybrid retrieval?",
        "results": [
            {
                "source": "sample.md",
                "page_number": None,
                "chunk_index": 0,
                "score": 0.91,
                "text": (
                    "Hybrid retrieval combines semantic search with keyword search."
                ),
            }
        ],
    }

    assert service.last_question == "What is hybrid retrieval?"
    assert service.last_limit == 3


def test_search_endpoint_rejects_blank_question() -> None:
    service = FakeSearchService()
    app.dependency_overrides[get_search_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={"question": "   "},
            )
    finally:
        app.dependency_overrides.pop(get_search_service, None)

    assert response.status_code == 422
    assert service.last_question is None
