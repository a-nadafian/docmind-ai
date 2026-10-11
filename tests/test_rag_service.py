from app.rag.service import RAGService
from app.retrieval.base import RetrievedChunk


class FakeSearchService:
    def __init__(
        self,
        chunks: list[RetrievedChunk],
    ) -> None:
        self.chunks = chunks
        self.last_question: str | None = None
        self.last_limit: int | None = None

    def search(
        self,
        question: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        self.last_question = question
        self.last_limit = limit
        return self.chunks


class FakeChatModel:
    def __init__(self) -> None:
        self.called = False
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None
        self.max_tokens: int | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 800,
    ) -> str:
        self.called = True
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_tokens = max_tokens
        return "Hybrid retrieval combines both approaches [S1]."


def test_rag_service_generates_answer_from_evidence() -> None:
    chunk = RetrievedChunk(
        source="sample.md",
        text="Hybrid retrieval combines semantic search with keyword search.",
        chunk_index=0,
        page_number=None,
        score=0.91,
    )

    search = FakeSearchService([chunk])
    model = FakeChatModel()
    service = RAGService(search, model)

    result = service.answer(
        "What is hybrid retrieval?",
        top_k=3,
    )

    assert result.answer.endswith("[S1].")
    assert result.sources == [chunk]
    assert search.last_question == "What is hybrid retrieval?"
    assert search.last_limit == 3

    assert model.called
    assert model.user_prompt is not None
    assert '"id": "S1"' in model.user_prompt
    assert "sample.md" in model.user_prompt
    assert model.max_tokens == 800


def test_rag_service_does_not_generate_without_evidence() -> None:
    search = FakeSearchService([])
    model = FakeChatModel()
    service = RAGService(search, model)

    result = service.answer("A question without matching documents")

    assert result.sources == []
    assert "No relevant passages" in result.answer
    assert not model.called
