import pytest

from app.embeddings.fastembed_provider import FastEmbedProvider


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class FakeFastEmbedModel:
    def passage_embed(self, texts: list[str]):
        return (FakeVector([0.25] * 384) for _ in texts)

    def query_embed(self, texts: list[str]):
        return (FakeVector([0.5] * 384) for _ in texts)


@pytest.fixture
def provider() -> FastEmbedProvider:
    return FastEmbedProvider(model=FakeFastEmbedModel())


def test_embed_documents_returns_correct_vectors(
    provider: FastEmbedProvider,
) -> None:
    vectors = provider.embed_documents(["First document", "Second document"])

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert vectors[0][0] == 0.25


def test_embed_query_returns_correct_vector(
    provider: FastEmbedProvider,
) -> None:
    vector = provider.embed_query("Search query")

    assert len(vector) == 384
    assert vector[0] == 0.5


def test_embed_documents_accepts_empty_list(
    provider: FastEmbedProvider,
) -> None:
    assert provider.embed_documents([]) == []


def test_embed_query_rejects_blank_text(
    provider: FastEmbedProvider,
) -> None:
    with pytest.raises(ValueError, match="empty strings"):
        provider.embed_query("   ")
