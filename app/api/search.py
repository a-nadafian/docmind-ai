from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.embeddings import FastEmbedProvider
from app.retrieval import QdrantVectorStore, RetrievedChunk
from app.retrieval.service import SemanticSearchService

router = APIRouter(tags=["Search"])


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Question must not be blank.")

        return normalized


class SearchHit(BaseModel):
    source: str
    page_number: int | None = None
    chunk_index: int
    score: float
    text: str


class SearchResponse(BaseModel):
    question: str
    results: list[SearchHit]


@lru_cache(maxsize=1)
def get_search_service() -> SemanticSearchService:
    """Lazily initialize the embedding and vector store clients."""
    embedder = FastEmbedProvider()
    vector_store = QdrantVectorStore(
        vector_size=embedder.dimension,
    )

    return SemanticSearchService(
        embedder=embedder,
        vector_store=vector_store,
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search indexed document chunks",
)
def search_documents(
    request: SearchRequest,
    service: Annotated[
        SemanticSearchService,
        Depends(get_search_service),
    ],
) -> SearchResponse:
    """Return relevant document passages for a natural-language question."""
    chunks: list[RetrievedChunk] = service.search(
        question=request.question,
        limit=request.top_k,
    )

    return SearchResponse(
        question=request.question,
        results=[
            SearchHit(
                source=chunk.source,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                text=chunk.text,
            )
            for chunk in chunks
        ],
    )
