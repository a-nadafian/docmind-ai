from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.api.search import get_search_service
from app.llm.base import LLMConfigurationError, LLMProviderError
from app.llm.openai_compatible import OpenAICompatibleChatModel
from app.rag.service import RAGService

router = APIRouter(tags=["RAG"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Question must not be blank.")

        return normalized


class AnswerSource(BaseModel):
    id: str
    source: str
    page_number: int | None = None
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[AnswerSource]


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Initialize the RAG service on its first use."""
    try:
        chat_model = OpenAICompatibleChatModel.from_env()
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return RAGService(
        search_service=get_search_service(),
        chat_model=chat_model,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer a question using indexed documents",
)
def ask_question(
    request: AskRequest,
    service: Annotated[
        RAGService,
        Depends(get_rag_service),
    ],
) -> AskResponse:
    """Return a grounded answer with references to retrieved chunks."""
    try:
        result = service.answer(
            question=request.question,
            top_k=request.top_k,
        )
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    sources = [
        AnswerSource(
            id=f"S{index}",
            source=chunk.source,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
        )
        for index, chunk in enumerate(result.sources, start=1)
    ]

    return AskResponse(
        question=request.question,
        answer=result.answer,
        sources=sources,
    )
