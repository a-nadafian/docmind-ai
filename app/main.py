from fastapi import FastAPI

from app.api.health import router as health_router

app = FastAPI(
    title="DocMind AI",
    description=(
        "Advanced RAG system with hybrid retrieval, reranking, and evaluation."
    ),
    version="0.1.0",
)

app.include_router(health_router)


@app.get("/", tags=["General"], summary="API overview")
def root() -> dict[str, str]:
    """Return basic information about the API."""
    return {
        "service": "docmind-ai",
        "message": "DocMind AI API is running",
        "docs": "/docs",
        "health": "/health",
    }
