from pathlib import Path

from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_document
from app.retrieval.base import VectorStore


def index_file(
    file_path: str | Path,
    embedder: EmbeddingProvider,
    vector_store: VectorStore,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    """Load, chunk, embed, and index one document."""
    pages = load_document(file_path)

    chunks = chunk_documents(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(f"No text chunks were produced from {file_path}")

    vectors = embedder.embed_documents([chunk.text for chunk in chunks])

    return vector_store.replace_chunks(chunks, vectors)
