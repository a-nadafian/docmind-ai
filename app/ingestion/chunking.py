from app.ingestion.models import DocumentChunk, DocumentPage


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """Split text into overlapping chunks without losing characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            minimum_boundary = start + chunk_size // 2
            boundary = text.rfind(" ", minimum_boundary, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - chunk_overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_documents(
    pages: list[DocumentPage],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    """Create chunks while preserving source and page metadata."""
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        page_chunks = split_text(
            page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for text in page_chunks:
            chunks.append(
                DocumentChunk(
                    source=page.source,
                    text=text,
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                )
            )
            chunk_index += 1

    return chunks
