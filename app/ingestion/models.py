from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentPage:
    source: str
    text: str
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    source: str
    text: str
    chunk_index: int
    page_number: int | None = None
