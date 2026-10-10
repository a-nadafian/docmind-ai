import pymupdf
import pytest

from app.ingestion.chunking import chunk_documents, split_text
from app.ingestion.loaders import load_document
from app.ingestion.models import DocumentPage


def test_load_markdown_preserves_text_and_source(tmp_path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "# DocMind AI\n\nDocument retrieval basics.",
        encoding="utf-8",
    )

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].source == "guide.md"
    assert pages[0].page_number is None
    assert "# DocMind AI" in pages[0].text


def test_load_pdf_preserves_page_numbers(tmp_path) -> None:
    path = tmp_path / "guide.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Document retrieval basics.")
        document.save(path)

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].source == "guide.pdf"
    assert pages[0].page_number == 1
    assert "Document retrieval basics." in pages[0].text


def test_load_document_rejects_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Some text", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)


def test_split_text_creates_overlapping_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(100))

    chunks = split_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_documents_preserves_metadata() -> None:
    page = DocumentPage(
        source="manual.pdf",
        text=" ".join(f"word{i}" for i in range(100)),
        page_number=3,
    )

    chunks = chunk_documents(
        [page],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunk.source == "manual.pdf" for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_split_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        split_text("some text", chunk_size=100, chunk_overlap=100)
