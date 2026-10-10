from pathlib import Path

import pymupdf

from app.ingestion.models import DocumentPage

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


def load_document(file_path: str | Path) -> list[DocumentPage]:
    """Extract text and source metadata from a PDF or Markdown file."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or '<no extension>'}")

    if suffix == ".pdf":
        pages: list[DocumentPage] = []

        with pymupdf.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text", sort=True).strip()

                if text:
                    pages.append(
                        DocumentPage(
                            source=path.name,
                            text=text,
                            page_number=page_number,
                        )
                    )

        if not pages:
            raise ValueError(
                f"No extractable text found in PDF: {path.name}. "
                "The PDF may be scanned or empty."
            )

        return pages

    text = path.read_text(encoding="utf-8-sig").strip()

    if not text:
        raise ValueError(f"Document is empty: {path.name}")

    return [
        DocumentPage(
            source=path.name,
            text=text,
            page_number=None,
        )
    ]
