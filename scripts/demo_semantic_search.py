from pathlib import Path

from app.embeddings import FastEmbedProvider
from app.ingestion.indexing import index_file
from app.retrieval import QdrantVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = PROJECT_ROOT / "data" / "fixtures" / "sample.md"

QUERY = "What is hybrid retrieval and how does it work?"


def main() -> None:
    embedder = FastEmbedProvider()
    store = QdrantVectorStore(vector_size=embedder.dimension)

    try:
        indexed_count = index_file(
            SAMPLE_FILE,
            embedder=embedder,
            vector_store=store,
        )
        print(f"Indexed chunks: {indexed_count}")

        query_vector = embedder.embed_query(QUERY)
        results = store.search(query_vector, limit=3)

        print(f"\nQuery: {QUERY}")
        print(f"Retrieved results: {len(results)}")

        for rank, result in enumerate(results, start=1):
            print(
                f"\n[{rank}] score={result.score:.4f} "
                f"source={result.source} "
                f"page={result.page_number} "
                f"chunk={result.chunk_index}"
            )
            print(result.text[:500])

    finally:
        store.close()


if __name__ == "__main__":
    main()
