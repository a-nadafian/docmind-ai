from collections.abc import Sequence
from typing import Any

from fastembed import TextEmbedding

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedProvider:
    """Generate embeddings using FastEmbed."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        model: Any | None = None,
    ) -> None:
        supported_models = TextEmbedding.list_supported_models()

        model_info = next(
            (item for item in supported_models if item["model"] == model_name),
            None,
        )

        if model_info is None:
            raise ValueError(f"Unsupported FastEmbed model: {model_name}")

        self.model_name = model_name
        self.dimension = int(model_info["dim"])

        # Model injection makes unit testing possible without downloading
        # model weights.
        self._model = (
            model if model is not None else TextEmbedding(model_name=model_name)
        )

    def _encode(
        self,
        texts: Sequence[str],
        encoder_name: str,
    ) -> list[list[float]]:
        normalized_texts = [text.strip() for text in texts]

        if any(not text for text in normalized_texts):
            raise ValueError("Texts must not contain empty strings.")

        if not normalized_texts:
            return []

        encoder = getattr(self._model, encoder_name)
        raw_vectors = list(encoder(normalized_texts))

        if len(raw_vectors) != len(normalized_texts):
            raise RuntimeError(
                "The embedding model returned an unexpected number of vectors."
            )

        vectors: list[list[float]] = []

        for raw_vector in raw_vectors:
            values = (
                raw_vector.tolist()
                if hasattr(raw_vector, "tolist")
                else list(raw_vector)
            )
            vector = [float(value) for value in values]

            if len(vector) != self.dimension:
                raise RuntimeError(
                    "Unexpected embedding dimension: "
                    f"expected {self.dimension}, got {len(vector)}."
                )

            vectors.append(vector)

        return vectors

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for document passages."""
        return self._encode(texts, "passage_embed")

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a search query."""
        vectors = self._encode([text], "query_embed")
        return vectors[0]
