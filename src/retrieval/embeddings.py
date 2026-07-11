"""Sentence-transformers embedding wrapper for local vector search."""

from __future__ import annotations

from collections.abc import Sequence


MODEL_LOAD_HELP = (
    "Unable to load the embedding model. The first run may need to download "
    "the model from Hugging Face. Check your network connection, or pre-download "
    "the model, then rerun the command."
)


class SentenceTransformerEmbedder:
    """Small wrapper that applies E5 query/passage prefixes consistently."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "sentence-transformers is required. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - environment/network dependent
            raise RuntimeError(f"{MODEL_LOAD_HELP} Model: {model_name}") from exc

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [f"passage: {text}" for text in texts]
        return self._encode(prefixed)

    def embed_query(self, query: str) -> list[float]:
        return self._encode([f"query: {query}"])[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            vectors = self.model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            vectors = self.model.encode(list(texts), show_progress_bar=False)
        return [list(map(float, vector)) for vector in vectors]
