from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingModel:
    """Thin wrapper around sentence-transformers for encoding text."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into normalized embeddings.

        Returns an (N, D) float32 array where each row is unit-length.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string. Returns a 1-D embedding vector."""
        return self.encode([text])[0]

    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors.

        Assumes both vectors are already L2-normalized (which encode() guarantees),
        so the dot product equals cosine similarity.
        """
        return float(np.dot(emb1, emb2))
