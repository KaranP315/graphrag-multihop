"""Cross-encoder reranker for candidate chunk lists.

Takes a query + candidate documents and rescores them with a
cross-encoder model, giving much better relevance ordering than
bi-encoder cosine alone.
"""
from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL

logger = logging.getLogger(__name__)


class Reranker:
    """Wraps a cross-encoder model for query-document reranking."""

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or RERANKER_MODEL
        logger.info("Loading reranker model: %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 10,
        text_key: str = "text",
    ) -> list[dict]:
        """Score each document against the query and return the top-k.

        Each returned dict gets an additional ``rerank_score`` key.
        """
        if not documents:
            return []

        pairs = [(query, doc[text_key]) for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return ranked[:top_k]
