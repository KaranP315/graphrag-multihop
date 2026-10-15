"""Vector search over chunk embeddings stored in Neo4j.

Leverages Neo4j's built-in vector index (`db.index.vector.queryNodes`)
so we don't need a separate vector database.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.embeddings import EmbeddingModel
    from src.graph_store import GraphStore


class VectorStore:
    """Query the Neo4j vector index for semantically similar chunks."""

    def __init__(self, graph_store: GraphStore, embedding_model: EmbeddingModel) -> None:
        self.graph_store = graph_store
        self.embedding_model = embedding_model

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Encode *query* and return the top-k most similar chunks."""
        embedding = self.embedding_model.encode_single(query).tolist()
        return self.search_by_embedding(embedding, top_k=top_k)

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 20,
    ) -> list[dict]:
        """Search using a pre-computed embedding vector."""
        cypher = """
        CALL db.index.vector.queryNodes(
            'chunk_embedding_index',
            $top_k,
            $embedding
        ) YIELD node, score
        RETURN node.chunk_id AS chunk_id,
               node.text      AS text,
               node.title     AS title,
               score
        """
        with self.graph_store.driver.session() as session:
            result = session.run(cypher, top_k=top_k, embedding=embedding)
            return [dict(record) for record in result]
