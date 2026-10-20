"""GraphRAG package with lazy attribute exports (PEP 562)."""
from __future__ import annotations

import importlib

__all__ = [
    "HotpotQALoader",
    "Question",
    "Chunker",
    "Chunk",
    "EmbeddingModel",
    "LLMClient",
    "Entity",
    "Relation",
    "ExtractionResult",
    "GraphExtractor",
    "EntityResolver",
    "GraphStore",
    "VectorStore",
    "BaselineRAG",
    "GraphRAG",
    "Reranker",
    "QuestionDecomposer",
]

_MAP = {
    "HotpotQALoader": "src.data_loader",
    "Question": "src.data_loader",
    "Chunker": "src.chunker",
    "Chunk": "src.chunker",
    "EmbeddingModel": "src.embeddings",
    "LLMClient": "src.llm_client",
    "Entity": "src.schemas",
    "Relation": "src.schemas",
    "ExtractionResult": "src.schemas",
    "GraphExtractor": "src.extraction",
    "EntityResolver": "src.entity_resolution",
    "GraphStore": "src.graph_store",
    "VectorStore": "src.vector_store",
    "BaselineRAG": "src.baseline_rag",
    "GraphRAG": "src.graph_rag",
    "Reranker": "src.reranker",
    "QuestionDecomposer": "src.decomposition",
}


def __getattr__(name: str):
    if name in _MAP:
        mod = importlib.import_module(_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
