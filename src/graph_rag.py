"""
Hybrid graph + vector RAG system.

Combines dense retrieval with knowledge-graph traversal: vector search finds
an initial set of relevant chunks, entity linking + multi-hop expansion pulls
in structurally connected evidence, and a cross-encoder reranker merges
everything before prompting the LLM.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.vector_store import VectorStore
from src.graph_store import GraphStore
from src.llm_client import LLMClient
from src.reranker import Reranker
from src.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Use the provided context passages AND knowledge-graph triples to answer. "
    "If the information is insufficient, say so. "
    "Keep your answer concise — one or two sentences."
)

ENTITY_EXTRACTION_PROMPT = """Extract the named entities from the following question.
Return a JSON object with a single key "entities" whose value is a list of entity name strings.
Only include proper nouns, specific names, or notable concepts — not generic words.

Question: {question}

JSON:"""

ANSWER_TEMPLATE = """Context passages:
{context}

Knowledge-graph triples:
{triples}

Question: {question}

Answer:"""


def _format_chunks(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Untitled")
        text = c.get("text", "").strip()
        parts.append(f"[{i}] (Title: {title})\n{text}")
    return "\n\n".join(parts)


def _format_triples(triples: list[tuple]) -> str:
    if not triples:
        return "(none)"
    lines = []
    for subj, rel, obj in triples[:30]:  # cap to keep prompt reasonable
        lines.append(f"• {subj}  —[{rel}]→  {obj}")
    return "\n".join(lines)


class GraphRAG:
    """Hybrid retrieval combining vector search with knowledge-graph expansion."""

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        llm: LLMClient,
        reranker: Reranker,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.llm = llm
        self.reranker = reranker
        self.embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Entity extraction & linking
    # ------------------------------------------------------------------

    def _extract_question_entities(self, question: str) -> list[str]:
        """Use the LLM to pull entity names out of the question."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)
        try:
            data = self.llm.generate_json(prompt, system_prompt="You extract entities from text.")
            entities = data.get("entities", [])
            # sometimes the model wraps each entity in extra quotes or nests lists
            return [str(e).strip() for e in entities if str(e).strip()]
        except Exception:
            logger.warning("Entity extraction failed; falling back to empty list")
            return []

    def _link_to_graph(self, entity_names: list[str]) -> list[str]:
        """Fuzzy-match extracted names against actual graph nodes."""
        if not entity_names:
            return []

        linked: list[str] = []
        for name in entity_names:
            try:
                matches = self.graph_store.find_matching_entities(name, limit=3)
                if matches:
                    linked.extend(matches)
                else:
                    linked.append(name)
            except Exception:
                linked.append(name)
        return list(dict.fromkeys(linked))  # deduplicate, preserve order

    # ------------------------------------------------------------------
    # Full answer pipeline
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k_vector: int = 20,
        top_k_graph: int = 20,
        top_k_final: int = 10,
        hops: int = 2,
    ) -> dict:
        """
        End-to-end hybrid retrieval + generation.

        1. Vector search
        2. Entity extraction & graph linking
        3. Multi-hop graph expansion
        4. Merge & deduplicate
        5. Rerank
        6. Fetch relevant triples
        7. Build prompt and call LLM
        """

        # 1) Vector retrieval
        vector_hits = self.vector_store.search(question, top_k=top_k_vector)
        vector_chunks = [
            {
                "chunk_id": h.get("chunk_id", ""),
                "text": h.get("text", ""),
                "title": h.get("title", ""),
                "score": h.get("score", 0.0),
                "source": "vector",
            }
            for h in vector_hits
        ]

        # 2) Extract entities from the question and link to graph
        question_entities = self._extract_question_entities(question)
        linked_entities = self._link_to_graph(question_entities)
        logger.info("Linked entities: %s", linked_entities)

        # 3) Graph expansion — get neighboring chunks via multi-hop traversal
        graph_chunks: list[dict] = []
        if linked_entities:
            try:
                neighbors = self.graph_store.get_entity_neighbors(
                    linked_entities, hops=hops, limit=top_k_graph
                )
                for nb in neighbors:
                    graph_chunks.append(
                        {
                            "chunk_id": nb.get("chunk_id", ""),
                            "text": nb.get("text", ""),
                            "title": nb.get("title", ""),
                            "score": nb.get("score", 0.0),
                            "source": "graph",
                        }
                    )
            except Exception:
                logger.warning("Graph expansion failed", exc_info=True)

        # 4) Merge and deduplicate by chunk_id
        seen_ids: set[str] = set()
        merged: list[dict] = []
        for chunk in vector_chunks + graph_chunks:
            cid = chunk.get("chunk_id", "")
            if cid and cid in seen_ids:
                continue
            seen_ids.add(cid)
            merged.append(chunk)

        # 5) Rerank the merged set
        if merged:
            docs_for_rerank = [
                {"text": c["text"], "title": c.get("title", ""), "chunk_id": c.get("chunk_id", "")}
                for c in merged
            ]
            reranked = self.reranker.rerank(question, docs_for_rerank, top_k=top_k_final)
        else:
            reranked = []

        # 6) Gather triples for the linked entities
        triples: list[tuple] = []
        if linked_entities:
            try:
                triples = self.graph_store.get_entity_triples(linked_entities)
            except Exception:
                logger.warning("Triple fetch failed", exc_info=True)

        # 7) Build prompt
        context_str = _format_chunks(reranked)
        triples_str = _format_triples(triples)
        prompt = ANSWER_TEMPLATE.format(
            context=context_str, triples=triples_str, question=question
        )

        # 8) Generate answer
        llm_answer = self.llm.generate(prompt, system_prompt=ANSWER_SYSTEM_PROMPT)

        return {
            "answer": llm_answer.strip(),
            "retrieved_chunks": reranked,
            "graph_chunks": graph_chunks,
            "vector_chunks": vector_chunks,
            "triples": triples,
            "context": context_str,
            "question_entities": question_entities,
        }
