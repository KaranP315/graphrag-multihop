"""
Plain vector-search RAG baseline.

Retrieves chunks via embedding similarity, optionally reranks with a
cross-encoder, then feeds the top passages to the LLM for answer generation.
"""
from __future__ import annotations

from src.vector_store import VectorStore
from src.llm_client import LLMClient
from src.reranker import Reranker

ANSWER_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the provided context passages. "
    "If the context does not contain enough information, say so honestly. "
    "Keep your answer concise — ideally one or two sentences."
)

ANSWER_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer:"""


def _format_chunks(chunks: list[dict]) -> str:
    """Number and concatenate retrieved passages for the prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Untitled")
        text = c.get("text", "").strip()
        parts.append(f"[{i}] (Title: {title})\n{text}")
    return "\n\n".join(parts)


class BaselineRAG:
    """Vector-only retrieval-augmented generation."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMClient,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker

    def answer(self, question: str, top_k: int = 10) -> dict:
        """
        Retrieve context via vector search, optionally rerank, and generate
        an answer with the LLM.

        Returns a dict with keys: answer, retrieved_chunks, context.
        """
        # pull more candidates when reranking so the cross-encoder has room
        fetch_k = top_k * 3 if self.reranker else top_k
        raw_hits = self.vector_store.search(question, top_k=fetch_k)

        if self.reranker and raw_hits:
            documents = [
                {"text": h["text"], "title": h.get("title", ""), "chunk_id": h.get("chunk_id", "")}
                for h in raw_hits
            ]
            reranked = self.reranker.rerank(question, documents, top_k=top_k)
            retrieved = reranked
        else:
            retrieved = raw_hits[:top_k]

        context_str = _format_chunks(retrieved)

        prompt = ANSWER_TEMPLATE.format(context=context_str, question=question)
        llm_answer = self.llm.generate(prompt, system_prompt=ANSWER_SYSTEM_PROMPT)

        return {
            "answer": llm_answer.strip(),
            "retrieved_chunks": retrieved,
            "context": context_str,
        }
