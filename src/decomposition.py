"""
Question decomposition for multi-hop reasoning.

Breaks a complex question into simpler sub-questions, answers them
sequentially while accumulating context, then synthesises a final answer
to the original question.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------- protocol so we accept either BaselineRAG or GraphRAG ----------
class RAGSystem(Protocol):
    def answer(self, question: str, **kwargs: Any) -> dict: ...


# ---------- prompt templates ----------

DECOMPOSE_SYSTEM = (
    "You are an expert at breaking complex questions into simpler sub-questions. "
    "Output valid JSON only."
)

DECOMPOSE_PROMPT = """Break the following multi-hop question into 2-3 simpler, self-contained sub-questions that, when answered in order, provide enough information to answer the original.

Question: {question}

Return a JSON object: {{"sub_questions": ["q1", "q2", ...]}}

JSON:"""

SYNTHESIS_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Use the accumulated evidence below to answer the original question. "
    "Be concise — one or two sentences."
)

SYNTHESIS_PROMPT = """Original question: {question}

Sub-questions and their answers:
{qa_pairs}

Accumulated context:
{context}

Based on all of the above, answer the original question concisely.

Answer:"""


class QuestionDecomposer:
    """Decomposes multi-hop questions and chains sub-answers."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def decompose(self, question: str) -> list[str]:
        """
        Ask the LLM to split *question* into 2–3 simpler sub-questions.
        Falls back to returning the original question if parsing fails.
        """
        prompt = DECOMPOSE_PROMPT.format(question=question)
        try:
            data = self.llm.generate_json(prompt, system_prompt=DECOMPOSE_SYSTEM)
            subs = data.get("sub_questions", [])
            if isinstance(subs, list) and len(subs) >= 1:
                return [str(s).strip() for s in subs if str(s).strip()]
        except Exception:
            logger.warning("Decomposition failed; using original question", exc_info=True)
        return [question]

    def answer_with_decomposition(
        self,
        question: str,
        rag_system: RAGSystem,
    ) -> dict:
        """
        1. Decompose the question.
        2. Answer each sub-question in order, feeding earlier answers as
           additional context to the next retrieval call.
        3. Synthesise a final answer from all accumulated evidence.
        """
        sub_questions = self.decompose(question)
        logger.info("Sub-questions: %s", sub_questions)

        sub_answers: list[str] = []
        all_context_parts: list[str] = []

        for i, sq in enumerate(sub_questions):
            # prepend prior Q/A pairs so retrieval can benefit from them
            augmented_query = sq
            if sub_answers:
                prior = " | ".join(
                    f"Q: {sub_questions[j]} A: {sub_answers[j]}"
                    for j in range(len(sub_answers))
                )
                augmented_query = f"{sq} (Background: {prior})"

            result = rag_system.answer(augmented_query)
            sub_answers.append(result["answer"])
            all_context_parts.append(result.get("context", ""))

        # build the Q/A pairs string for the synthesis prompt
        qa_lines = []
        for sq, sa in zip(sub_questions, sub_answers):
            qa_lines.append(f"Q: {sq}\nA: {sa}")
        qa_str = "\n\n".join(qa_lines)

        # keep accumulated context manageable — last 4000 chars max
        full_context = "\n\n---\n\n".join(all_context_parts)
        if len(full_context) > 4000:
            full_context = full_context[-4000:]

        synthesis_prompt = SYNTHESIS_PROMPT.format(
            question=question, qa_pairs=qa_str, context=full_context
        )
        final_answer = self.llm.generate(
            synthesis_prompt, system_prompt=SYNTHESIS_SYSTEM
        )

        return {
            "answer": final_answer.strip(),
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
            "final_context": full_context,
        }
