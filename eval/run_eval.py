"""
Evaluation harness for comparing RAG systems on HotpotQA.

Runs each question through a RAG system, computes EM / F1 / supporting-fact
recall, and aggregates results by question type (bridge vs comparison).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.data_loader import Question
from eval.metrics import exact_match, token_f1, supporting_fact_recall

logger = logging.getLogger(__name__)


class Evaluator:
    """Batch evaluator for HotpotQA RAG systems."""

    def __init__(self, questions: list[Question]) -> None:
        self.questions = questions

    # ------------------------------------------------------------------
    # Core evaluation loop
    # ------------------------------------------------------------------

    def evaluate(
        self,
        rag_system: Any,
        system_name: str = "system",
        show_progress: bool = True,
    ) -> dict:
        """
        Run *rag_system.answer()* on every question and collect metrics.

        Returns a dict like::

            {
                "system": system_name,
                "overall": {"em": ..., "f1": ..., "sf_recall": ..., "count": ...},
                "bridge":  {"em": ..., "f1": ..., "sf_recall": ..., "count": ...},
                "comparison": {"em": ..., "f1": ..., "sf_recall": ..., "count": ...},
                "per_question": [ ... ],
                "elapsed_seconds": ...,
            }
        """
        per_q: list[dict] = []
        iterator = tqdm(self.questions, desc=system_name, disable=not show_progress)
        t0 = time.time()

        for q in iterator:
            try:
                result = rag_system.answer(q.question)
            except Exception as exc:
                logger.error("Error on question %s: %s", q.id, exc)
                result = {"answer": "", "retrieved_chunks": [], "context": ""}

            pred = result.get("answer", "")
            gold = q.answer

            em = exact_match(pred, gold)
            f1 = token_f1(pred, gold)
            sf = supporting_fact_recall(
                result.get("retrieved_chunks", []),
                q.supporting_facts,
            )

            per_q.append(
                {
                    "id": q.id,
                    "question": q.question,
                    "gold": gold,
                    "prediction": pred,
                    "type": q.type,
                    "level": q.level,
                    "em": em,
                    "f1": f1,
                    "sf_recall": sf,
                }
            )

        elapsed = time.time() - t0

        # aggregate
        overall = self._aggregate(per_q)
        bridge = self._aggregate([r for r in per_q if r["type"] == "bridge"])
        comparison = self._aggregate([r for r in per_q if r["type"] == "comparison"])

        return {
            "system": system_name,
            "overall": overall,
            "bridge": bridge,
            "comparison": comparison,
            "per_question": per_q,
            "elapsed_seconds": round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Comparison & persistence
    # ------------------------------------------------------------------

    def compare(self, results: dict[str, dict]) -> pd.DataFrame:
        """
        Build a comparison DataFrame from multiple evaluation runs.

        *results* maps system name → evaluate() output.
        """
        rows = []
        for name, res in results.items():
            for split in ("overall", "bridge", "comparison"):
                agg = res.get(split, {})
                rows.append(
                    {
                        "system": name,
                        "split": split,
                        "em": agg.get("em", 0.0),
                        "f1": agg.get("f1", 0.0),
                        "sf_recall": agg.get("sf_recall", 0.0),
                        "count": agg.get("count", 0),
                    }
                )
        df = pd.DataFrame(rows)
        return df.set_index(["system", "split"])

    @staticmethod
    def save_results(results: dict, path: Path) -> None:
        """Serialise evaluation results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # strip per_question for the summary file (it can be huge)
        slim = {k: v for k, v in results.items() if k != "per_question"}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2, default=str)
        logger.info("Results saved to %s", path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(records: list[dict]) -> dict:
        if not records:
            return {"em": 0.0, "f1": 0.0, "sf_recall": 0.0, "count": 0}
        n = len(records)
        return {
            "em": round(sum(r["em"] for r in records) / n, 4),
            "f1": round(sum(r["f1"] for r in records) / n, 4),
            "sf_recall": round(sum(r["sf_recall"] for r in records) / n, 4),
            "count": n,
        }
