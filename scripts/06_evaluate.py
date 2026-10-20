#!/usr/bin/env python
"""
Run full evaluation comparing Baseline Vector RAG vs. GraphRAG vs. GraphRAG + Decomposition.

Computes:
  - Exact Match (EM)
  - Token-level F1
  - Supporting-Fact Recall (multi-hop retrieval accuracy)
  - Breakdown by question type: bridge (multi-hop) vs. comparison

Usage:
    python scripts/06_evaluate.py [--sample-size N] [--no-decomp] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    DATA_DIR,
    HOTPOTQA_SAMPLE_SIZE,
    RESULTS_DIR,
)
from src.data_loader import HotpotQALoader
from src.embeddings import EmbeddingModel
from src.llm_client import LLMClient
from src.graph_store import GraphStore
from src.vector_store import VectorStore
from src.reranker import Reranker
from src.baseline_rag import BaselineRAG
from src.graph_rag import GraphRAG
from src.decomposition import QuestionDecomposer
from eval.run_eval import Evaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DecomposedSystemWrapper:
    """Adapts QuestionDecomposer to provide the standard .answer(question) signature."""

    def __init__(self, decomposer: QuestionDecomposer, rag_system: GraphRAG) -> None:
        self.decomposer = decomposer
        self.rag = rag_system

    def answer(self, question: str) -> dict:
        return self.decomposer.answer_with_decomposition(question, self.rag)


def print_comparison_table(results_dict: dict[str, dict]) -> None:
    """Print an ASCII comparison table across systems and splits."""
    systems = list(results_dict.keys())
    splits = ["overall", "bridge", "comparison"]

    print("\n" + "=" * 84)
    print(f"{'System':<30} | {'Split':<12} | {'EM (%)':<8} | {'F1 (%)':<8} | {'SF Recall (%)':<14} | {'Count':<6}")
    print("-" * 84)

    for sys_name, res in results_dict.items():
        for split in splits:
            s_data = res.get(split, {})
            em = s_data.get("em", 0.0) * 100
            f1 = s_data.get("f1", 0.0) * 100
            sf = s_data.get("sf_recall", 0.0) * 100
            cnt = s_data.get("count", 0)
            print(f"{sys_name:<30} | {split:<12} | {em:>7.2f}% | {f1:>7.2f}% | {sf:>13.2f}% | {cnt:>6}")
        print("-" * 84)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Baseline Vector RAG vs. GraphRAG.")
    parser.add_argument("--sample-size", type=int, default=HOTPOTQA_SAMPLE_SIZE,
                        help="Number of questions to evaluate on.")
    parser.add_argument("--no-decomp", action="store_true",
                        help="Skip question decomposition evaluation to save LLM calls.")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR),
                        help="Directory to save evaluation results.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("GraphRAG Multi-Hop Evaluation Harness")
    print(f"Sample size: {args.sample_size} questions")
    print(f"Output directory: {out_dir}")
    print("=" * 84)

    # 1. Load questions
    print("\n[1/5] Loading evaluation questions …")
    loader = HotpotQALoader()
    questions = loader.load(sample_size=args.sample_size)
    print(f"  Loaded {len(questions)} evaluation questions.")

    bridge_cnt = sum(1 for q in questions if q.type == "bridge")
    comp_cnt = sum(1 for q in questions if q.type == "comparison")
    print(f"  Distribution: {bridge_cnt} bridge (multi-hop), {comp_cnt} comparison")

    # 2. Initialise components
    print("\n[2/5] Initialising models and database connections …")
    embed_model = EmbeddingModel()
    llm = LLMClient()
    reranker = Reranker()
    graph_store = GraphStore()
    vector_store = VectorStore(graph_store, embed_model)

    baseline_rag = BaselineRAG(vector_store=vector_store, llm=llm, reranker=reranker)
    graph_rag = GraphRAG(
        vector_store=vector_store,
        graph_store=graph_store,
        llm=llm,
        reranker=reranker,
        embedding_model=embed_model,
    )

    evaluator = Evaluator(questions)
    all_results: dict[str, dict] = {}

    # 3. Evaluate Baseline
    print("\n[3/5] Evaluating Baseline (Vector RAG) …")
    t0 = time.time()
    baseline_res = evaluator.evaluate(baseline_rag, system_name="Baseline Vector RAG")
    all_results["Baseline Vector RAG"] = baseline_res
    print(f"  Completed in {time.time() - t0:.1f}s")
    print(f"  Overall: EM={baseline_res['overall']['em']*100:.1f}%, F1={baseline_res['overall']['f1']*100:.1f}%, SF Recall={baseline_res['overall']['sf_recall']*100:.1f}%")

    # 4. Evaluate GraphRAG
    print("\n[4/5] Evaluating GraphRAG (Hybrid Graph + Vector) …")
    t0 = time.time()
    graph_res = evaluator.evaluate(graph_rag, system_name="GraphRAG (Hybrid)")
    all_results["GraphRAG (Hybrid)"] = graph_res
    print(f"  Completed in {time.time() - t0:.1f}s")
    print(f"  Overall: EM={graph_res['overall']['em']*100:.1f}%, F1={graph_res['overall']['f1']*100:.1f}%, SF Recall={graph_res['overall']['sf_recall']*100:.1f}%")

    # 5. Evaluate Question Decomposition (Optional extension)
    if not args.no_decomp:
        print("\n[5/5] Evaluating GraphRAG + Question Decomposition …")
        decomposer = QuestionDecomposer(llm)
        decomp_system = DecomposedSystemWrapper(decomposer, graph_rag)
        t0 = time.time()
        decomp_res = evaluator.evaluate(decomp_system, system_name="GraphRAG + Decomposition")
        all_results["GraphRAG + Decomposition"] = decomp_res
        print(f"  Completed in {time.time() - t0:.1f}s")
        print(f"  Overall: EM={decomp_res['overall']['em']*100:.1f}%, F1={decomp_res['overall']['f1']*100:.1f}%, SF Recall={decomp_res['overall']['sf_recall']*100:.1f}%")
    else:
        print("\n[5/5] Skipping question decomposition (--no-decomp specified).")

    # Display comparison
    print_comparison_table(all_results)

    # Save outputs
    results_json_path = out_dir / "evaluation_results.json"
    comparison_csv_path = out_dir / "comparison.csv"

    evaluator.save_results(all_results, results_json_path)
    df_compare = evaluator.compare(all_results)
    df_compare.to_csv(comparison_csv_path)

    print(f"\nSaved detailed summary to: {results_json_path}")
    print(f"Saved tabular comparison to: {comparison_csv_path}")

    # Calculate headline resume metrics (bridge split)
    base_b = baseline_res.get("bridge", {})
    graph_b = graph_res.get("bridge", {})

    b_f1_base = base_b.get("f1", 0.0) * 100
    b_f1_graph = graph_b.get("f1", 0.0) * 100
    b_sf_base = base_b.get("sf_recall", 0.0) * 100
    b_sf_graph = graph_b.get("sf_recall", 0.0) * 100

    sf_gain = b_sf_graph - b_sf_base

    print("\n" + "=" * 84)
    print("KEY PERFORMANCE HIGHLIGHTS & RESUME LINE")
    print("=" * 84)
    print(f"• Bridge Multi-Hop F1       : {b_f1_base:.1f}%  ->  {b_f1_graph:.1f}%  (+{b_f1_graph - b_f1_base:.1f}%)")
    print(f"• Supporting-Fact Recall    : {b_sf_base:.1f}%  ->  {b_sf_graph:.1f}%  (+{sf_gain:.1f} points)")
    print("\nReady-to-use Resume Bullet:")
    print(
        f'  "Built a GraphRAG system combining knowledge graph traversal with vector retrieval, '
        f'raising multi-hop answer F1 from {b_f1_base:.1f}% to {b_f1_graph:.1f}% and supporting-fact recall '
        f'by {sf_gain:.1f} points over a vector-only baseline on HotpotQA."'
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
