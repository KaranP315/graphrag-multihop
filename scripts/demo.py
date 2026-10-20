#!/usr/bin/env python
"""
Interactive CLI demo for GraphRAG.

Compare Baseline Vector RAG against GraphRAG on multi-hop questions in real time.
Shows retrieved chunks, traversed graph triples, and final answers.

Usage:
    python scripts/demo.py [--mode {both,baseline,graphrag}] [--question "Your question"]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingModel
from src.llm_client import LLMClient
from src.graph_store import GraphStore
from src.vector_store import VectorStore
from src.reranker import Reranker
from src.baseline_rag import BaselineRAG
from src.graph_rag import GraphRAG
from src.decomposition import QuestionDecomposer

SAMPLE_QUESTIONS = [
    "Where was the director of the film Inception born?",
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What government position was held by the author of the novella that inspired Apocalypse Now?",
    "Which award did the composer of the musical Hamilton win in 2016?",
]


def print_banner() -> None:
    print(r"""
   ____                 _     ____      _     ____ 
  / ___|_ __ __ _ _ __ | |__ |  _ \    / \   / ___|
 | |  _| '__/ _` | '_ \| '_ \| |_) |  / _ \ | |  _ 
 | |_| | | | (_| | |_) | | | |  _ <  / ___ \| |_| |
  \____|_|  \__,_| .__/|_| |_|_| \_\/_/   \_\\____|
                 |_|                               
      Multi-Hop Knowledge-Graph Augmented Retrieval
    """)


def format_triples_box(triples: list[tuple]) -> None:
    if not triples:
        print("  (no graph triples found)")
        return
    for src, rel, tgt in triples[:15]:
        print(f"  • \033[94m{src}\033[0m  ──[\033[93m{rel}\033[0m]──▶  \033[92m{tgt}\033[0m")
    if len(triples) > 15:
        print(f"  ... and {len(triples) - 15} more triples.")


def run_query(
    question: str,
    baseline_rag: BaselineRAG | None,
    graph_rag: GraphRAG | None,
    decomposer: QuestionDecomposer | None,
    mode: str = "both",
    hops: int = 2,
) -> None:
    print("\n" + "=" * 80)
    print(f"❓ QUESTION: \033[1m{question}\033[0m")
    print("=" * 80)

    # 1. Baseline Vector RAG
    if mode in ("both", "baseline") and baseline_rag:
        print("\n\033[1;36m▶ 1. BASELINE (Plain Vector RAG)\033[0m")
        t0 = time.time()
        try:
            b_res = baseline_rag.answer(question)
            t_base = time.time() - t0
            print(f"  ⏱  Latency: {t_base:.2f}s")
            print(f"  📄 Retrieved {len(b_res.get('retrieved_chunks', []))} passages")
            print(f"  💬 Answer: \033[32m{b_res.get('answer', '')}\033[0m")
        except Exception as e:
            print(f"  ❌ Error running baseline: {e}")

    # 2. GraphRAG
    if mode in ("both", "graphrag") and graph_rag:
        print("\n\033[1;35m▶ 2. GRAPHRAG (Hybrid Graph + Vector)\033[0m")
        t0 = time.time()
        try:
            g_res = graph_rag.answer(question, hops=hops)
            t_graph = time.time() - t0
            print(f"  ⏱  Latency: {t_graph:.2f}s")
            print(f"  🔍 Question Entities: {g_res.get('question_entities', [])}")
            print(f"  🌐 Vector chunks: {len(g_res.get('vector_chunks', []))} | Graph-expanded chunks: {len(g_res.get('graph_chunks', []))}")
            print(f"  🔗 Traversed Triples ({len(g_res.get('triples', []))} total):")
            format_triples_box(g_res.get("triples", []))
            print(f"\n  💬 Answer: \033[1;32m{g_res.get('answer', '')}\033[0m")
        except Exception as e:
            print(f"  ❌ Error running GraphRAG: {e}")

    # 3. Optional Decomposition
    if decomposer and mode == "both":
        print("\n\033[1;33m▶ 3. QUESTION DECOMPOSITION BREAKDOWN\033[0m")
        try:
            sub_qs = decomposer.decompose(question)
            print("  🧩 Decomposed into sub-questions:")
            for i, sq in enumerate(sub_qs, 1):
                print(f"     [{i}] {sq}")
        except Exception as e:
            print(f"  ❌ Decomposition error: {e}")

    print("\n" + "-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive GraphRAG Demo.")
    parser.add_argument("--mode", choices=["both", "baseline", "graphrag"], default="both",
                        help="Which RAG systems to execute.")
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to query immediately.")
    parser.add_argument("--hops", type=int, default=2,
                        help="Graph expansion radius.")
    parser.add_argument("--with-decomp", action="store_true",
                        help="Show question decomposition steps.")
    args = parser.parse_args()

    print_banner()

    print("Initialising models & database connections...")
    try:
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
        decomposer = QuestionDecomposer(llm) if args.with_decomp else None
        print("✓ All components ready!\n")
    except Exception as e:
        print(f"\n⚠️  Connection or initialization warning: {e}")
        print("Note: If Neo4j is not currently running, start it with: docker compose up -d\n")
        return

    # Single question mode
    if args.question:
        run_query(
            question=args.question,
            baseline_rag=baseline_rag,
            graph_rag=graph_rag,
            decomposer=decomposer,
            mode=args.mode,
            hops=args.hops,
        )
        return

    # Interactive loop
    print("Pre-loaded HotpotQA sample questions:")
    for idx, sample in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"  [{idx}] {sample}")
    print("  [0] Custom question input")
    print("  [q] Quit\n")

    while True:
        try:
            choice = input("\nSelect a question number (or enter custom text, 'q' to quit) > ").strip()
            if not choice or choice.lower() in ("q", "quit", "exit"):
                print("Exiting demo. Bye!")
                break

            if choice.isdigit():
                num = int(choice)
                if 1 <= num <= len(SAMPLE_QUESTIONS):
                    q_text = SAMPLE_QUESTIONS[num - 1]
                elif num == 0:
                    q_text = input("Enter your custom multi-hop question: ").strip()
                    if not q_text:
                        continue
                else:
                    print("Invalid selection.")
                    continue
            else:
                q_text = choice

            run_query(
                question=q_text,
                baseline_rag=baseline_rag,
                graph_rag=graph_rag,
                decomposer=decomposer,
                mode=args.mode,
                hops=args.hops,
            )
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
