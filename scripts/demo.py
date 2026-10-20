#!/usr/bin/env python
"""
Interactive CLI demo for GraphRAG.

Compare Baseline Vector RAG against GraphRAG on multi-hop questions in real time.
Shows retrieved chunks, traversed graph triples, and final answers.

Supports offline / mock simulation mode via `--mock` so you can test immediately
without needing active Neo4j credentials or paid LLM tokens.

Usage:
    python scripts/demo.py [--mock] [--mode {both,baseline,graphrag}] [--question "Your question"]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_QUESTIONS = [
    "Where was the director of the film Inception born?",
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What government position was held by the author of the novella that inspired Apocalypse Now?",
    "Which award did the composer of the musical Hamilton win in 2016?",
]

MOCK_KNOWLEDGE_BASE = {
    "Where was the director of the film Inception born?": {
        "baseline": {
            "answer": "The provided context mentions Inception is a 2010 film directed by Christopher Nolan, but does not state where he was born.",
            "retrieved_chunks": [
                {"title": "Inception (film)", "text": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan."}
            ],
            "latency": 0.82
        },
        "graphrag": {
            "entities": ["Inception", "director"],
            "linked": ["Inception", "Christopher Nolan"],
            "triples": [
                ("Inception", "DIRECTED_BY", "Christopher Nolan"),
                ("Christopher Nolan", "BORN_IN", "Westminster, London"),
                ("Christopher Nolan", "OCCUPATION", "Film Director"),
                ("Inception", "RELEASED_IN", "2010")
            ],
            "vector_chunks": [{"title": "Inception (film)", "text": "Inception is a 2010 film directed by Christopher Nolan."}],
            "graph_chunks": [{"title": "Christopher Nolan", "text": "Christopher Edward Nolan was born in Westminster, London on 30 July 1970."}],
            "answer": "Christopher Nolan, the director of Inception, was born in Westminster, London.",
            "latency": 1.74
        },
        "sub_questions": [
            "Who is the director of the film Inception?",
            "Where was Christopher Nolan born?"
        ]
    },
    "Were Scott Derrickson and Ed Wood of the same nationality?": {
        "baseline": {
            "answer": "Scott Derrickson is an American filmmaker. Information regarding Ed Wood's nationality is missing from the retrieved context.",
            "retrieved_chunks": [
                {"title": "Scott Derrickson", "text": "Scott Derrickson is an American director and screenwriter known for Doctor Strange."}
            ],
            "latency": 0.79
        },
        "graphrag": {
            "entities": ["Scott Derrickson", "Ed Wood"],
            "linked": ["Scott Derrickson", "Ed Wood"],
            "triples": [
                ("Scott Derrickson", "COUNTRY_OF", "United States"),
                ("Ed Wood", "COUNTRY_OF", "United States"),
                ("Scott Derrickson", "OCCUPATION", "Filmmaker"),
                ("Ed Wood", "OCCUPATION", "Filmmaker")
            ],
            "vector_chunks": [{"title": "Scott Derrickson", "text": "Scott Derrickson is an American director."}],
            "graph_chunks": [{"title": "Ed Wood", "text": "Edward Davis Wood Jr. was an American filmmaker and actor."}],
            "answer": "Yes, both Scott Derrickson and Ed Wood were of American nationality.",
            "latency": 1.68
        },
        "sub_questions": [
            "What is the nationality of Scott Derrickson?",
            "What is the nationality of Ed Wood?"
        ]
    }
}


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


def run_mock_query(question: str, mode: str = "both", with_decomp: bool = False) -> None:
    print("\n" + "=" * 80)
    print(f"❓ QUESTION: \033[1m{question}\033[0m")
    print("=" * 80)

    data = MOCK_KNOWLEDGE_BASE.get(question)
    if not data:
        # Generic mock fallback
        data = {
            "baseline": {
                "answer": "The baseline vector search could not find connecting passages across multiple documents.",
                "retrieved_chunks": [{"title": "Passage 1", "text": "Contains initial entity reference."}],
                "latency": 0.85
            },
            "graphrag": {
                "entities": ["Query Entity 1", "Target Relation"],
                "linked": ["Query Entity 1"],
                "triples": [("Entity 1", "RELATES_TO", "Entity 2"), ("Entity 2", "LOCATED_IN", "Result")],
                "vector_chunks": [{"title": "Passage 1", "text": "Reference to Entity 1"}],
                "graph_chunks": [{"title": "Passage 2", "text": "Evidence for Entity 2 and Result"}],
                "answer": "Graph traversal connected Entity 1 to Entity 2 via 2-hop BFS, retrieving the ground-truth answer.",
                "latency": 1.82
            },
            "sub_questions": ["Identify the intermediate entity", "Resolve target attribute"]
        }

    # 1. Baseline
    if mode in ("both", "baseline"):
        print("\n\033[1;36m▶ 1. BASELINE (Plain Vector RAG)\033[0m")
        time.sleep(0.4)
        b = data["baseline"]
        print(f"  ⏱  Latency: {b['latency']}s")
        print(f"  📄 Retrieved {len(b['retrieved_chunks'])} passages: {[c['title'] for c in b['retrieved_chunks']]}")
        print(f"  💬 Answer: \033[31m{b['answer']}\033[0m")

    # 2. GraphRAG
    if mode in ("both", "graphrag"):
        print("\n\033[1;35m▶ 2. GRAPHRAG (Hybrid Graph + Vector)\033[0m")
        time.sleep(0.6)
        g = data["graphrag"]
        print(f"  ⏱  Latency: {g['latency']}s")
        print(f"  🔍 Question Entities: {g['entities']}  --> Linked: {g['linked']}")
        print(f"  🌐 Vector chunks: {len(g['vector_chunks'])} | Graph-expanded chunks: {len(g['graph_chunks'])}")
        print(f"  🔗 Traversed Triples ({len(g['triples'])} total):")
        format_triples_box(g["triples"])
        print(f"\n  💬 Answer: \033[1;32m{g['answer']}\033[0m")

    # 3. Optional Decomposition
    if with_decomp and mode == "both":
        print("\n\033[1;33m▶ 3. QUESTION DECOMPOSITION BREAKDOWN\033[0m")
        for i, sq in enumerate(data.get("sub_questions", []), 1):
            print(f"     [{i}] {sq}")

    print("\n" + "-" * 80)


def run_live_query(
    question: str,
    baseline_rag,
    graph_rag,
    decomposer,
    mode: str = "both",
    hops: int = 2,
) -> None:
    print("\n" + "=" * 80)
    print(f"❓ QUESTION: \033[1m{question}\033[0m")
    print("=" * 80)

    # 1. Baseline
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

    # 3. Decomposition
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
    parser.add_argument("--mock", action="store_true",
                        help="Run in offline simulation mode without live Neo4j/OpenAI.")
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

    is_mock = args.mock
    baseline_rag = None
    graph_rag = None
    decomposer = None

    if not is_mock:
        print("Initialising models & database connections...")
        try:
            from src.embeddings import EmbeddingModel
            from src.llm_client import LLMClient
            from src.reranker import Reranker
            from src.graph_store import GraphStore
            from src.vector_store import VectorStore
            from src.baseline_rag import BaselineRAG
            from src.graph_rag import GraphRAG
            from src.decomposition import QuestionDecomposer

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
            print("✓ All live components ready!\n")
        except Exception as e:
            print(f"\n⚠️  Live Neo4j or API key connection not active: {e}")
            print("👉 Falling back to offline Simulation Mode (no Neo4j/OpenAI required).\n")
            is_mock = True

    if is_mock:
        print("\033[1;33m[Running in Offline Simulation Mode — HotpotQA Pre-indexed Graph]\033[0m\n")

    # Single question mode
    if args.question:
        if is_mock:
            run_mock_query(args.question, mode=args.mode, with_decomp=args.with_decomp)
        else:
            run_live_query(
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

            if is_mock:
                run_mock_query(q_text, mode=args.mode, with_decomp=args.with_decomp)
            else:
                run_live_query(
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
