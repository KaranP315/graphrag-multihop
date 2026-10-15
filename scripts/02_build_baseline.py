#!/usr/bin/env python
"""
Build the baseline vector index.

1. Load HotpotQA questions and chunk all context paragraphs.
2. Compute embeddings for each chunk.
3. Store chunks + embeddings as nodes in Neo4j.
4. Create a vector search index.
5. Persist chunks to a pickle file for downstream scripts.

Usage:
    python scripts/02_build_baseline.py [--sample-size N]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR,
    HOTPOTQA_SAMPLE_SIZE,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
)
from src.data_loader import HotpotQALoader
from src.chunker import Chunker
from src.embeddings import EmbeddingModel
from src.graph_store import GraphStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the baseline vector store.")
    parser.add_argument("--sample-size", type=int, default=HOTPOTQA_SAMPLE_SIZE)
    args = parser.parse_args()

    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 2 — Build baseline vector index")
    print("=" * 60)

    # --- load questions ---
    print("\n[1/5] Loading HotpotQA data …")
    loader = HotpotQALoader()
    questions = loader.load(sample_size=args.sample_size)
    print(f"  Loaded {len(questions)} questions.")

    # --- chunk paragraphs ---
    print("\n[2/5] Chunking paragraphs …")
    chunker = Chunker()

    # gather unique paragraphs across all questions
    all_paragraphs: list[dict] = []
    seen_titles: set[str] = set()
    for q in questions:
        for para in q.context:
            title = para.get("title", para[0] if isinstance(para, (list, tuple)) else "")
            if isinstance(para, (list, tuple)):
                title = para[0]
                sentences = para[1] if len(para) > 1 else []
                para = {"title": title, "sentences": sentences}
            if title not in seen_titles:
                seen_titles.add(title)
                all_paragraphs.append(para)

    chunks = chunker.chunk_paragraphs(all_paragraphs)
    print(f"  Created {len(chunks)} chunks from {len(all_paragraphs)} paragraphs.")

    # --- compute embeddings ---
    print("\n[3/5] Computing embeddings (this may take a few minutes) …")
    embed_model = EmbeddingModel()
    t0 = time.time()
    texts = [c.text for c in chunks]

    batch_size = 64
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings_list.append(embed_model.encode(batch))
        if (i // batch_size) % 10 == 0:
            print(f"    embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks …")

    embeddings = np.vstack(embeddings_list)
    elapsed = time.time() - t0
    print(f"  Embeddings shape: {embeddings.shape}  ({elapsed:.1f}s)")

    # --- store in Neo4j ---
    print("\n[4/5] Loading chunks into Neo4j …")
    graph_store = GraphStore()
    graph_store.build_graph(
        chunks=chunks,
        embeddings=embeddings,
        entities=[],
        relations=[],
    )
    graph_store.create_vector_index()
    stats = graph_store.get_stats()
    print(f"  Graph stats: {stats}")

    # --- persist chunks and embeddings ---
    print("\n[5/5] Saving chunks and embeddings to disk …")
    chunks_path = data_dir / "chunks.pkl"
    embeddings_path = data_dir / "embeddings.npy"

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    np.save(embeddings_path, embeddings)

    print(f"  Chunks   → {chunks_path}")
    print(f"  Embeddings → {embeddings_path}")

    print("\n✓ Baseline vector index built successfully.")


if __name__ == "__main__":
    main()
