#!/usr/bin/env python
"""
Load resolved entities, relations, chunks, and embeddings into Neo4j
to build the full knowledge graph.

Usage:
    python scripts/05_load_graph.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR
from src.graph_store import GraphStore


def main() -> None:
    data_dir = Path(DATA_DIR)

    print("=" * 60)
    print("Step 5 — Load full knowledge graph into Neo4j")
    print("=" * 60)

    # load all artifacts
    chunks_path = data_dir / "chunks.pkl"
    embeddings_path = data_dir / "embeddings.npy"
    resolved_path = data_dir / "resolved_extraction.pkl"

    missing = [p for p in (chunks_path, embeddings_path, resolved_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"  ERROR: missing {p}")
        print("Run the previous pipeline steps first.")
        sys.exit(1)

    print("\n[1/3] Loading artifacts from disk …")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(embeddings_path)
    with open(resolved_path, "rb") as f:
        resolved = pickle.load(f)
    entities = resolved["entities"]
    relations = resolved["relations"]

    print(f"  Chunks    : {len(chunks)}")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Entities  : {len(entities)}")
    print(f"  Relations : {len(relations)}")

    # build the graph
    print("\n[2/3] Building graph in Neo4j …")
    graph_store = GraphStore()
    t0 = time.time()
    graph_store.build_graph(
        chunks=chunks,
        embeddings=embeddings,
        entities=entities,
        relations=relations,
    )
    graph_store.create_vector_index()
    elapsed = time.time() - t0
    print(f"  Graph built in {elapsed:.1f}s")

    # stats
    print("\n[3/3] Graph statistics:")
    stats = graph_store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ Knowledge graph loaded successfully.")


if __name__ == "__main__":
    main()
