#!/usr/bin/env python
"""
Extract entities and relations from chunks using the LLM.

Reads chunks from the pickle produced by 02_build_baseline.py,
runs GraphExtractor, and saves raw (un-resolved) entities and
relations to disk.

Usage:
    python scripts/03_extract_graph.py [--batch-size N]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR
from src.extraction import GraphExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract KG triples from chunks.")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Chunks per LLM extraction call.")
    args = parser.parse_args()

    data_dir = Path(DATA_DIR)
    chunks_path = data_dir / "chunks.pkl"

    print("=" * 60)
    print("Step 3 — Extract entities and relations")
    print("=" * 60)

    # load chunks
    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run 02_build_baseline.py first.")
        sys.exit(1)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"\nLoaded {len(chunks)} chunks.")

    # extract
    print("\nRunning graph extraction (this will take a while) …")
    extractor = GraphExtractor()
    t0 = time.time()
    entities, relations = extractor.extract_from_chunks(chunks)
    elapsed = time.time() - t0

    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"  Raw entities : {len(entities)}")
    print(f"  Raw relations: {len(relations)}")

    # type distributions
    ent_types = Counter(e.type for e in entities)
    rel_types = Counter(r.relation_type for r in relations)

    print("\nEntity type distribution:")
    for t, cnt in ent_types.most_common(15):
        print(f"  {t:30s} {cnt}")

    print("\nRelation type distribution:")
    for t, cnt in rel_types.most_common(15):
        print(f"  {t:30s} {cnt}")

    # save
    raw_path = data_dir / "raw_extraction.pkl"
    with open(raw_path, "wb") as f:
        pickle.dump({"entities": entities, "relations": relations}, f)
    print(f"\nSaved raw extraction → {raw_path}")

    print("\n✓ Graph extraction complete.")


if __name__ == "__main__":
    main()
