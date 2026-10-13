#!/usr/bin/env python
"""
Resolve (deduplicate) extracted entities and update relations accordingly.

Reads raw entities/relations from disk, runs EntityResolver, saves the
cleaned versions, and prints before/after statistics.

Usage:
    python scripts/04_resolve_entities.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR
from src.entity_resolution import EntityResolver


def main() -> None:
    data_dir = Path(DATA_DIR)
    raw_path = data_dir / "raw_extraction.pkl"

    print("=" * 60)
    print("Step 4 — Resolve (deduplicate) entities")
    print("=" * 60)

    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run 03_extract_graph.py first.")
        sys.exit(1)

    with open(raw_path, "rb") as f:
        raw = pickle.load(f)
    entities_raw = raw["entities"]
    relations_raw = raw["relations"]

    print(f"\nBefore resolution:")
    print(f"  Entities : {len(entities_raw)}")
    print(f"  Relations: {len(relations_raw)}")

    print("\nRunning entity resolution …")
    resolver = EntityResolver()
    t0 = time.time()
    entities_resolved, relations_resolved = resolver.resolve(entities_raw, relations_raw)
    elapsed = time.time() - t0

    print(f"\nAfter resolution ({elapsed:.1f}s):")
    print(f"  Entities : {len(entities_resolved)}  (removed {len(entities_raw) - len(entities_resolved)})")
    print(f"  Relations: {len(relations_resolved)}  (removed {len(relations_raw) - len(relations_resolved)})")

    dedup_pct = (1 - len(entities_resolved) / max(len(entities_raw), 1)) * 100
    print(f"\n  Deduplication rate: {dedup_pct:.1f}%")

    # save
    resolved_path = data_dir / "resolved_extraction.pkl"
    with open(resolved_path, "wb") as f:
        pickle.dump({"entities": entities_resolved, "relations": relations_resolved}, f)
    print(f"\nSaved resolved extraction → {resolved_path}")

    print("\n✓ Entity resolution complete.")


if __name__ == "__main__":
    main()
