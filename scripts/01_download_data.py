#!/usr/bin/env python
"""
Download the HotpotQA development set and inspect it.

Usage:
    python scripts/01_download_data.py [--sample-size N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, HOTPOTQA_SAMPLE_SIZE
from src.data_loader import HotpotQALoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preview HotpotQA data.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=HOTPOTQA_SAMPLE_SIZE,
        help="Number of questions to sample (default from config).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1 — Download HotpotQA data")
    print("=" * 60)

    loader = HotpotQALoader()
    questions = loader.load(sample_size=args.sample_size)

    print(f"\nLoaded {len(questions)} questions.")

    # quick breakdown
    types = {}
    levels = {}
    for q in questions:
        types[q.type] = types.get(q.type, 0) + 1
        levels[q.level] = levels.get(q.level, 0) + 1

    print("\nQuestion types:")
    for t, cnt in sorted(types.items()):
        print(f"  {t}: {cnt}")

    print("\nDifficulty levels:")
    for lv, cnt in sorted(levels.items()):
        print(f"  {lv}: {cnt}")

    # show a sample
    print("\n— Sample question —")
    q = questions[0]
    print(f"  ID       : {q.id}")
    print(f"  Question : {q.question}")
    print(f"  Answer   : {q.answer}")
    print(f"  Type     : {q.type}")
    print(f"  Level    : {q.level}")
    print(f"  Sup facts: {len(q.supporting_facts)} entries")
    print(f"  Context  : {len(q.context)} paragraphs")

    print("\n✓ Data download complete.")


if __name__ == "__main__":
    main()
