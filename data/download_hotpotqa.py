"""Download the HotpotQA distractor-setting dev set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "hotpot_dev_distractor_v1.json"


def download_hotpotqa(url: str = HOTPOTQA_URL, dest: Path = OUTPUT_PATH) -> Path:
    """Stream-download the HotpotQA JSON file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading HotpotQA dev set from:\n  {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="hotpotqa",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"Saved to {dest}")
    return dest


def main() -> None:
    dest = download_hotpotqa()

    # quick sanity check
    with open(dest, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions from the dev set.")


if __name__ == "__main__":
    main()
