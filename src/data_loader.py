from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import DATA_DIR


@dataclass
class Question:
    """A single HotpotQA question with its context and supporting facts."""

    id: str
    question: str
    answer: str
    type: str  # "bridge" or "comparison"
    level: str  # "easy", "medium", or "hard"
    supporting_facts: list[tuple[str, int]]  # (title, sentence_idx)
    context: list[tuple[str, list[str]]]  # (title, list of sentences)


class HotpotQALoader:
    """Loads and samples questions from the HotpotQA distractor-setting dev set."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        if data_path is None:
            self.data_path = Path(DATA_DIR) / "hotpot_dev_distractor_v1.json"
        else:
            self.data_path = Path(data_path)

    def load(self, sample_size: int = 200) -> list[Question]:
        """Load the JSON file and return a random sample of Question objects.

        Uses a fixed seed so results are reproducible across runs.
        """
        if not self.data_path.exists():
            from data.download_hotpotqa import download_hotpotqa
            print(f"Dataset not found at {self.data_path}. Downloading...")
            download_hotpotqa(dest=self.data_path)

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # deterministic sampling
        rng = random.Random(42)
        sampled = rng.sample(raw_data, min(sample_size, len(raw_data)))

        questions: list[Question] = []
        for item in sampled:
            # Parse supporting facts (handles both CMU raw list and HF dict formats)
            raw_sf = item.get("supporting_facts", [])
            if isinstance(raw_sf, dict):
                sup_facts = list(zip(raw_sf.get("title", []), raw_sf.get("sent_id", [])))
            elif isinstance(raw_sf, list):
                sup_facts = [(fact[0], int(fact[1])) for fact in raw_sf if len(fact) >= 2]
            else:
                sup_facts = []

            # Parse context (handles both CMU raw list and HF dict formats)
            raw_ctx = item.get("context", [])
            if isinstance(raw_ctx, dict):
                ctx = list(zip(raw_ctx.get("title", []), raw_ctx.get("sentences", [])))
            elif isinstance(raw_ctx, list):
                ctx = [(entry[0], entry[1]) for entry in raw_ctx if len(entry) >= 2]
            else:
                ctx = []

            q = Question(
                id=str(item.get("_id", item.get("id", ""))),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                type=item.get("type", "bridge"),
                level=item.get("level", "medium"),
                supporting_facts=sup_facts,
                context=ctx,
            )
            questions.append(q)

        return questions

    def get_all_paragraphs(self, questions: list[Question]) -> list[dict]:
        """Flatten question contexts into unique paragraphs.

        Returns a list of dicts with keys: title, sentences, doc_id.
        Deduplicates by title so each paragraph appears only once.
        """
        seen_titles: set[str] = set()
        paragraphs: list[dict] = []

        for q in questions:
            for title, sentences in q.context:
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                # build a stable doc id from the title
                doc_id = title.lower().replace(" ", "_").replace("/", "_")
                paragraphs.append({
                    "title": title,
                    "sentences": sentences,
                    "doc_id": doc_id,
                })

        return paragraphs
