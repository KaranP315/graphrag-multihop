"""
Evaluation metrics for HotpotQA-style question answering.

Provides exact-match, token-level F1, and supporting-fact recall — the
standard metrics used by the HotpotQA leaderboard.
"""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """
    Lowercase, strip articles / punctuation / extra whitespace.
    This is the standard HotpotQA normalization used in the official eval script.
    """
    text = text.lower()
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if the normalized prediction matches the gold answer, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """
    Token-level F1 between the normalized prediction and gold answer.
    Returns 0.0 when either side is empty after normalization.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    # tokens in common (with multiplicity)
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def supporting_fact_recall(
    retrieved_chunks: list[dict],
    supporting_facts: list[tuple],
    paragraphs: list[dict] | None = None,
) -> float:
    """
    What fraction of ground-truth supporting-fact *titles* appear among the
    retrieved chunks?

    Parameters
    ----------
    retrieved_chunks : list[dict]
        Each dict should have a ``"title"`` key.
    supporting_facts : list[tuple]
        Each tuple is ``(title, sent_idx)`` from the HotpotQA dataset.
    paragraphs : list[dict] | None
        Unused — kept for API compatibility if callers pass it.

    Returns
    -------
    float  in [0.0, 1.0]
    """
    if not supporting_facts:
        return 1.0  # nothing to recall

    # unique gold titles
    gold_titles = {title.lower().strip() for title, _ in supporting_facts}
    retrieved_titles = {
        c.get("title", "").lower().strip() for c in retrieved_chunks
    }

    if not gold_titles:
        return 1.0

    hits = gold_titles & retrieved_titles
    return len(hits) / len(gold_titles)
