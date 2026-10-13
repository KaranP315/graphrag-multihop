"""Entity resolution — merge duplicate entities via normalization and embeddings.

Two-pass approach:
  1. Exact match on normalized names (lowercase, stripped punctuation).
  2. Embedding-based fuzzy match within each entity-type group.
"""
from __future__ import annotations

import logging
import re
import string
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from src.schemas import Entity, Relation

if TYPE_CHECKING:
    from src.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class EntityResolver:
    """Deduplicates entities and rewires relations to canonical names."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        if embedding_model is None:
            from src.embeddings import EmbeddingModel
            embedding_model = EmbeddingModel()
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_name(name: str) -> str:
        """Lowercase, strip, remove punctuation, collapse whitespace."""
        name = name.lower().strip()
        name = name.translate(str.maketrans("", "", string.punctuation))
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def resolve(
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> tuple[list[Entity], list[Relation]]:
        """Deduplicate entities and update relation endpoints accordingly.

        Returns the pruned entity list and the updated relations.
        """
        # group by type so we only compare apples to apples
        type_groups: dict[str, list[Entity]] = defaultdict(list)
        for ent in entities:
            type_groups[ent.type].append(ent)

        # full name mapping (original name -> canonical name)
        name_map: dict[str, str] = {}

        for etype, group in type_groups.items():
            # --- pass 1: exact normalized match ---
            norm_to_canonical: dict[str, str] = {}
            for ent in group:
                norm = self.normalize_name(ent.name)
                if norm not in norm_to_canonical:
                    norm_to_canonical[norm] = ent.name  # first seen wins
                name_map[ent.name] = norm_to_canonical[norm]

            # --- pass 2: embedding-based fuzzy merge ---
            unique_canonical = list(set(norm_to_canonical.values()))
            if len(unique_canonical) > 1:
                merge_map = self._merge_groups(unique_canonical, self.similarity_threshold)
                # update name_map through the chain
                for original, canonical in list(name_map.items()):
                    if canonical in merge_map:
                        name_map[original] = merge_map[canonical]

        # build deduplicated entity list
        seen: set[str] = set()
        deduped_entities: list[Entity] = []
        for ent in entities:
            canonical = name_map.get(ent.name, ent.name)
            if canonical not in seen:
                seen.add(canonical)
                deduped_entities.append(Entity(name=canonical, type=ent.type))

        # rewrite relation endpoints
        updated_relations: list[Relation] = []
        for rel in relations:
            updated_relations.append(
                Relation(
                    source=name_map.get(rel.source, rel.source),
                    relation_type=rel.relation_type,
                    target=name_map.get(rel.target, rel.target),
                    chunk_id=rel.chunk_id,
                )
            )

        logger.info(
            "Entity resolution: %d -> %d entities",
            len(entities),
            len(deduped_entities),
        )
        return deduped_entities, updated_relations

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    def _merge_groups(
        self,
        entities: list[str],
        threshold: float,
    ) -> dict[str, str]:
        """Find embedding-similar names and pick a canonical representative.

        Returns a mapping from every merged name to its canonical form.
        """
        if len(entities) <= 1:
            return {}

        embeddings = self.embedding_model.encode(entities)  # (N, dim)
        # cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        # union-find style grouping: greedily merge into the first match
        canonical: dict[str, str] = {}
        for i in range(len(entities)):
            if entities[i] in canonical:
                continue
            for j in range(i + 1, len(entities)):
                if entities[j] in canonical:
                    continue
                if sim_matrix[i, j] >= threshold:
                    # prefer the shorter / more common name as canonical
                    canonical[entities[j]] = entities[i]

        return canonical
