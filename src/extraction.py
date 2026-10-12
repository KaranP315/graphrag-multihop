"""LLM-based entity and relation extraction from text chunks.

Uses structured prompting to coerce the LLM into returning JSON that
conforms to our fixed schema (ENTITY_TYPES / RELATION_TYPES).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tqdm import tqdm

from src.schemas import (
    ENTITY_TYPES,
    RELATION_TYPES,
    Entity,
    ExtractionResult,
    Relation,
)

if TYPE_CHECKING:
    from src.chunker import Chunk
    from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Extract entities and relations from the text. "
    "Return valid JSON with 'entities' (list of {{name, type}}) "
    "and 'relations' (list of {{source, relation_type, target}}). "
    f"Use ONLY these entity types: {ENTITY_TYPES}. "
    f"Use ONLY these relation types: {RELATION_TYPES}. "
    "If unsure, use OTHER for entity type or RELATED_TO for relation type."
)

_USER_PROMPT_TEMPLATE = """Extract all entities and relations from the following text.

TEXT:
{text}

Return a JSON object with exactly two keys:
- "entities": a list of objects, each with "name" (string) and "type" (one of {entity_types})
- "relations": a list of objects, each with "source" (string), "relation_type" (one of {relation_types}), and "target" (string)

If no entities or relations are found, return empty lists.
JSON:"""


class GraphExtractor:
    """Extracts a knowledge graph from text chunks via an LLM."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        if llm is None:
            from src.llm_client import LLMClient
            llm = LLMClient()
        self.llm = llm

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionResult:
        """Run extraction on a single chunk and return validated results."""
        prompt = _USER_PROMPT_TEMPLATE.format(
            text=chunk.text,
            entity_types=", ".join(ENTITY_TYPES),
            relation_types=", ".join(RELATION_TYPES),
        )

        raw = self.llm.generate_json(prompt, system_prompt=_SYSTEM_PROMPT)
        result = ExtractionResult.model_validate(raw)

        # stamp every relation with the source chunk id
        for rel in result.relations:
            rel.chunk_id = chunk.id

        return result

    def extract_from_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from multiple chunks.

        Failed chunks are skipped with a warning rather than blowing up
        the whole pipeline.
        """
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        iterator = tqdm(chunks, desc="Extracting graphs", disable=not show_progress)
        for chunk in iterator:
            try:
                result = self.extract_from_chunk(chunk)
                all_entities.extend(result.entities)
                all_relations.extend(result.relations)
            except Exception as exc:
                logger.warning(
                    "Extraction failed for chunk %s: %s — skipping",
                    chunk.id,
                    exc,
                )
                continue

        logger.info(
            "Extracted %d entities and %d relations from %d chunks",
            len(all_entities),
            len(all_relations),
            len(chunks),
        )
        return all_entities, all_relations
