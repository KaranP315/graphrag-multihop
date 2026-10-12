"""Pydantic models for knowledge graph extraction.

Defines a fixed, small type system for entities and relations
to keep the graph schema consistent and queryable.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "LOCATION", "EVENT",
    "WORK", "DATE", "OTHER",
]

RELATION_TYPES = [
    "BORN_IN", "DIED_IN", "LOCATED_IN", "PART_OF", "CREATED_BY",
    "DIRECTED_BY", "WROTE", "FOUNDED", "MEMBER_OF", "CAPITAL_OF",
    "COUNTRY_OF", "MARRIED_TO", "PARENT_OF", "SIBLING_OF",
    "AWARDED", "OCCUPATION", "GENRE", "RELEASED_IN", "RELATED_TO",
]


class Entity(BaseModel):
    name: str
    type: str = Field(..., description="One of: " + ", ".join(ENTITY_TYPES))


class Relation(BaseModel):
    source: str
    relation_type: str = Field(..., description="One of: " + ", ".join(RELATION_TYPES))
    target: str
    chunk_id: str = ""  # populated after extraction


class ExtractionResult(BaseModel):
    entities: list[Entity] = []
    relations: list[Relation] = []
