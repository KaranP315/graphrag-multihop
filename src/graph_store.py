"""Neo4j graph store — CRUD for entities, relations, and chunk nodes.

Handles graph construction, vector index creation, and multi-hop
neighbourhood retrieval used by the GraphRAG retriever.
"""
from __future__ import annotations

import logging
from typing import Any

import neo4j
from tqdm import tqdm

from config import EMBEDDING_DIM, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from src.schemas import Entity, Relation

logger = logging.getLogger(__name__)


class GraphStore:
    """Thin wrapper around Neo4j for knowledge graph operations."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self.driver = neo4j.GraphDatabase.driver(
            uri or NEO4J_URI,
            auth=(user or NEO4J_USER, password or NEO4J_PASSWORD),
        )
        # quick connectivity check
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri or NEO4J_URI)

    def close(self) -> None:
        self.driver.close()

    # ------------------------------------------------------------------
    # graph CRUD
    # ------------------------------------------------------------------

    def clear_graph(self) -> None:
        """Wipe every node and relationship in the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    def create_entity(self, name: str, entity_type: str) -> None:
        """MERGE an :Entity node keyed on name + type."""
        cypher = """
        MERGE (e:Entity {name: $name, type: $type})
        """
        with self.driver.session() as session:
            session.run(cypher, name=name, type=entity_type)

    def create_relation(
        self,
        source: str,
        target: str,
        rel_type: str,
        chunk_id: str,
    ) -> None:
        """Create a typed edge between two entities, carrying the source chunk_id."""
        # Neo4j relationship types can't be parameterised,
        # so we store rel_type as a property and use a generic label.
        cypher = """
        MATCH (a:Entity {name: $source})
        MATCH (b:Entity {name: $target})
        MERGE (a)-[r:RELATES_TO {rel_type: $rel_type}]->(b)
        ON CREATE SET r.chunk_id = $chunk_id
        """
        with self.driver.session() as session:
            session.run(
                cypher,
                source=source,
                target=target,
                rel_type=rel_type,
                chunk_id=chunk_id,
            )

    def create_chunk_node(
        self,
        chunk_id: str,
        text: str,
        title: str,
        embedding: list[float],
    ) -> None:
        """Create a :Chunk node with an embedding vector."""
        cypher = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.text = $text,
            c.title = $title,
            c.embedding = $embedding
        """
        with self.driver.session() as session:
            session.run(
                cypher,
                chunk_id=chunk_id,
                text=text,
                title=title,
                embedding=embedding,
            )

    def link_chunk_to_entities(
        self,
        chunk_id: str,
        entity_names: list[str],
    ) -> None:
        """Create (:Chunk)-[:MENTIONS]->(:Entity) edges."""
        cypher = """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        MATCH (e:Entity) WHERE e.name IN $names
        MERGE (c)-[:MENTIONS]->(e)
        """
        with self.driver.session() as session:
            session.run(cypher, chunk_id=chunk_id, names=entity_names)

    # ------------------------------------------------------------------
    # vector index
    # ------------------------------------------------------------------

    def create_vector_index(self) -> None:
        """Create a cosine-similarity vector index on Chunk.embedding."""
        cypher = """
        CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: $dim,
            `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(cypher, dim=EMBEDDING_DIM)
        logger.info(
            "Vector index created (dim=%d, cosine)", EMBEDDING_DIM
        )

    # ------------------------------------------------------------------
    # bulk build
    # ------------------------------------------------------------------

    def build_graph(
        self,
        entities: list[Entity],
        relations: list[Relation],
        chunks: list[Any],
        embeddings: list[list[float]],
    ) -> None:
        """Orchestrate full graph construction from extracted data.

        Args:
            entities: Deduplicated entity list.
            relations: Relation list (with chunk_ids).
            chunks: Original Chunk objects.
            embeddings: Parallel list of embedding vectors for each chunk.
        """
        logger.info("Building graph: %d entities, %d relations, %d chunks",
                     len(entities), len(relations), len(chunks))

        # 1. entities
        for ent in tqdm(entities, desc="Creating entities"):
            self.create_entity(ent.name, ent.type)

        # 2. relations
        for rel in tqdm(relations, desc="Creating relations"):
            self.create_relation(rel.source, rel.target, rel.relation_type, rel.chunk_id)

        # 3. chunk nodes
        for chunk, emb in tqdm(
            zip(chunks, embeddings), total=len(chunks), desc="Creating chunk nodes"
        ):
            self.create_chunk_node(chunk.id, chunk.text, chunk.title, emb)

        # 4. link chunks to entities they mention
        # pre-compute which entity names appear in each chunk's relations
        chunk_entity_map: dict[str, set[str]] = {}
        for rel in relations:
            chunk_entity_map.setdefault(rel.chunk_id, set()).update(
                [rel.source, rel.target]
            )
        for chunk_id, names in tqdm(
            chunk_entity_map.items(), desc="Linking chunks to entities"
        ):
            self.link_chunk_to_entities(chunk_id, list(names))

        # 5. vector index
        self.create_vector_index()

        logger.info("Graph build complete")

    # ------------------------------------------------------------------
    # retrieval queries
    # ------------------------------------------------------------------

    def get_entity_neighbors(
        self,
        entity_names: list[str],
        hops: int = 2,
        limit: int = 20,
    ) -> list[dict]:
        """Multi-hop neighbourhood retrieval around named entities.

        Expands N hops from each entity, collects the chunks that mention
        entities along those paths.

        Returns list of dicts with keys: chunk_id, text, paths_count.
        """
        cypher = f"""
        MATCH (e:Entity) WHERE e.name IN $names
        MATCH path = (e)-[:RELATES_TO*0..{hops}]-(neighbor:Entity)
        WITH neighbor, count(path) AS paths_count
        MATCH (c:Chunk)-[:MENTIONS]->(neighbor)
        RETURN c.chunk_id AS chunk_id,
               c.text AS text,
               c.title AS title,
               sum(paths_count) AS paths_count
        ORDER BY paths_count DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, names=entity_names, limit=limit)
            return [dict(record) for record in result]

    def get_entity_triples(
        self,
        entity_names: list[str],
        hops: int = 1,
    ) -> list[tuple[str, str, str]]:
        """Return (source, relation_type, target) triples around the entities."""
        cypher = f"""
        MATCH (e:Entity) WHERE e.name IN $names
        MATCH path = (e)-[r:RELATES_TO*1..{hops}]-(other:Entity)
        UNWIND relationships(path) AS rel
        WITH startNode(rel) AS src, rel, endNode(rel) AS tgt
        RETURN DISTINCT src.name AS source,
               rel.rel_type AS relation_type,
               tgt.name AS target
        """
        with self.driver.session() as session:
            result = session.run(cypher, names=entity_names)
            return [(r["source"], r["relation_type"], r["target"]) for r in result]

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        """Return basic graph statistics."""
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            edges = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS cnt"
            ).single()["cnt"]
            entities = session.run(
                "MATCH (e:Entity) RETURN count(e) AS cnt"
            ).single()["cnt"]
            chunks = session.run(
                "MATCH (c:Chunk) RETURN count(c) AS cnt"
            ).single()["cnt"]

        return {
            "total_nodes": nodes,
            "total_edges": edges,
            "entities": entities,
            "chunks": chunks,
        }

    def find_matching_entities(self, query_name: str, limit: int = 5) -> list[str]:
        """Find entity names matching query_name (case-insensitive exact or substring)."""
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower($name) OR toLower(e.name) CONTAINS toLower($name)
        RETURN e.name AS name,
               CASE WHEN toLower(e.name) = toLower($name) THEN 2 ELSE 1 END AS score
        ORDER BY score DESC, length(e.name) ASC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, name=query_name, limit=limit)
            return [record["name"] for record in result]
