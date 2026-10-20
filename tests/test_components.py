"""Unit tests for GraphRAG components (runs without external API keys or Neo4j)."""
import unittest

from src.chunker import Chunker
from src.schemas import Entity, Relation, ExtractionResult, ENTITY_TYPES, RELATION_TYPES
from src.entity_resolution import EntityResolver
from eval.metrics import normalize_answer, exact_match, token_f1, supporting_fact_recall


class TestChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = Chunker()
        self.sample_paragraphs = [
            {
                "title": "Inception",
                "sentences": [
                    "Inception is a 2010 science fiction film written and directed by Christopher Nolan.",
                    "The film stars Leonardo DiCaprio as a professional thief."
                ],
                "doc_id": "inception"
            },
            {
                "title": "Christopher Nolan",
                "sentences": [
                    "Christopher Edward Nolan is a British-American film director.",
                    "He was born in Westminster, London."
                ],
                "doc_id": "christopher_nolan"
            }
        ]

    def test_chunk_paragraphs(self):
        chunks = self.chunker.chunk_paragraphs(self.sample_paragraphs)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].id, "inception_chunk_0")
        self.assertEqual(chunks[0].title, "Inception")
        self.assertIn("Leonardo DiCaprio", chunks[0].text)


class TestSchemas(unittest.TestCase):
    def test_valid_extraction_result(self):
        data = {
            "entities": [
                {"name": "Christopher Nolan", "type": "PERSON"},
                {"name": "Inception", "type": "WORK"},
                {"name": "London", "type": "LOCATION"}
            ],
            "relations": [
                {"source": "Inception", "relation_type": "DIRECTED_BY", "target": "Christopher Nolan"},
                {"source": "Christopher Nolan", "relation_type": "BORN_IN", "target": "London"}
            ]
        }
        result = ExtractionResult.model_validate(data)
        self.assertEqual(len(result.entities), 3)
        self.assertEqual(len(result.relations), 2)
        self.assertEqual(result.relations[0].relation_type, "DIRECTED_BY")


class TestEntityResolution(unittest.TestCase):
    def test_lexical_normalization(self):
        self.assertEqual(EntityResolver.normalize_name("  Christopher Nolan! "), "christopher nolan")
        self.assertEqual(EntityResolver.normalize_name("London, U.K."), "london uk")

    def test_exact_resolution_and_rewiring(self):
        # Dummy resolver without embedding model (tests lexical pass)
        class DummyResolver(EntityResolver):
            def __init__(self):
                self.similarity_threshold = 0.85
            def _merge_groups(self, names, threshold):
                return {}

        resolver = DummyResolver()
        entities = [
            Entity(name="Christopher Nolan", type="PERSON"),
            Entity(name="christopher nolan", type="PERSON"),
            Entity(name="Inception", type="WORK")
        ]
        relations = [
            Relation(source="Inception", relation_type="DIRECTED_BY", target="christopher nolan", chunk_id="c1")
        ]

        deduped_ents, updated_rels = resolver.resolve(entities, relations)
        self.assertEqual(len(deduped_ents), 2)
        self.assertEqual(updated_rels[0].target, "Christopher Nolan")


class TestEvaluationMetrics(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_answer("The United States of America"), "united states of america")
        self.assertEqual(normalize_answer("London, UK."), "london uk")

    def test_exact_match(self):
        self.assertEqual(exact_match("Christopher Nolan", "christopher nolan"), 1.0)
        self.assertEqual(exact_match("London", "Paris"), 0.0)

    def test_token_f1(self):
        # Exact match
        self.assertEqual(token_f1("Westminster London", "Westminster London"), 1.0)
        # Partial overlap (1 token common out of 2 and 3)
        f1 = token_f1("London UK", "Westminster London England")
        self.assertAlmostEqual(f1, 2 * (1/2) * (1/3) / ((1/2) + (1/3)), places=4)
        # No overlap
        self.assertEqual(token_f1("Apple", "Banana"), 0.0)

    def test_supporting_fact_recall(self):
        retrieved_chunks = [
            {"title": "Inception", "text": "..."},
            {"title": "Christopher Nolan", "text": "..."}
        ]
        gold_facts = [
            ("Inception", 0),
            ("Christopher Nolan", 1)
        ]
        recall = supporting_fact_recall(retrieved_chunks, gold_facts)
        self.assertEqual(recall, 1.0)

        # Missing one fact
        gold_facts_with_missing = [
            ("Inception", 0),
            ("Christopher Nolan", 1),
            ("Leonardo DiCaprio", 0)
        ]
        recall_partial = supporting_fact_recall(retrieved_chunks, gold_facts_with_missing)
        self.assertAlmostEqual(recall_partial, 2 / 3, places=4)


if __name__ == "__main__":
    unittest.main()
