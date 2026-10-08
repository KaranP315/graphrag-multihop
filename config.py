"""
Central configuration — reads from .env or falls back to sane defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Neo4j ────────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag123")

# ── LLM (any OpenAI-compatible endpoint) ─────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Embeddings ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ── Reranker ──────────────────────────────────────────────────────────────
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Retrieval knobs ──────────────────────────────────────────────────────
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "20"))
TOP_K_GRAPH = int(os.getenv("TOP_K_GRAPH", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "10"))
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "2"))

# ── Data ──────────────────────────────────────────────────────────────────
HOTPOTQA_SAMPLE_SIZE = int(os.getenv("HOTPOTQA_SAMPLE_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))  # paragraphs are short, overlap not needed
