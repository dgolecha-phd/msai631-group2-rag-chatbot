"""Settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", ROOT / "data" / "docs"))
FEEDBACK_FILE = Path(os.getenv("RAG_FEEDBACK_FILE", ROOT / "feedback.jsonl"))

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Small open model that runs on a laptop CPU (about 3 GB of memory). Use Qwen/Qwen2.5-0.5B-Instruct if memory is very tight (answers are less reliable).
GEN_MODEL = os.getenv("RAG_GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# "generate" = retrieve, then write an answer with the language model.
# "extractive" = retrieve only and show the best passage (fast, no language model).
MODE = os.getenv("RAG_MODE", "generate")

CHUNK_WORDS = int(os.getenv("RAG_CHUNK_WORDS", "110"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "25"))
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
# Below this cosine similarity the bot says it does not know instead of guessing.
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.18"))
SCORE_WINDOW = float(os.getenv("RAG_SCORE_WINDOW", "0.12"))
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "220"))
