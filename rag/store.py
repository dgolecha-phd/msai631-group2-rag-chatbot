"""Embeddings and a FAISS vector index."""
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config
from .ingest import Chunk


class VectorStore:
    def __init__(self, chunks: List[Chunk], model_name: str = config.EMBED_MODEL):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        # The title is added to the text so a chunk keeps its topic when embedded.
        texts = [f"{c.title}. {c.text}" for c in chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine on unit vectors
        self.index.add(vectors)

    def search(self, query: str, k: int = config.TOP_K) -> List[Tuple[Chunk, float]]:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
