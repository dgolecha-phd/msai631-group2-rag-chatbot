"""The full RAG pipeline: retrieve, check confidence, then answer."""
from dataclasses import dataclass, field
from typing import List

from . import config
from .ingest import Chunk, load_chunks
from .store import VectorStore

NO_ANSWER = (
    "I could not find that in the course materials I have, so I would rather not guess. "
    "Try rewording the question, or ask your instructor during office hours."
)


@dataclass
class Result:
    answer: str
    sources: List[dict] = field(default_factory=list)  # title, doc_id, score
    answered: bool = True


class RagPipeline:
    def __init__(self, mode: str = config.MODE):
        self.mode = mode
        self.store = VectorStore(load_chunks())
        self.generator = None
        if mode == "generate":
            try:
                from .generate import Generator
                self.generator = Generator()
            except Exception as exc:  # no internet, not enough memory, and so on
                print(f"[warning] Language model could not load ({exc}). Using extractive mode.")
                self.mode = "extractive"

    def ask(self, question: str) -> Result:
        question = (question or "").strip()
        if not question:
            return Result("Please type a question.", answered=False)
        hits = self.store.search(question)
        hits = [(c, s) for c, s in hits if s >= config.MIN_SCORE]
        if not hits:
            return Result(NO_ANSWER, answered=False)
        # Keep only passages close to the best match, so weak matches do not confuse the answer.
        best = hits[0][1]
        hits = [(c, s) for c, s in hits if s >= best - config.SCORE_WINDOW]

        chunks = [c for c, _ in hits]
        sources = [{"title": c.title, "doc_id": c.doc_id, "score": round(s, 2)} for c, s in hits]
        if self.mode == "generate" and self.generator is not None:
            text = self.generator.answer(question, chunks)
            if not text:
                text = chunks[0].text
        else:
            text = chunks[0].text
        return Result(text, sources, True)
