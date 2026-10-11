"""Load the knowledge-base documents and split them into small chunks."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

from . import config


@dataclass
class Chunk:
    doc_id: str      # file name, for example 02_grading_and_late_policy.md
    title: str       # first heading of the document
    text: str        # the chunk text
    index: int       # position of the chunk inside its document


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return fallback


def split_words(text: str, size: int, overlap: int) -> List[str]:
    """Split text into windows of about `size` words that overlap by `overlap` words."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(1, size - overlap)
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return pieces


def load_chunks(docs_dir: Path = config.DOCS_DIR) -> List[Chunk]:
    files = sorted(Path(docs_dir).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md documents found in {docs_dir}")
    chunks: List[Chunk] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        title = _title(text, path.stem)
        for i, piece in enumerate(split_words(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP)):
            chunks.append(Chunk(doc_id=path.name, title=title, text=piece, index=i))
    return chunks
