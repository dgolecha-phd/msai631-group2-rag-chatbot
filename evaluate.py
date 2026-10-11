"""Check retrieval quality on a small question set. Run: python evaluate.py"""
import json
import sys
from pathlib import Path

from rag import config
from rag.ingest import load_chunks
from rag.store import VectorStore

QUESTIONS = json.loads((Path(__file__).parent / "tests" / "questions.json").read_text())


def main() -> int:
    store = VectorStore(load_chunks())
    answerable = [q for q in QUESTIONS if q["expected"]]
    hit1 = hit3 = 0
    for q in answerable:
        hits = store.search(q["question"], k=3)
        docs = [c.doc_id for c, _ in hits]
        hit1 += docs[0] == q["expected"]
        hit3 += q["expected"] in docs
        flag = "ok " if docs[0] == q["expected"] else ("top3" if q["expected"] in docs else "MISS")
        print(f"[{flag}] {q['question']}  ->  {docs[0]} ({hits[0][1]:.2f})")

    refused = 0
    off_topic = [q for q in QUESTIONS if not q["expected"]]
    for q in off_topic:
        best = store.search(q["question"], k=1)[0][1]
        ok = best < config.MIN_SCORE
        refused += ok
        print(f"[{'ok ' if ok else 'FAIL'}] off-topic: {q['question']}  (best score {best:.2f})")

    n = len(answerable)
    print(f"\nTop-1 hit rate: {hit1}/{n}   Top-3 hit rate: {hit3}/{n}   "
          f"Off-topic refused: {refused}/{len(off_topic)}")
    return 0 if hit3 == n and refused == len(off_topic) else 1


if __name__ == "__main__":
    sys.exit(main())
