"""Run with: python -m unittest discover -s tests -v

Uses extractive mode so the tests are fast and do not need the language model.
"""
import json
import os
import sys
import unittest
from pathlib import Path

os.environ["RAG_MODE"] = "extractive"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingest import load_chunks, split_words  # noqa: E402
from rag.pipeline import NO_ANSWER, RagPipeline  # noqa: E402

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text())


class TestIngest(unittest.TestCase):
    def test_documents_load(self):
        chunks = load_chunks()
        self.assertGreaterEqual(len({c.doc_id for c in chunks}), 10)
        self.assertTrue(all(c.text.strip() for c in chunks))

    def test_split_words_overlap(self):
        text = " ".join(str(i) for i in range(100))
        pieces = split_words(text, 40, 10)
        self.assertEqual(pieces[0].split()[0], "0")
        self.assertTrue(set(pieces[0].split()) & set(pieces[1].split()))
        self.assertEqual(pieces[-1].split()[-1], "99")


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = RagPipeline(mode="extractive")

    def test_empty_question(self):
        self.assertFalse(self.pipe.ask("   ").answered)

    def test_answerable_questions_find_the_right_document_in_top_three(self):
        for q in (x for x in QUESTIONS if x["expected"]):
            hits = self.pipe.store.search(q["question"], k=3)
            self.assertIn(q["expected"], [c.doc_id for c, _ in hits], q["question"])

    def test_off_topic_questions_are_refused(self):
        for q in (x for x in QUESTIONS if not x["expected"]):
            r = self.pipe.ask(q["question"])
            self.assertFalse(r.answered, q["question"])
            self.assertEqual(r.answer, NO_ANSWER)

    def test_answers_include_sources(self):
        r = self.pipe.ask("How much is the final project worth?")
        self.assertTrue(r.answered)
        self.assertEqual(r.sources[0]["doc_id"], "04_final_project.md")


if __name__ == "__main__":
    unittest.main()
