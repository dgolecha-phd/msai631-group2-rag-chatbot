# Course Study Helper: a small RAG chatbot (MSAI 631, Group 2)

A retrieval-augmented generation (RAG) chatbot that answers student questions about a course
(syllabus details, assignment clarifications, study tips, and basic concepts). Answers are
written only from a small knowledge base of 14 short documents, and every answer lists the
documents it used. If the answer is not in the documents, the bot says so instead of guessing.

The 14 documents in `data/docs/` are **sample documents written for this project** for a made-up
course ("SAMPLE 101"). They are not real course materials. Replace them with real course
documents (plain `.md` files) to use the bot for a real course.

## How it works

1. `rag/ingest.py` reads every `.md` file and splits it into chunks of about 110 words.
2. `rag/store.py` turns each chunk into an embedding with `all-MiniLM-L6-v2` and stores it in a FAISS index.
3. `rag/pipeline.py` embeds the question, finds the 3 closest chunks, and refuses to answer if the best match is too weak.
4. `rag/generate.py` gives the matching chunks to a small open model (`Qwen2.5-1.5B-Instruct`) that writes a short answer with source numbers.
5. `app.py` shows the chat in a Gradio page with the sources, and Helpful / Not helpful buttons that save to `feedback.jsonl`.

No paid API, no fine-tuning, and no API keys are needed.

## Run it

Requires Python 3.10 or newer and about 4 GB of free memory. The first run downloads two models (about 3.5 GB in total).

```bash
git clone <this repository's URL>
cd <repository folder>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the address printed in the terminal (usually http://127.0.0.1:7860).
Answers take roughly 10 to 30 seconds on a laptop CPU.

If memory is very tight, use the smaller model (answers are less reliable, see Results):

```bash
RAG_GEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct python app.py
```

If no language model can load (for example, no internet), the app prints a warning and falls back to
showing the best matching passage. You can also force this fast mode with `RAG_MODE=extractive`.

## Test it

```bash
python -m unittest discover -s tests -v   # 6 quick tests, no language model needed
python evaluate.py                        # retrieval check on 18 test questions
```

## Results from our own testing

| Check | Result |
|---|---|
| Right document in top 3 results (16 answerable test questions) | 16 of 16 |
| Right document ranked first | 14 of 16 |
| Off-topic questions refused (2 questions) | 2 of 2 |
| 6 hand-checked questions, 1.5B model | 6 of 6 correct |
| Smaller 0.5B model on 6 hand-checked questions (a different set that included an off-topic one) | 2 answers clearly wrong: it said a partner is allowed on the final project, and it gave contradictory answers about ChatGPT |

These numbers come from a small test set that we wrote ourselves, using documents we also wrote.
They show the pipeline works. They do not prove it will be this accurate on real course documents.

## Known limits

- The 1.5B model is small. It can still misread a passage, so the sources are shown with every answer.
- The refusal threshold (`RAG_MIN_SCORE`, 0.18) was set by hand from our test questions.
- The bot has no memory of earlier questions in a conversation. Each question is answered on its own.
- The feedback buttons only save votes to `feedback.jsonl`. Nothing reads them yet.

## Deploy to a Hugging Face Space (free tier)

1. Create a new Space on Hugging Face, choose the Gradio SDK, and the free CPU hardware.
2. Push this repository's files to the Space (or mirror the GitHub repository to it).
3. Add the line `sdk: gradio` and `app_file: app.py` to a README header as the Space asks.
4. The free CPU may be slow with the 1.5B model. If so, set the variable `RAG_GEN_MODEL` to `Qwen/Qwen2.5-0.5B-Instruct` in the Space settings.

The Space has not been deployed yet.

## Project layout

```
app.py              Gradio interface
evaluate.py         retrieval quality check
rag/                config, ingest, store, generate, pipeline
data/docs/          14 sample knowledge-base documents
tests/              unit tests and the test question set
requirements.txt
```

## Team

| Member | Role in the proposal | Name |
|---|---|---|
| A | Design, model selection, RAG pipeline, deployment, integration | (Devshree) |
| B | Document ingestion, chunking, embeddings, vector store, retrieval testing | (fill in) |
| C | Gradio interface, prompts, citations, feedback, README, slides | (fill in) |
