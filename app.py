"""Gradio chat interface for the Group 2 study-support chatbot.

Run locally:   python app.py
Then open the address printed in the terminal (usually http://127.0.0.1:7860).
"""
import json
import time

import gradio as gr

from rag import config
from rag.pipeline import RagPipeline

print("Loading models. The first run downloads them and can take a few minutes...")
PIPELINE = RagPipeline()
print(f"Ready. Mode: {PIPELINE.mode}")

EXAMPLES = [
    "How much is the final project worth?",
    "What is the late work policy?",
    "What is the difference between mean and median?",
    "Can I use AI tools on my assignments?",
]


def format_sources(sources):
    if not sources:
        return "No sources used."
    lines = [f"{i}. {s['title']} (`{s['doc_id']}`, match {s['score']})" for i, s in enumerate(sources, 1)]
    return "**Sources**\n" + "\n".join(lines)


def respond(message, history, last):
    history = history or []
    if not (message or "").strip():
        return "", history, format_sources([]), last
    result = PIPELINE.ask(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.answer},
    ]
    last = {"question": message, "answer": result.answer, "answered": result.answered}
    return "", history, format_sources(result.sources), last


def save_feedback(vote, last):
    if not last:
        return "Ask a question first, then rate the answer."
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "vote": vote, **last}
    with open(config.FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return "Thank you. Your feedback was saved."


with gr.Blocks(title="Course Study Helper") as demo:
    gr.Markdown(
        "# Course Study Helper\n"
        "Ask about the syllabus, assignments, study tips, or basic concepts. "
        "Answers come only from a small set of sample course documents, and every answer "
        "lists its sources. If the answer is not in the documents, the helper will say so."
    )
    chat = gr.Chatbot(label="Conversation", height=380)
    box = gr.Textbox(label="Your question", placeholder="Type a question and press Enter")
    with gr.Row():
        send = gr.Button("Ask", variant="primary")
        up = gr.Button("Helpful")
        down = gr.Button("Not helpful")
    sources_md = gr.Markdown("No sources yet.")
    status = gr.Markdown("")
    last = gr.State(None)
    gr.Examples(EXAMPLES, inputs=box, label="Try one of these")

    for trigger in (box.submit, send.click):
        trigger(respond, [box, chat, last], [box, chat, sources_md, last])
    up.click(lambda s: save_feedback("up", s), last, status)
    down.click(lambda s: save_feedback("down", s), last, status)

if __name__ == "__main__":
    demo.launch()
