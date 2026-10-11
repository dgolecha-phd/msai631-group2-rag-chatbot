"""Answer writer. Uses a small open language model, with a safe fallback."""
from typing import List

from . import config
from .ingest import Chunk

SYSTEM_PROMPT = (
    "You are a study-support assistant for university students. Answer the question "
    "using ONLY the numbered sources. Keep the answer to two to four short sentences. "
    "Put the source number in square brackets after each fact, like [1]. "
    "If the sources do not contain the answer, say you do not know."
)


class Generator:
    def __init__(self, model_name: str = config.GEN_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model.eval()

    def answer(self, question: str, chunks: List[Chunk]) -> str:
        sources = "\n\n".join(f"[{i}] {c.title}: {c.text}" for i, c in enumerate(chunks, 1))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
