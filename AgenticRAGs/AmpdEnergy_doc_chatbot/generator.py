# generator.py
from transformers import pipeline

class LocalGenerator:
    def __init__(self, model_name="google/flan-t5-small", device=-1):
        # device=-1: CPU. If you have GPU, set device=0
        self.model_name = model_name
        self.generator = pipeline("text2text-generation", model=model_name, device=device, truncation=True)

    def answer(self, question, contexts, max_new_tokens=256):
        """
        contexts: list of dicts {page, text, image_path}
        Build a prompt using contexts and ask the model.
        """
        context_text = "\n\n".join([f"[Page {c['page']}] {c['text']}" for c in contexts])
        prompt = f"""You are a helpful technical assistant. Use ONLY the context provided to answer the question. If the answer is not in the context, say "I don't know; see pages referenced below" and list the pages.
Context:
{context_text}

Question: {question}

Answer concisely and cite pages when possible (format: [Page X]).
"""
        # HuggingFace text2text pipeline needs small prompts sometimes; keep prompt under token limits
        result = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return result[0]["generated_text"]
