# retriever.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FaissRetriever:
    def __init__(self, index_path="index.faiss", meta_path="meta.json", embed_model="all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf8") as f:
            self.meta = json.load(f)
        self.embedder = SentenceTransformer(embed_model)

    def _embed_query(self, query):
        v = self.embedder.encode([query], convert_to_numpy=True)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v.astype('float32')

    def retrieve(self, query, top_k=10, rerank_k=5):
        q = self._embed_query(query)
        D, I = self.index.search(q, top_k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            meta = self.meta[idx] 
            hits.append({"score": float(score), "id": meta["id"], "page": meta["page"], "text": meta["text"], "image_paths": meta.get("image_paths")})
        # Basic reranking: compute actual cosine between query and original text embeddings (recompute to be safe)
        # We'll re-embed top_k texts with the embedder (faster: reuse stored embeddings if available; here we recompute)
        texts = [h["text"] for h in hits]
        text_embeds = self.embedder.encode(texts, convert_to_numpy=True)
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, text_embeds)[0]
        for i, s in enumerate(sims):
            hits[i]["rerank_score"] = float(s)
        hits_sorted = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
        return hits_sorted[:rerank_k]
