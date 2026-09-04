import json
import numpy as np
from sentence_transformers import SentenceTransformer

class RetrievalAgent:
    def __init__(self, kb_path, model_name="all-MiniLM-L6-v2"):
        with open(kb_path) as f:
            self.entries = json.load(f)["knowledge_base"]
        print(f"[RetrievalAgent] Loading '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        texts = [f"{e['subject']}: {e['sample_query']}" for e in self.entries]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        print(f"[RetrievalAgent] Indexed {len(self.entries)} entries.")

    def retrieve(self, query, top_k=1):
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ query_emb
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [{"subject": self.entries[i]["subject"],
                  "answer": self.entries[i]["answer"],
                  "similarity": float(sims[i])} for i in top_idx]
