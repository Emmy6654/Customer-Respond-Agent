import json
import numpy as np

class RetrievalAgent:
    """
    TF-IDF backend: no torch/model download needed, so it starts fast and
    uses very little memory — important for small hosting tiers like
    Railway's free/starter plan, where a full embeddings model can cause
    an out-of-memory crash on startup.
    """
    def __init__(self, kb_path):
        from sklearn.feature_extraction.text import TfidfVectorizer

        with open(kb_path) as f:
            self.entries = json.load(f)["knowledge_base"]

        texts = [f"{e['subject']}: {e['sample_query']}" for e in self.entries]
        print("[RetrievalAgent] Fitting TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"[RetrievalAgent] Indexed {len(self.entries)} entries.")

    def retrieve(self, query, top_k=1):
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [{"subject": self.entries[i]["subject"],
                  "answer": self.entries[i]["answer"],
                  "similarity": float(sims[i])} for i in top_idx]
