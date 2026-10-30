import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import INDEX_PATH, CHUNKS_PATH

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class KnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks: list[dict] = []
        self.index = None
        self._load()

    def _load(self):
        if INDEX_PATH.exists() and CHUNKS_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(INDEX_PATH))
            CHUNKS_PATH.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_document(self, filename: str, texts: list[str]) -> int:
        if not texts:
            return 0
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend({"text": t, "filename": filename, "chunk_id": i} for i, t in enumerate(texts))
        self._save()
        return len(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None or not self.chunks:
            return []
        vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(vector, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = self.chunks[int(idx)].copy()
                item["score"] = float(score)
                results.append(item)
        return results

    def list_documents(self) -> list[dict]:
        counts = {}
        for chunk in self.chunks:
            counts[chunk["filename"]] = counts.get(chunk["filename"], 0) + 1
        return [{"filename": name, "chunks": count} for name, count in sorted(counts.items())]

kb = KnowledgeBase()
