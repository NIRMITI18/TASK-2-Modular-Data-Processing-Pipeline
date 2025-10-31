"""Vector store abstraction supporting FAISS when available, otherwise
an in-memory fallback with disk persistence.
"""
from typing import List, Dict
import os
import json
import numpy as np


class VectorStore:
    def __init__(self, index_dir: str = "index"):
        self.index_dir = index_dir
        self.vectors = []
        self.metadatas = []
        self.ids = []
        self._use_faiss = False
        try:
            import faiss  # type: ignore
            self.faiss = faiss
            self._use_faiss = True
            self._index = None
        except Exception:
            self.faiss = None
            self._index = None

    def add(self, id: str, embedding: List[float], metadata: Dict):
        self.ids.append(id)
        self.vectors.append(np.array(embedding, dtype=np.float32))
        self.metadatas.append(metadata)

    def save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        arr = np.vstack(self.vectors) if self.vectors else np.zeros((0, 0), dtype=np.float32)
        np.save(os.path.join(self.index_dir, "vectors.npy"), arr)
        with open(os.path.join(self.index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"ids": self.ids, "metadatas": self.metadatas}, f, ensure_ascii=False)

        if self._use_faiss and arr.size:
            d = arr.shape[1]
            index = self.faiss.IndexFlatIP(d)
            # normalize vectors for cosine similarity
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr_norm = arr / norms
            index.add(arr_norm.astype('float32'))
            self.faiss.write_index(index, os.path.join(self.index_dir, "faiss.idx"))

    def _load(self):
        vec_path = os.path.join(self.index_dir, "vectors.npy")
        meta_path = os.path.join(self.index_dir, "meta.json")
        if os.path.exists(vec_path):
            arr = np.load(vec_path)
            self.vectors = [arr[i] for i in range(arr.shape[0])]
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
                self.ids = meta.get("ids", [])
                self.metadatas = meta.get("metadatas", [])

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        # if faiss index exists, try using it
        if self._use_faiss:
            try:
                idx_path = os.path.join(self.index_dir, "faiss.idx")
                if os.path.exists(idx_path):
                    index = self.faiss.read_index(idx_path)
                    q = np.array(query_embedding, dtype=np.float32)
                    q = q / (np.linalg.norm(q) or 1.0)
                    D, I = index.search(np.expand_dims(q, 0), k)
                    out = []
                    for score, i in zip(D[0], I[0]):
                        if i < 0 or i >= len(self.metadatas):
                            continue
                        out.append({"id": self.ids[i], "score": float(score), "metadata": self.metadatas[i]})
                    return out
            except Exception:
                pass

        # fallback brute-force cosine
        if not self.vectors:
            self._load()
        if not self.vectors:
            return []

        mat = np.vstack(self.vectors)
        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = (mat @ q) / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0))
        idxs = np.argsort(-sims)[:k]
        out = []
        for i in idxs:
            out.append({"id": self.ids[int(i)], "score": float(sims[int(i)]), "metadata": self.metadatas[int(i)]})
        return out
