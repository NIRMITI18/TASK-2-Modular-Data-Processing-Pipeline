"""Embedding model abstraction.

Attempts to use sentence-transformers. If unavailable, falls back to a
TF-IDF + SVD-based dense representation (deterministic, lighter-weight).
"""
from typing import List
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._use_sbert = False
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name)
            self._use_sbert = True
        except Exception:
            # fallback
            self.encoder = None
            self._use_sklearn = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._tfidf = TfidfVectorizer(max_features=2000)
            # initialize svd lazily because n_components depends on data shape
            self._svd = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._use_sbert:
            embs = self.encoder.encode(texts, show_progress_bar=False)
            return [list(map(float, e)) for e in np.array(embs)]

        # sklearn fallback
        tf = self._tfidf.fit_transform(texts)
        n_features = tf.shape[1]
        n_samples = tf.shape[0]
        n_components = min(128, max(1, n_features), max(1, n_samples - 1))
        from sklearn.decomposition import TruncatedSVD
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = self._svd.fit_transform(tf)
        # normalize to unit length
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        reduced = reduced / norms
        return [list(map(float, r)) for r in reduced]
