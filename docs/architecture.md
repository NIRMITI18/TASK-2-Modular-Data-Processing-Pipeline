Architecture and design choices

This project demonstrates a small, modular pipeline for extracting, cleaning,
structuring, embedding, and storing unstructured textual data. The pipeline is
split into clear components so each concern can be extended or replaced without
affecting the rest of the system.

Components
- Loader: handles file-type specific extraction (CSV, TXT, PDF, HTML). It
  returns a uniform list of items of the form {source, text}. The loader is
  directory-aware so it can process many files in a batch.
- Cleaner: applies deterministic, dependency-light normalizations (whitespace
  collapsing, removal of non-printable characters, snippet extraction). This
  keeps text predictable for downstream embedding and indexing.
- EmbeddingModel: an abstraction over text embeddings. It prefers
  sentence-transformers for high-quality dense vectors, and falls back to a
  reproducible TF-IDF + SVD pipeline if that library isn't available. This
  provides a sensible accuracy vs. dependency tradeoff for demos and testing.
- VectorStore: pluggable storage for embeddings. If FAISS is present the
  implementation persists a FAISS index for efficient nearest-neighbor search.
  Otherwise it falls back to an in-memory matrix with persisted numpy/json
  artifacts and a brute-force cosine search. This ensures functionality even
  in restricted environments.

Why these tools
 - SentenceTransformers: balances performance and quality for semantic
   embeddings and is widely used in research and production.
 - FAISS: provides a fast, memory-efficient nearest-neighbor backend when
   available. The code is written to still function when FAISS can't be
   installed, which simplifies testing and portability.

Scalability considerations
 - Modularity: each component is a focused module, so you can scale or swap
   parts independently (for example, replace the Loader to add S3 or database
   sources, or swap embeddings with an external service).
 - Batch and streaming: the loader design supports directory-level batching.
   For larger datasets you'd add chunked reads, async IO, and streaming writes
   to avoid holding all data in memory.
 - Vector indexing: for production scale (millions of vectors) use an
   approximate nearest neighbor index (FAISS HNSW or IVF+PQ) stored on disk
   and sharded across machines. Also consider incremental indexing rather than
   rebuilding.

How to run
 - Install the optional deps listed in requirements.txt when you want better
   embeddings and FAISS persistence. The code runs with minimal dependencies.
 - Run the pipeline with `python main.py data/example_raw.csv` to produce
   `data/cleaned_output.json` and a persisted index.
