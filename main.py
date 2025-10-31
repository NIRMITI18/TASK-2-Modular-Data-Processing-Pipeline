"""
Modular Text Processing Pipeline
--------------------------------
This script demonstrates a scalable backend pipeline for:
- Extracting text from PDFs and images
- Cleaning and chunking text
- Generating embeddings
- Storing them in a FAISS vector index for semantic retrieval
"""

"""Modular data processing pipeline (entry point).

Run as: python main.py

This script demonstrates a pipeline that loads raw data (CSV, PDF, HTML),
cleans and structures text, computes embeddings, and stores vectors in a
pluggable vector store (FAISS or fallback in-memory).

It is intentionally dependency-tolerant: it will try to use faiss/chromadb
if available, otherwise fall back to a simple in-memory index.
"""
import argparse
from pipeline.loader import Loader
from pipeline.cleaner import Cleaner
from pipeline.embeddings import EmbeddingModel
from pipeline.vectorstore import VectorStore
import json
import os


def run_pipeline(input_path: str, out_json: str, index_dir: str = "index"):
	os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
	os.makedirs(index_dir, exist_ok=True)

	print(f"Loading data from {input_path}...")
	loader = Loader()
	items = loader.load(input_path)

	print(f"Loaded {len(items)} raw items. Cleaning...")
	cleaner = Cleaner()
	cleaned = [cleaner.clean_text(it) for it in items]

	# Save cleaned output sample
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(cleaned, f, ensure_ascii=False, indent=2)

	print("Computing embeddings...")
	embedder = EmbeddingModel()
	texts = [c["text"] for c in cleaned]
	embeddings = embedder.embed_texts(texts)

	print("Storing vectors...")
	store = VectorStore(index_dir=index_dir)
	for doc_id, (meta, vec) in enumerate(zip(cleaned, embeddings)):
		store.add(id=str(doc_id), embedding=vec, metadata=meta)

	store.save()
	print("Pipeline run complete.")


def create_arg_parser():
	p = argparse.ArgumentParser(description="Run text pipeline")
	p.add_argument("input", help="Path to input file or directory (csv, pdf, html)")
	p.add_argument("--out", default="data/cleaned_output.json", help="Path to write cleaned JSON")
	p.add_argument("--index-dir", default="index", help="Directory to store vector index")
	return p


if __name__ == "__main__":
	parser = create_arg_parser()
	args = parser.parse_args()
	run_pipeline(args.input, args.out, args.index_dir)
