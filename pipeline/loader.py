"""Loaders for different file types: CSV, TXT, PDF, HTML.

Each loader returns a list of dicts: {"source": path, "text": raw_text}
"""
from typing import List
import os
import csv


def _load_csv(path: str) -> List[dict]:
    items = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # assume first column contains textual content if header unknown
        for i, row in enumerate(reader):
            if not row:
                continue
            text = " ".join(row)
            items.append({"source": f"{path}#row:{i}", "text": text})
    return items


def _load_txt(path: str) -> List[dict]:
    with open(path, encoding='utf-8') as f:
        text = f.read()
    return [{"source": path, "text": text}]


def _load_pdf(path: str) -> List[dict]:
    # try PyPDF2 first
    try:
        from PyPDF2 import PdfReader
    except Exception:
        # PDF support not available
        return [{"source": path, "text": ""}]

    text = ""
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            try:
                text += p.extract_text() or ""
            except Exception:
                continue
    except Exception:
        text = ""
    return [{"source": path, "text": text}]


def _load_html(path: str) -> List[dict]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return [{"source": path, "text": ""}]

    with open(path, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=' ', strip=True)
    return [{"source": path, "text": text}]


class Loader:
    def load(self, path: str) -> List[dict]:
        if os.path.isdir(path):
            # load all files in dir
            out = []
            for fname in sorted(os.listdir(path)):
                full = os.path.join(path, fname)
                out.extend(self.load(full))
            return out

        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv",):
            return _load_csv(path)
        if ext in (".txt",):
            return _load_txt(path)
        if ext in (".pdf",):
            return _load_pdf(path)
        if ext in (".html", ".htm"):
            return _load_html(path)

        # unknown file type: attempt text read
        try:
            return _load_txt(path)
        except Exception:
            return [{"source": path, "text": ""}]
