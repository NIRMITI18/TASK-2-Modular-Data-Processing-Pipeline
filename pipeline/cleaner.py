"""Simple text cleaner: normalize whitespace, remove control chars, basic sentence splitting."""
import re
from typing import Dict


RE_WS = re.compile(r"\s+")


class Cleaner:
    def __init__(self):
        pass

    def clean_text(self, item) -> Dict:
        """Accepts either a dict {'source', 'text'} or a raw string.

        Returns a metadata dict: {'source', 'text', 'tokens', 'summary'}
        """
        if isinstance(item, dict):
            source = item.get("source")
            raw = item.get("text") or ""
        else:
            source = None
            raw = item or ""

        # basic cleaning
        text = RE_WS.sub(" ", raw).strip()
        # remove non-printable characters
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\t')

        # naive sentence trimming
        if len(text) > 1000:
            preview = text[:1000].rsplit(' ', 1)[0]
        else:
            preview = text

        return {
            "source": source,
            "text": text,
            "snippet": preview[:400]
        }
