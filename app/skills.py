"""Vocabulary-based skill extraction (same idea as the SkillRadar regex extractor).

Vocab file format, one skill per line:  canonical|alias1|alias2
"""
import re
from functools import lru_cache
from pathlib import Path


def load_vocab(path: str) -> dict[str, list[str]]:
    vocab = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names = [n.strip() for n in line.split("|") if n.strip()]
        vocab[names[0]] = names
    return vocab


def _pattern(term: str) -> re.Pattern:
    # Boundaries that work for terms like C++, Node.js, CI/CD
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9+#])", re.IGNORECASE)


class SkillExtractor:
    def __init__(self, vocab: dict[str, list[str]]):
        self.patterns = {canon: [_pattern(a) for a in aliases] for canon, aliases in vocab.items()}

    def extract(self, text: str) -> list[str]:
        found = [canon for canon, pats in self.patterns.items() if any(p.search(text) for p in pats)]
        return sorted(found)


@lru_cache
def default_extractor(path: str) -> SkillExtractor:
    return SkillExtractor(load_vocab(path))
