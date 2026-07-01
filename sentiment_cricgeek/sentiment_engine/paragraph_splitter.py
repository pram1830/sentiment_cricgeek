from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


@dataclass
class ParagraphUnit:
    index: int
    text: str
    sentences: List[str]


def split_into_sentences(paragraph: str) -> List[str]:
    """Split a paragraph into sentence-like units."""

    content = paragraph.strip()
    if not content:
        return []

    candidates = re.split(r"(?<=[.!?])\s+", content)
    return [c.strip() for c in candidates if c.strip()]


def split_into_paragraphs(text: str) -> List[ParagraphUnit]:
    """Split text into paragraph units with sentence lists."""

    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", normalized) if p.strip()]
    return [
        ParagraphUnit(index=i, text=paragraph, sentences=split_into_sentences(paragraph))
        for i, paragraph in enumerate(paragraphs, start=1)
    ]
