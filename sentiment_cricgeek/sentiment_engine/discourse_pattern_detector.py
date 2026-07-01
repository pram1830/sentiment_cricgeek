from __future__ import annotations

import re
from typing import List


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _contains_any(text: str, patterns: List[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def detect_discourse_score(text: str) -> float:
    lowered = text.lower().strip()
    if not lowered:
        return 0.0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", lowered) if s.strip()]

    contrast_markers = [
        "however",
        "but",
        "while",
        "although",
        "at the same time",
        "instead of",
        "rather than",
        "in those situations",
    ]
    evaluation_frames = [
        "what made this important",
        "what changed the situation",
        "what influenced the outcome",
        "what shaped the result",
    ]
    event_markers = ["when", "after", "before", "during", "in the final", "session", "over"]
    interpretation_markers = [
        "this showed",
        "this suggests",
        "this means",
        "the point is",
        "it revealed",
        "because of this",
        "in those situations",
        "going forward",
        "this meant that",
        "this caused",
        "this made",
    ]
    impact_markers = [
        "as a result",
        "therefore",
        "impact",
        "outcome",
        "so the team",
        "which led to",
        "which reduced",
        "which improved",
        "which created",
        "which allowed",
        "because of this",
        "going forward",
        "which reduced",
        "in those situations",
    ]

    contrast_score = _clip(sum(1 for marker in contrast_markers if marker in lowered) / 2.0, 0.0, 1.0)
    evaluation_score = _clip(sum(1 for frame in evaluation_frames if frame in lowered) / 2.0, 0.0, 1.0)

    has_event = _contains_any(lowered, event_markers)
    has_interpretation = _contains_any(lowered, interpretation_markers)
    has_impact = _contains_any(lowered, impact_markers)
    explanation_arc = 1.0 if (has_event and has_interpretation and has_impact) else 0.0

    sentence_span_bonus = _clip((len(sentences) - 3) / 4.0, 0.0, 1.0)

    discourse_score = (
        0.3 * contrast_score
        + 0.2 * evaluation_score
        + 0.35 * explanation_arc
        + 0.15 * sentence_span_bonus
    )
    return _clip(discourse_score, 0.0, 1.0)


def detect_debate_style_score(text: str) -> float:
    lowered = text.lower().strip()
    if not lowered:
        return 0.0

    debate_markers = [
        "on one hand",
        "on the other hand",
        "some believe",
        "others argue",
        "depends on",
    ]
    hits = sum(1 for marker in debate_markers if marker in lowered)
    return _clip(hits / 2.0, 0.0, 1.0)
