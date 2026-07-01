from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

import numpy as np
from sentence_transformers import util

from .discourse_pattern_detector import detect_debate_style_score, detect_discourse_score


@dataclass
class ConstructivenessResult:
    constructiveness_score: float
    constructive_similarity: float
    non_constructive_similarity: float
    reasoning_marker_score: float
    suggestion_score: float
    explanation_depth_score: float
    strategic_cricket_reasoning_score: float
    discourse_score: float
    debate_style_score: float
    confidence: float


class ConstructivenessDetector:
    """Prototype-based constructiveness detector for sports writing."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    constructive_examples: List[str] = [
        "The team should rotate strike in the middle overs because dot-ball pressure caused the collapse.",
        "Selection can improve by choosing a death-over specialist to support the lead pacer.",
        "A better strategy is protecting square boundaries first, then forcing straight hits.",
        "The batting role order needs adjustment so anchors stay longer when early wickets fall.",
        "This performance was poor, but practical fixes in field placement and communication can help.",
        "Captaincy decisions can improve if matchups are planned before the powerplay ends.",
        "Both bowlers had good spells, yet the finishing role should be clearer under pressure.",
        "The plan failed because lengths drifted into hitting arcs, so yorker practice should increase.",
        "The player struggled today, but coaching support and role clarity can restore confidence.",
        "The team can recover quickly with better running between wickets and calmer shot selection.",
    ]

    non_constructive_examples: List[str] = [
        "This team is trash and always useless.",
        "Awful match, awful match, awful match, nothing else to say.",
        "Everyone is pathetic and should quit immediately.",
        "The captain is an idiot and the players are losers.",
        "Terrible, terrible, terrible, same garbage forever.",
        "I hate this team, no point explaining anything.",
        "They are hopeless clowns with zero brain.",
        "Worst players ever, absolute joke, no effort.",
        "Complete nonsense from fools who cannot do anything right.",
        "Utter disaster and nothing can ever improve.",
    ]

    reasoning_markers: List[str] = [
        "because",
        "since",
        "if",
        "then",
        "therefore",
        "which means",
        "this shows",
        "this suggests",
        "as a result",
        "so that",
        "due to",
        "instead of",
        "rather than",
        "compared to",
        "could improve",
        "should improve",
        "would help",
        "needs improvement",
        "can improve",
        "better strategy",
        "when that happens",
        "because of that",
        "which made it difficult",
        "which reduced",
        "which affected",
        "which created",
        "which limited",
        "as a result",
        "this meant that",
        "this caused",
        "this made",
    ]
    suggestion_markers: List[str] = [
        "should",
        "could",
        "would",
        "needs to",
        "can improve",
        "must consider",
        "better if",
        "important to",
    ]
    problem_markers: List[str] = ["problem", "issue", "struggle", "failed", "collapse", "mistake", "weak"]
    implication_markers: List[str] = ["so", "therefore", "which means", "as a result", "then", "will", "which reduced", "which improved", "that allowed", "that created", "that prevented"]
    explanation_connectors: List[str] = [
        "instead of",
        "rather than",
        "which reduced",
        "which improved",
        "that allowed",
        "that created",
        "that prevented",
    ]
    strategic_cricket_terms: List[str] = [
        "selection decisions",
        "team balance",
        "role clarity",
        "match conditions",
        "bowling flexibility",
        "middle overs",
        "pitch conditions",
        "tactical role",
        "long-term structure",
        "combination decisions",
        "usage strategy",
        "team composition",
        "opposition strengths",
        "attack balance",
        "containment role",
    ]
    simple_problem_phrases: List[str] = [
        "difficult to perform",
        "difficult for them to perform",
        "selection becomes unclear",
        "planning becomes weak",
        "performance suffers",
        "learning does not happen",
        "learning did not happen",
    ]
    simple_consequence_phrases: List[str] = [
        "reduced confidence",
        "team loses benefit",
        "team loses the benefit",
        "results repeat",
        "results repeated",
        "which reduced",
        "which affected",
        "which created",
        "which limited",
        "this caused",
        "this made",
        "as a result",
    ]
    respectful_markers: List[str] = ["respect", "fair", "appreciate", "support", "calm", "balanced"]
    abuse_markers: List[str] = ["idiot", "moron", "stupid", "trash", "useless", "pathetic", "loser", "clown"]

    def __init__(self) -> None:
        self._constructive_vecs = None
        self._non_constructive_vecs = None

    def _ensure_banks(self, embedder) -> None:
        if self._constructive_vecs is None:
            self._constructive_vecs = embedder.encode(self.constructive_examples, convert_to_tensor=True)
        if self._non_constructive_vecs is None:
            self._non_constructive_vecs = embedder.encode(self.non_constructive_examples, convert_to_tensor=True)

    @staticmethod
    def _marker_score(text: str, markers: List[str], normalizer: float) -> float:
        lowered = text.lower()
        hits = sum(1 for marker in markers if marker in lowered)
        return max(0.0, min(1.0, hits / normalizer))

    def _reasoning_marker_score(self, text: str) -> float:
        return self._marker_score(text=text, markers=self.reasoning_markers, normalizer=4.0)

    def _suggestion_score(self, text: str) -> float:
        return self._marker_score(text=text, markers=self.suggestion_markers, normalizer=3.0)

    def _explanation_depth_score(self, text: str) -> float:
        lowered = text.lower()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if not sentences:
            return 0.0

        has_problem = any(marker in lowered for marker in self.problem_markers)
        has_explanation = (
            any(marker in lowered for marker in self.reasoning_markers)
            or any(marker in lowered for marker in self.explanation_connectors)
            or bool(re.search(r"\bif\b.*\bthen\b", lowered))
        )
        has_implication = any(marker in lowered for marker in self.implication_markers)

        chain_score = 1.0 if (has_problem and has_explanation and has_implication) else 0.0
        sentence_bonus = max(0.0, min(1.0, (len(sentences) - 2) / 3.0))
        depth = max(0.0, min(1.0, 0.7 * chain_score + 0.3 * sentence_bonus))

        # New relaxed rule for analyst reasoning chains.
        if has_problem and has_explanation and has_implication:
            depth = max(depth, 0.6)

        simple_problem_hit = any(phrase in lowered for phrase in self.simple_problem_phrases)
        simple_consequence_hit = any(phrase in lowered for phrase in self.simple_consequence_phrases)
        if simple_problem_hit and simple_consequence_hit:
            depth = max(depth, 0.6)

        return depth

    def _strategic_cricket_reasoning_score(self, text: str) -> float:
        return self._marker_score(text=text, markers=self.strategic_cricket_terms, normalizer=3.0)

    def _respectful_non_toxic_boost(self, text: str, reasoning_score: float) -> float:
        lowered = text.lower()
        respectful_tone = any(marker in lowered for marker in self.respectful_markers)
        toxic_cues = any(marker in lowered for marker in self.abuse_markers)
        if reasoning_score >= 0.2 and respectful_tone and not toxic_cues:
            return 0.12
        return 0.0

    def detect(self, text: str, embedder) -> ConstructivenessResult:
        if not text.strip():
            return ConstructivenessResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        self._ensure_banks(embedder)
        text_vec = embedder.encode(text, convert_to_tensor=True)

        constructive_sims = util.cos_sim(text_vec, self._constructive_vecs)[0].cpu().numpy()
        non_constructive_sims = util.cos_sim(text_vec, self._non_constructive_vecs)[0].cpu().numpy()

        constructive_similarity = float(np.max(constructive_sims))
        non_constructive_similarity = float(np.max(non_constructive_sims))

        similarity_margin = constructive_similarity - non_constructive_similarity
        embedding_score = 1.0 / (1.0 + float(np.exp(-8.0 * similarity_margin)))

        reasoning_marker_score = self._reasoning_marker_score(text)
        suggestion_score = self._suggestion_score(text)
        explanation_depth_score = self._explanation_depth_score(text)
        strategic_cricket_reasoning_score = self._strategic_cricket_reasoning_score(text)
        discourse_score = detect_discourse_score(text)
        debate_style_score = detect_debate_style_score(text)

        score = (
            0.3 * embedding_score
            + 0.16 * reasoning_marker_score
            + 0.16 * suggestion_score
            + 0.14 * explanation_depth_score
            + 0.14 * discourse_score
            + 0.1 * strategic_cricket_reasoning_score
        )

        # Debate-style balancing phrases should modestly boost confidence and constructive intent.
        if debate_style_score >= 0.4:
            score += 0.05 * debate_style_score

        score = max(0.0, min(1.0, score + self._respectful_non_toxic_boost(text, reasoning_marker_score)))

        confidence = min(
            1.0,
            abs(similarity_margin) * 2.0
            + 0.25 * reasoning_marker_score
            + 0.2 * explanation_depth_score
            + 0.2 * discourse_score
            + 0.15 * debate_style_score
            + 0.2 * strategic_cricket_reasoning_score,
        )

        return ConstructivenessResult(
            constructiveness_score=score,
            constructive_similarity=constructive_similarity,
            non_constructive_similarity=non_constructive_similarity,
            reasoning_marker_score=reasoning_marker_score,
            suggestion_score=suggestion_score,
            explanation_depth_score=explanation_depth_score,
            strategic_cricket_reasoning_score=strategic_cricket_reasoning_score,
            discourse_score=discourse_score,
            debate_style_score=debate_style_score,
            confidence=confidence,
        )

    def detect_as_dict(self, text: str, embedder) -> Dict[str, float]:
        result = self.detect(text=text, embedder=embedder)
        return {
            "constructiveness_score": round(result.constructiveness_score, 4),
            "constructive_similarity": round(result.constructive_similarity, 4),
            "non_constructive_similarity": round(result.non_constructive_similarity, 4),
            "reasoning_marker_score": round(result.reasoning_marker_score, 4),
            "suggestion_score": round(result.suggestion_score, 4),
            "explanation_depth_score": round(result.explanation_depth_score, 4),
            "strategic_cricket_reasoning_score": round(result.strategic_cricket_reasoning_score, 4),
            "discourse_score": round(result.discourse_score, 4),
            "debate_style_score": round(result.debate_style_score, 4),
            "confidence": round(result.confidence, 4),
        }
