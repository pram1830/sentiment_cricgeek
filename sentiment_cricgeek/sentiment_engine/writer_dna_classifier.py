from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import util


@dataclass
class WriterDNAResult:
    writer_type: str
    writer_type_probabilities: Dict[str, float]
    rule_scores: Dict[str, float]
    embedding_scores: Dict[str, float]


class WriterDNAClassifier:
    """Hybrid writer DNA detector using rules and embedding similarity."""

    WRITER_TYPES = ["Passionate Fan", "Analyst", "Storyteller", "Debater", "All-Rounder"]
    ARCHETYPES: Dict[str, List[str]] = {
        "Passionate Fan": [
            "I care deeply about this team and want practical improvements before the next match.",
            "This is emotional but respectful fan writing focused on support and constructive fixes.",
        ],
        "Analyst": [
            "The argument is tactical and evidence-driven, with clear cause and effect reasoning.",
            "This is analytical writing with structured cricket reasoning and clear logic.",
        ],
        "Storyteller": [
            "This paragraph narrates a match moment with vivid emotion and coherent flow.",
            "It reads like a cricket story with scene setting and emotional progression.",
        ],
        "Debater": [
            "This writing presents two sides and weighs arguments in a balanced way.",
            "The paragraph uses rebuttal language and compares competing viewpoints.",
        ],
        "All-Rounder": [
            "This writing blends fan passion, analysis, narrative, and balanced argument.",
            "A hybrid cricket style combining several writing modes coherently.",
        ],
    }

    RULE_TERMS: Dict[str, List[str]] = {
        "Passionate Fan": ["we", "our", "support", "believe", "proud", "heart", "fans", "care", "back"],
        "Analyst": ["because", "therefore", "evidence", "data", "strategy", "tactic", "trend", "rate", "phase", "matchup", "sequence", "process", "problem", "approach", "structure", "discipline", "role"],
        "Storyteller": ["when", "then", "moment", "remember", "crowd", "scene", "suddenly", "after", "lights", "silence", "roar", "watched"],
        "Debater": ["however", "on the other hand", "although", "while", "but", "whereas", "counter", "both", "fair", "balanced", "argue", "argument", "viewpoint", "compare", "comparing"],
        "All-Rounder": ["overall", "balance", "mix", "both", "combined", "together"],
    }

    def _rule_scores(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        token_count = max(len(tokens), 1)
        phrase_text = text.lower()

        scores: Dict[str, float] = {}
        for writer_type, terms in self.RULE_TERMS.items():
            hits = 0.0
            for term in terms:
                if " " in term:
                    hits += 1.0 if term in phrase_text else 0.0
                else:
                    hits += sum(1 for token in tokens if token == term)
            scores[writer_type] = min(1.0, hits / max(2.0, 0.12 * token_count))

        return scores

    def _embedding_scores(self, text: str, embedder) -> Dict[str, float]:
        if not text.strip():
            return {writer_type: 0.0 for writer_type in self.WRITER_TYPES}

        text_vec = embedder.encode(text, convert_to_tensor=True)
        scores: Dict[str, float] = {}
        for writer_type, prototypes in self.ARCHETYPES.items():
            proto_vec = embedder.encode(prototypes, convert_to_tensor=True)
            sims = util.cos_sim(text_vec, proto_vec)[0].cpu().numpy()
            top_sim = float(np.max(sims))
            scores[writer_type] = max(0.0, min(1.0, (top_sim - 0.15) / 0.65))
        return scores

    @staticmethod
    def _normalize(raw_scores: Dict[str, float]) -> Dict[str, float]:
        values = np.array([max(0.0, raw_scores[k]) for k in WriterDNAClassifier.WRITER_TYPES], dtype=float)
        if float(values.sum()) <= 0.0:
            values = np.ones_like(values)
        probs = values / float(values.sum())
        return {k: float(v) for k, v in zip(WriterDNAClassifier.WRITER_TYPES, probs)}

    def classify(
        self,
        blog_text: str,
        embedder=None,
        zero_shot_tiebreaker=None,
    ) -> WriterDNAResult:
        if not blog_text.strip():
            zero = {k: 0.0 for k in self.WRITER_TYPES}
            probs = {k: 0.0 for k in self.WRITER_TYPES}
            probs["All-Rounder"] = 1.0
            return WriterDNAResult(
                writer_type="All-Rounder",
                writer_type_probabilities=probs,
                rule_scores=zero,
                embedding_scores=zero,
            )

        rule_scores = self._rule_scores(blog_text)
        embed_scores = self._embedding_scores(blog_text, embedder) if embedder is not None else {k: 0.0 for k in self.WRITER_TYPES}
        combined = {
            writer_type: (
                0.65 * rule_scores[writer_type] + 0.35 * embed_scores[writer_type]
                if embedder is not None
                else rule_scores[writer_type]
            )
            for writer_type in self.WRITER_TYPES
        }
        probs = self._normalize(combined)

        ranked: List[Tuple[str, float]] = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_type, top_prob = ranked[0]
        second_prob = ranked[1][1]

        # Optional tie-breaker hook only for close calls; not required for normal operation.
        if zero_shot_tiebreaker is not None and (top_prob - second_prob) < 0.08:
            tie_result = zero_shot_tiebreaker(blog_text)
            if tie_result in self.WRITER_TYPES:
                top_type = tie_result

        return WriterDNAResult(
            writer_type=top_type,
            writer_type_probabilities={k: round(v, 4) for k, v in probs.items()},
            rule_scores={k: round(v, 4) for k, v in rule_scores.items()},
            embedding_scores={k: round(v, 4) for k, v in embed_scores.items()},
        )
