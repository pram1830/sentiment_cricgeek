from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import util


@dataclass
class StanceResult:
    stance_label: str
    stance_probabilities: Dict[str, float]
    stance_confidence: float
    primary_stance_label: str
    primary_stance_probabilities: Dict[str, float]
    primary_stance_confidence: float
    style_tags: List[str]
    scalar_metrics: Dict[str, float]
    sarcasm_gate_reason: str
    quoted_attack_detected: bool
    attack_endorsement_detected: bool
    attack_rejection_detected: bool
    contrast_rejection_detected: bool
    fairness_defense_score: float
    reputation_defense_score: float
    criticism_reference_score: float
    context_change_score: float
    evaluation_redirection_score: float
    credibility_restoration_score: float
    contrast_structure_score: float
    causal_defense_score: float
    credibility_defense_score: float
    supportive_defense_strength: float
    paragraph_stances: List[Dict[str, Any]]
    overall_stance: Dict[str, Any]


class StanceDetector:
    PRIMARY_STANCE_LABELS = [
        "SUPPORTIVE_DEFENSE",
        "DIRECT_CRITICISM",
        "SARCASTIC_ATTACK",
        "NEUTRAL_COMMENTARY",
        "MIXED_OR_AMBIGUOUS",
    ]

    STANCE_LABELS = [
        "SUPPORTIVE_DEFENSE",
        "CONSTRUCTIVE_CRITICISM",
        "NEUTRAL_ANALYSIS",
        "BALANCED_DEBATE",
        "DISMISSIVE_COMPLAINT",
        "DIRECT_ATTACK",
        "SARCASTIC_CRITICISM",
        "MIXED_STANCE",
    ]

    SUPPORTIVE_SIGNALS = [
        "defend player",
        "unfair criticism",
        "context matters",
        "different scenario",
        "evolved player",
        "mentally stronger",
        "future potential",
        "needs confidence",
        "team did not support",
    ]
    CONSTRUCTIVE_SIGNALS = [
        "should have used",
        "could have worked better",
        "instead of",
        "depending on conditions",
        "role clarity",
        "selection pattern",
        "team balance",
    ]
    COMPLAINT_SIGNALS = [
        "same mistake again",
        "nothing changes",
        "always happens",
        "waste selection",
        "management never learns",
        "again and again",
        "same confusion",
        "repeating mistakes",
        "frustrating to watch",
        "ignored repeatedly",
        "again and again ignored",
        "same issue again",
        "no clear plan",
        "no long-term plan",
        "repeated confusion",
    ]
    ATTACK_SIGNALS = [
        "pathetic",
        "useless",
        "slow player",
        "bad player",
        "overrated",
        "should not play",
        "clearly not performing",
        "management ignoring problems",
        "keeps selecting players who are not performing",
        "issues are being ignored",
        "problems are being ignored",
        "obvious issues ignored",
        "selectors ignoring",
        "management ignoring",
        "selection mistakes continue",
        "selection problems continue",
    ]
    SUGGESTION_SIGNALS = [
        "should",
        "could",
        "needs to",
        "need to",
        "must",
        "recommend",
        "better to",
        "improve",
        "fix",
        "instead of",
    ]
    SARCASM_EXAGGERATION_METAPHORS = [
        "masterclass in suspense",
        "meditation retreat",
        "rare art",
        "optional feature",
        "geological era",
    ]
    SARCASM_DELAYED_NEGATIVE_PHRASES = [
        "if only the scoreboard agreed",
        "until you check the scoreboard",
        "except the results",
        "for everyone except the team",
        "still somehow not performing",
    ]
    SARCASM_ABSURD_COMPARISONS = [
        "like a geological era",
        "like a meditation retreat",
        "like a rare art exhibit",
        "as optional as fielding",
    ]
    SARCASM_PRAISE_WORDS = [
        "masterclass",
        "brilliant",
        "amazing",
        "fantastic",
        "legendary",
        "outstanding",
        "elite",
        "visionary",
        "genius",
    ]
    SARCASM_PERFORMANCE_MISMATCH = [
        "not performing",
        "ignored",
        "selection mistakes",
        "selection problems",
        "same mistake",
        "repeated confusion",
        "slow",
        "dropped",
    ]
    SARCASM_VALIDATION_CASES = [
        (
            "His batting is a masterclass in suspense, visionary really, because nobody knows when runs arrive.",
            True,
        ),
        (
            "Brilliant tempo if the goal is to make boundaries an optional feature.",
            True,
        ),
        (
            "Some analysts prefer him opening while others value flexibility for team balance.",
            False,
        ),
        (
            "The criticism ignores his role change and improved game awareness this season.",
            False,
        ),
        (
            "Amazing intent, until you check the scoreboard and selection mistakes continue.",
            True,
        ),
    ]

    QUOTE_CONTEXT_MARKERS = [
        "people say",
        "some say",
        "others think",
        "we cannot say",
        "it is wrong to say",
        "it is unfair to say",
        "calling him",
        "labeling him",
        "just because",
    ]
    REJECTION_MARKERS = [
        "we cannot say",
        "cannot say",
        "it is wrong to say",
        "it is unfair to say",
        "wrong",
        "unfair",
        "incorrect",
        "not true",
        "not fair",
        "do not agree",
    ]
    ENDORSEMENT_MARKERS = [
        "and they are right",
        "and thats right",
        "they are right",
        "right to call",
        "correct to call",
    ]
    CONTRAST_MARKERS = ["but", "however", "instead", "rather", "actually", "irrelevant", "unfair", "incorrect"]
    FAIRNESS_DEFENSE_PATTERNS = [
        "irrelevant to judge",
        "not fair to judge",
        "unfair to judge",
        "cannot judge based on",
        "should not judge based on",
        "different phase of career",
        "context was different",
        "role was different",
        "he has evolved",
        "player has improved",
        "progressing now",
        "recent improvement",
        "environment matters",
        "confidence matters",
        "team role matters",
        "still developing",
        "young player evolves",
        "needs right environment",
        "numbers do not tell the full story",
        "stats alone do not explain",
    ]
    REPUTATION_DEFENSE_PATTERNS = [
        "torchbearer",
        "important player",
        "valuable player",
        "clear player mentally",
        "strong mindset",
        "talented player",
        "high potential",
        "future looks good",
        "has shown improvement",
        "still has ability",
        "quality player",
    ]
    CRITICISM_REFERENCE_PATTERNS = [
        "people judge",
        "people still judge",
        "many judge",
        "many people judge",
        "some judge",
        "critics judge",
        "critics judge based on",
        "fans judge",
        "fans judge based on",
        "often judged",
        "judged based on earlier",
        "judged based on past",
        "judged based on previous performances",
        "focus on strike rate",
        "focus only on numbers",
        "based on earlier seasons",
        "based on past performances",
        "based only on statistics",
        "concerns from earlier phase",
        "based on earlier concerns",
        "based on earlier weaknesses",
        "previous strike rate",
        "earlier numbers",
        "old performances",
        "old numbers",
        "old numbers alone",
        "based on old numbers",
    ]
    CONTEXT_CHANGE_PATTERNS = [
        "role changed",
        "confidence changed",
        "environment changed",
        "team role changed",
        "different situation",
        "different phase",
        "recent improvement",
        "recent performances suggest",
        "has improved",
        "has adapted",
        "still developing",
        "young player improves",
        "settled into role",
        "clearer environment",
        "role was different",
        "context was different",
        "has evolved",
    ]
    EVALUATION_REDIRECTION_PATTERNS = [
        "makes more sense to evaluate",
        "better to evaluate",
        "should evaluate based on",
        "judge based on current",
        "current progress matters",
        "recent progress matters",
        "not only based on earlier numbers",
        "instead of relying on earlier numbers",
        "rather than earlier numbers",
        "evaluate him differently now",
        "not fair to judge based on old numbers",
        "not fair to judge based on old numbers alone",
    ]
    CREDIBILITY_RESTORATION_PATTERNS = [
        "still a quality player",
        "still important player",
        "still valuable player",
        "has strong ability",
        "has improved recently",
        "has shown improvement",
        "future looks positive",
        "has potential",
        "clear player mentally",
        "talented player",
    ]
    CONTRAST_STRUCTURE_PATTERNS = [
        "but",
        "however",
        "instead",
        "rather",
        "although",
        "even though",
        "while others think",
        "some believe",
        "critics argue",
    ]
    DISAGREEMENT_CONTEXT_PATTERNS = [
        "people say",
        "some say",
        "others think",
        "many argue",
        "critics say",
        "critics argue",
        "while others think",
        "some believe",
        "judge",
        "judged",
    ]

    PROTOTYPES: Dict[str, List[str]] = {
        "SUPPORTIVE_DEFENSE": [
            "The criticism is unfair because context matters and the player needs confidence.",
            "This is a supportive defense that rejects unfair labels and explains context.",
        ],
        "CONSTRUCTIVE_CRITICISM": [
            "The team should have used a clearer role with better selection balance depending on conditions.",
            "Constructive criticism proposes better tactical choices and role clarity.",
        ],
        "NEUTRAL_ANALYSIS": [
            "This is a neutral analysis of performance factors without emotional blame.",
            "Balanced factual commentary with no direct stance against the player.",
        ],
        "BALANCED_DEBATE": [
            "On one hand there is criticism, on the other hand there is context, so a balanced debate is needed.",
            "The writer compares viewpoints and reaches a fair middle ground.",
        ],
        "DISMISSIVE_COMPLAINT": [
            "This is repetitive complaint language saying nothing changes and management never learns.",
            "Dismissive ranting without practical solutions.",
        ],
        "DIRECT_ATTACK": [
            "This is a direct personal attack calling the player useless and overrated.",
            "Hostile endorsement of insulting labels toward a player.",
        ],
        "SARCASTIC_CRITICISM": [
            "This is sarcastic criticism using praise words in a mocking and contradictory way.",
            "Mocking sports commentary with exaggerated metaphors and delayed negative intent.",
        ],
        "MIXED_STANCE": [
            "The paragraph contains mixed cues and does not strongly fit one stance.",
            "A blend of support, criticism, and debate without a dominant stance.",
        ],
    }

    BENCHMARK_DATASET_PATH = Path(__file__).resolve().parent / "benchmarks" / "cricgeek_stance_benchmark_200.json"
    BENCHMARK_EMBEDDING_CACHE_PATH = Path(__file__).resolve().parent / "benchmarks" / "cricgeek_stance_benchmark_extended.embedding_cache.npz"

    def __init__(self) -> None:
        self._benchmark_ready = False
        self._centroid_embeddings: Dict[str, np.ndarray] = {}
        self._centroid_features: Dict[str, np.ndarray] = {}
        self._benchmark_embeddings: np.ndarray | None = None
        self._benchmark_labels: List[str] = []
        self._semantic_cache_ready = False
        self._semantic_centroids: Dict[str, np.ndarray] = {}
        self._sarcasm_threshold = self._tune_sarcasm_threshold()

    def _signal_score(self, text: str, signals: List[str], normalizer: float = 3.0) -> float:
        lowered = text.lower()
        hits = sum(1 for signal in signals if signal in lowered)
        return max(0.0, min(1.0, hits / normalizer))

    def _sarcasm_features(self, text: str) -> Dict[str, float]:
        lowered = text.lower()

        metaphor_hit = any(phrase in lowered for phrase in self.SARCASM_EXAGGERATION_METAPHORS)
        delayed_negative_hit = any(phrase in lowered for phrase in self.SARCASM_DELAYED_NEGATIVE_PHRASES)
        absurd_comparison_hit = any(phrase in lowered for phrase in self.SARCASM_ABSURD_COMPARISONS)
        praise_hit = any(word in lowered for word in self.SARCASM_PRAISE_WORDS)
        mismatch_hit = any(phrase in lowered for phrase in self.SARCASM_PERFORMANCE_MISMATCH)

        praise_mismatch_combo = re.search(
            r"(masterclass|brilliant|amazing|fantastic|legendary|outstanding|elite).{0,50}(not performing|ignored|mistake|slow|dropped)",
            lowered,
        ) is not None
        contradiction_marker_combo = re.search(
            r"(visionary|brilliant|amazing|masterclass).{0,80}(but|until|except|while).{0,80}(ignored|mistake|not performing|slow)",
            lowered,
        ) is not None
        mocking_tail_combo = re.search(
            r"(visionary|genius|masterclass).{0,30}really",
            lowered,
        ) is not None

        exaggeration_units = 0.0
        exaggeration_units += 1.0 if metaphor_hit else 0.0
        exaggeration_units += 1.0 if absurd_comparison_hit else 0.0
        exaggeration_units += 1.0 if "five stages of grief" in lowered else 0.0
        exaggeration_score = max(0.0, min(1.0, exaggeration_units / 2.0))

        contradiction_units = 0.0
        contradiction_units += 1.0 if praise_mismatch_combo else 0.0
        contradiction_units += 1.0 if contradiction_marker_combo else 0.0
        contradiction_units += 1.0 if mocking_tail_combo else 0.0
        contradiction_units += 1.0 if (praise_hit and "?" in text) else 0.0
        contradiction_units += 1.0 if (praise_hit and "apparently" in lowered) else 0.0
        contradiction_units += 1.0 if (praise_hit and mismatch_hit) else 0.0
        contradiction_score = max(0.0, min(1.0, contradiction_units / 2.0))

        context_mismatch_score = max(
            0.0,
            min(
                1.0,
                1.0 if (praise_hit and (mismatch_hit or metaphor_hit or absurd_comparison_hit or "apparently" in lowered)) else 0.0,
            ),
        )

        mocking_units = 0.0
        mocking_units += 1.0 if (praise_hit and metaphor_hit) else 0.0
        mocking_units += 1.0 if (praise_hit and absurd_comparison_hit) else 0.0
        mocking_units += 1.0 if praise_mismatch_combo else 0.0
        praise_in_mocking_context_score = max(0.0, min(1.0, mocking_units / 2.0))

        delayed_negative_score = max(0.0, min(1.0, 1.0 if delayed_negative_hit else 0.0))

        sarcasm_score = (
            0.2 * exaggeration_score
            + 0.3 * contradiction_score
            + 0.2 * context_mismatch_score
            + 0.2 * praise_in_mocking_context_score
            + 0.1 * delayed_negative_score
        )
        sarcasm_score = max(0.0, min(1.0, sarcasm_score))

        return {
            "sarcasm_score": sarcasm_score,
            "exaggeration_score": exaggeration_score,
            "contradiction_score": contradiction_score,
            "context_mismatch_score": context_mismatch_score,
            "praise_in_mocking_context_score": praise_in_mocking_context_score,
            "delayed_negative_score": delayed_negative_score,
        }

    def _sarcasm_score(self, text: str) -> float:
        return float(self._sarcasm_features(text).get("sarcasm_score", 0.0))

    def _tune_sarcasm_threshold(self) -> float:
        candidates = [round(value, 2) for value in np.arange(0.35, 0.71, 0.05)]
        best_threshold = 0.45
        best_score = -1.0

        for threshold in candidates:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for text, expected_sarcastic in self.SARCASM_VALIDATION_CASES:
                predicted = self._sarcasm_score(text) >= threshold
                if expected_sarcastic and predicted:
                    tp += 1
                elif expected_sarcastic and not predicted:
                    fn += 1
                elif not expected_sarcastic and predicted:
                    fp += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            score = 0.6 * recall + 0.25 * precision + 0.15 * specificity

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        return best_threshold

    def _style_tags(
        self,
        text: str,
        sarcasm_score: float,
        exaggeration_score: float,
        contradiction_score: float,
        context_mismatch_score: float,
        directness_score: float,
    ) -> List[str]:
        lowered = text.lower()
        tags: List[str] = []

        if directness_score >= 0.45 or ("?" not in text and re.search(r"\b(is|are|has|have|keeps|clearly|should|must|handled|adapted|played|repeated|working)\b", lowered)):
            tags.append("DIRECT")
        if sarcasm_score >= self._sarcasm_threshold or contradiction_score >= 0.5:
            tags.append("SARCASTIC")
        if exaggeration_score >= 0.5:
            tags.append("HYPERBOLIC")
        if "?" in text or re.search(r"(really\?|seriously\?|is this|what exactly)", lowered):
            tags.append("RHETORICAL")
        if '"' in text or "'" in text or re.search(r"(people say|some say|others believe|analysts say)", lowered):
            tags.append("QUOTED")
        if re.search(r"(analysts|reports|people say|some say|others believe)", lowered):
            tags.append("REPORTED")
        if sarcasm_score >= self._sarcasm_threshold and directness_score < 0.45:
            tags.append("IMPLIED")
        if re.search(r"(honestly|really|you know|i think|it feels)", lowered):
            tags.append("CONVERSATIONAL")

        # Preserve insertion order and uniqueness.
        return list(dict.fromkeys(tags))

    def _resolve_primary_stance(
        self,
        supportive_signal: float,
        criticism_signal: float,
        sarcastic_signal: float,
        neutrality_score: float,
        sarcasm_score: float,
        contradiction_score: float,
        context_mismatch_score: float,
        sarcasm_gate_triggered: bool,
        stronger_explicit_supportive: bool,
    ) -> tuple[str, Dict[str, float], str]:
        support_component = max(0.0, supportive_signal)
        criticism_component = max(0.0, criticism_signal)
        sarcastic_component = max(0.0, sarcastic_signal)
        neutral_component = max(0.0, neutrality_score)
        mixed_component = max(0.0, 1.0 - abs(support_component - criticism_component) - 0.2)

        primary_raw = {
            "SUPPORTIVE_DEFENSE": support_component,
            "DIRECT_CRITICISM": criticism_component,
            "SARCASTIC_ATTACK": sarcastic_component,
            "NEUTRAL_COMMENTARY": neutral_component,
            "MIXED_OR_AMBIGUOUS": mixed_component,
        }

        if sarcasm_score >= self._sarcasm_threshold and contradiction_score >= 0.35 and context_mismatch_score >= 0.35:
            primary_raw["SUPPORTIVE_DEFENSE"] = max(0.0, primary_raw["SUPPORTIVE_DEFENSE"] - 0.35)

        if sarcasm_gate_triggered and not stronger_explicit_supportive:
            primary_raw["SARCASTIC_ATTACK"] = max(primary_raw["SARCASTIC_ATTACK"], 0.78)

        total = float(sum(primary_raw.values()))
        if total <= 0.0:
            primary_probs = {label: 1.0 / len(self.PRIMARY_STANCE_LABELS) for label in self.PRIMARY_STANCE_LABELS}
        else:
            primary_probs = {label: round(float(primary_raw[label] / total), 4) for label in self.PRIMARY_STANCE_LABELS}

        primary_label = max(primary_probs.items(), key=lambda item: item[1])[0]
        reason = "max_primary_probability"

        if sarcasm_gate_triggered and not stronger_explicit_supportive:
            primary_label = "SARCASTIC_ATTACK"
            reason = "sarcasm_gate_forced_sarcastic_attack"
        elif support_component >= 0.28 and criticism_component >= 0.28 and abs(support_component - criticism_component) <= 0.15:
            primary_label = "MIXED_OR_AMBIGUOUS"
            reason = "close_support_and_criticism_balance"
        elif criticism_component >= 0.34 and criticism_component >= support_component - 0.02 and sarcastic_component < 0.35:
            primary_label = "DIRECT_CRITICISM"
            reason = "criticism_signal_near_or_above_support"
        elif support_component >= 0.33 and support_component >= criticism_component + 0.08 and sarcastic_component < 0.35:
            primary_label = "SUPPORTIVE_DEFENSE"
            reason = "supportive_signal_above_criticism_margin"
        elif support_component >= 0.42 and criticism_component < 0.35 and sarcastic_component < 0.35:
            primary_label = "SUPPORTIVE_DEFENSE"
            reason = "dominant_supportive_signal"
        elif criticism_component >= 0.4 and sarcastic_component < 0.35:
            primary_label = "DIRECT_CRITICISM"
            reason = "dominant_direct_criticism_signal"
        elif support_component >= 0.45 and criticism_component >= 0.45 and abs(support_component - criticism_component) <= 0.15:
            primary_label = "MIXED_OR_AMBIGUOUS"
            reason = "balanced_support_and_criticism"
        elif support_component >= 0.35 and criticism_component >= 0.35 and abs(support_component - criticism_component) <= 0.2:
            primary_label = "MIXED_OR_AMBIGUOUS"
            reason = "close_polarity_signals"
        elif neutral_component >= 0.45 and support_component < 0.35 and criticism_component < 0.35 and sarcastic_component < 0.35:
            primary_label = "NEUTRAL_COMMENTARY"
            reason = "low_polarity_commentary"

        return primary_label, primary_probs, reason

        if metaphor_hit:
            score_units += 1.0
        if delayed_negative_hit:
            score_units += 1.0
        if absurd_comparison_hit:
            score_units += 1.0
        if praise_hit and metaphor_hit:
            score_units += 1.0
        if praise_mismatch_combo or (praise_hit and mismatch_hit):
            score_units += 1.0

        return max(0.0, min(1.0, score_units / 3.0))

    def _load_benchmark_samples(self) -> List[Dict[str, Any]]:
        if not self.BENCHMARK_DATASET_PATH.exists():
            return []
        try:
            raw = json.loads(self.BENCHMARK_DATASET_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict) and item.get("text") and item.get("expected_stance")]

    def _feature_vector(self, text: str) -> np.ndarray:
        lowered = text.lower()
        criticism = self.detect_criticism_reference_patterns(lowered)
        context = self.detect_context_change_patterns(lowered)
        evaluation = self.detect_evaluation_redirection_patterns(lowered)
        credibility = self.detect_credibility_restoration_patterns(lowered)
        contrast = self.detect_contrast_structure_patterns(lowered)
        suggestion = self._signal_score(lowered, self.SUGGESTION_SIGNALS, normalizer=3.0)
        complaint = self._signal_score(lowered, self.COMPLAINT_SIGNALS, normalizer=2.0)
        attack = self._signal_score(lowered, self.ATTACK_SIGNALS, normalizer=2.0)
        explanation = self._explanation_depth_score(lowered)
        discourse = self._signal_score(lowered, self.CONTRAST_MARKERS + ["because", "as a result", "this means"], normalizer=4.0)
        return np.array(
            [
                criticism,
                context,
                evaluation,
                credibility,
                contrast,
                suggestion,
                complaint,
                attack,
                explanation,
                discourse,
            ],
            dtype=float,
        )

    def _ensure_benchmark_model(self, embedder) -> None:
        if self._benchmark_ready:
            return
        samples = self._load_benchmark_samples()
        if not samples or embedder is None:
            self._benchmark_ready = True
            return

        texts = [str(item["text"]) for item in samples]
        labels = [str(item["expected_stance"]) for item in samples]
        embeddings = embedder.encode(texts, convert_to_tensor=True).cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms
        features = np.stack([self._feature_vector(text) for text in texts], axis=0)

        self._benchmark_embeddings = embeddings
        self._benchmark_labels = labels

        for label in self.STANCE_LABELS:
            idxs = [idx for idx, value in enumerate(labels) if value == label]
            if not idxs:
                continue
            emb_subset = embeddings[idxs]
            feat_subset = features[idxs]
            centroid_emb = np.mean(emb_subset, axis=0)
            norm = np.linalg.norm(centroid_emb)
            if norm > 0:
                centroid_emb = centroid_emb / norm
            self._centroid_embeddings[label] = centroid_emb
            self._centroid_features[label] = np.mean(feat_subset, axis=0)

        self._benchmark_ready = True

    def _predict_benchmark_probs(self, text: str, embedder) -> Dict[str, float]:
        self._ensure_benchmark_model(embedder)
        if (
            not self._centroid_embeddings
            or embedder is None
            or self._benchmark_embeddings is None
            or not self._benchmark_labels
        ):
            return {label: 1.0 / len(self.STANCE_LABELS) for label in self.STANCE_LABELS}

        text_emb = embedder.encode(text, convert_to_tensor=True).cpu().numpy()
        emb_norm = np.linalg.norm(text_emb)
        if emb_norm > 0:
            text_emb = text_emb / emb_norm
        feat = self._feature_vector(text)

        raw: Dict[str, float] = {label: 0.0 for label in self.STANCE_LABELS}

        sims = np.dot(self._benchmark_embeddings, text_emb)
        top_k = min(16, len(self._benchmark_labels))
        top_indices = np.argsort(-sims)[:top_k]
        for idx in top_indices:
            label = self._benchmark_labels[int(idx)]
            sim = float(sims[int(idx)])
            raw[label] = raw.get(label, 0.0) + max(0.0, sim) ** 2

        for label in self.STANCE_LABELS:
            centroid_emb = self._centroid_embeddings.get(label)
            centroid_feat = self._centroid_features.get(label)
            if centroid_emb is None or centroid_feat is None:
                continue
            emb_sim = float(np.dot(text_emb, centroid_emb))
            emb_sim = max(0.0, min(1.0, (emb_sim + 1.0) / 2.0))
            feat_dist = float(np.mean(np.abs(feat - centroid_feat)))
            feat_sim = max(0.0, min(1.0, 1.0 - feat_dist))
            raw[label] = raw.get(label, 0.0) + 0.35 * emb_sim + 0.25 * feat_sim

        return self._normalize(raw)

    def _ensure_semantic_centroids_from_cache(self) -> None:
        if self._semantic_cache_ready:
            return

        if not self.BENCHMARK_EMBEDDING_CACHE_PATH.exists():
            self._semantic_cache_ready = True
            return

        try:
            cache = np.load(self.BENCHMARK_EMBEDDING_CACHE_PATH, allow_pickle=True)
            embeddings = cache["embeddings"]
            labels = [str(label) for label in cache["labels"].tolist()]
        except Exception:
            self._semantic_cache_ready = True
            return

        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or not labels:
            self._semantic_cache_ready = True
            return

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms

        target_labels = ["SUPPORTIVE_DEFENSE", "CONSTRUCTIVE_CRITICISM", "BALANCED_DEBATE"]
        for stance in target_labels:
            idxs = [idx for idx, value in enumerate(labels) if value == stance]
            if not idxs:
                continue
            centroid = np.mean(embeddings[idxs], axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            self._semantic_centroids[stance] = centroid

        self._semantic_cache_ready = True

    def _semantic_similarities(self, text: str, embedder) -> Dict[str, float]:
        self._ensure_semantic_centroids_from_cache()
        if embedder is None or not self._semantic_centroids:
            return {
                "semantic_supportive_similarity": 0.0,
                "semantic_constructive_similarity": 0.0,
                "semantic_balanced_similarity": 0.0,
            }

        text_emb = embedder.encode(text, convert_to_tensor=True).cpu().numpy()
        text_norm = np.linalg.norm(text_emb)
        if text_norm > 0:
            text_emb = text_emb / text_norm

        def _sim(label: str) -> float:
            centroid = self._semantic_centroids.get(label)
            if centroid is None:
                return 0.0
            cosine = float(np.dot(text_emb, centroid))
            return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

        supportive = _sim("SUPPORTIVE_DEFENSE")
        constructive = _sim("CONSTRUCTIVE_CRITICISM")
        balanced = _sim("BALANCED_DEBATE")

        return {
            "semantic_supportive_similarity": max(0.0, min(1.0, supportive)),
            "semantic_constructive_similarity": max(0.0, min(1.0, constructive)),
            "semantic_balanced_similarity": max(0.0, min(1.0, balanced)),
        }

    def detect_fairness_defense_patterns(self, text: str) -> float:
        return self._signal_score(text.lower(), self.FAIRNESS_DEFENSE_PATTERNS, normalizer=3.0)

    def detect_reputation_defense_patterns(self, text: str) -> float:
        return self._signal_score(text.lower(), self.REPUTATION_DEFENSE_PATTERNS, normalizer=2.0)

    def detect_criticism_reference_patterns(self, text: str) -> float:
        lowered = text.lower()
        score = self._signal_score(lowered, self.CRITICISM_REFERENCE_PATTERNS, normalizer=3.0)
        guaranteed_patterns = [
            "people still judge",
            "many people judge",
            "fans judge based on",
            "critics judge based on",
            "judged based on earlier",
            "judged based on past",
            "judged based on previous performances",
            "concerns from earlier phase",
            "based on earlier concerns",
            "based on earlier weaknesses",
        ]
        if any(pattern in lowered for pattern in guaranteed_patterns):
            score = max(score, 0.4)
        return score

    def detect_context_change_patterns(self, text: str) -> float:
        return self._signal_score(text.lower(), self.CONTEXT_CHANGE_PATTERNS, normalizer=3.0)

    def detect_evaluation_redirection_patterns(self, text: str) -> float:
        return self._signal_score(text.lower(), self.EVALUATION_REDIRECTION_PATTERNS, normalizer=2.5)

    def detect_credibility_restoration_patterns(self, text: str) -> float:
        return self._signal_score(text.lower(), self.CREDIBILITY_RESTORATION_PATTERNS, normalizer=2.0)

    def detect_contrast_structure_patterns(self, text: str) -> float:
        lowered = text.lower()
        disagreement_context = any(marker in lowered for marker in self.DISAGREEMENT_CONTEXT_PATTERNS)
        if not disagreement_context:
            return 0.0
        score = self._signal_score(lowered, self.CONTRAST_STRUCTURE_PATTERNS, normalizer=2.0)
        if score > 0.0:
            return max(0.6, score)
        return 0.0

    def _contrast_rejection_detected(self, text: str) -> bool:
        lowered = text.lower()
        patterns = [
            r"people say .+ but",
            r"some say .+ but",
            r"others think .+ but",
            r"many argue .+ however",
            r"critics say .+ but",
        ]
        return any(re.search(pattern, lowered) is not None for pattern in patterns)

    def _quote_awareness(self, text: str) -> Dict[str, bool]:
        lowered = text.lower()
        quote_context = any(marker in lowered for marker in self.QUOTE_CONTEXT_MARKERS)
        attack_in_text = any(term in lowered for term in self.ATTACK_SIGNALS)
        quoted_attack_detected = quote_context and attack_in_text

        rejection_by_marker = any(marker in lowered for marker in self.REJECTION_MARKERS)
        endorsement_by_marker = any(marker in lowered for marker in self.ENDORSEMENT_MARKERS)

        rejection_structure = bool(re.search(r"people say.*(but|however|instead|rather|actually).*(wrong|unfair|incorrect)", lowered))
        endorsement_structure = bool(re.search(r"people say .* and (they are|that's) right", lowered))
        contrast_rejection_detected = self._contrast_rejection_detected(lowered)

        attack_rejection_detected = quoted_attack_detected and (rejection_by_marker or rejection_structure)
        attack_endorsement_detected = quoted_attack_detected and (endorsement_by_marker or endorsement_structure)

        return {
            "quoted_attack_detected": quoted_attack_detected,
            "attack_rejection_detected": attack_rejection_detected,
            "attack_endorsement_detected": attack_endorsement_detected,
            "contrast_rejection_detected": contrast_rejection_detected,
        }

    def _split_paragraphs(self, text: str) -> List[str]:
        parts = [segment.strip() for segment in re.split(r"\n\s*\n+", text) if segment.strip()]
        return parts if parts else [text.strip()]

    def _explanation_depth_score(self, text: str) -> float:
        lowered = text.lower()
        reasoning_markers = [
            "because",
            "therefore",
            "however",
            "while",
            "although",
            "if",
            "then",
            "which means",
            "as a result",
            "instead of",
            "rather than",
        ]
        marker_score = self._signal_score(lowered, reasoning_markers, normalizer=4.0)
        sentence_count = max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))
        sentence_depth = max(0.0, min(1.0, sentence_count / 4.0))
        return max(0.0, min(1.0, 0.7 * marker_score + 0.3 * sentence_depth))

    def _detect_single(self, text: str, embedder=None) -> StanceResult:
        lowered = text.lower().strip()
        if not lowered:
            probs = {label: 0.0 for label in self.STANCE_LABELS}
            probs["NEUTRAL_ANALYSIS"] = 1.0
            primary_probs = {label: 0.0 for label in self.PRIMARY_STANCE_LABELS}
            primary_probs["NEUTRAL_COMMENTARY"] = 1.0
            return StanceResult(
                stance_label="NEUTRAL_ANALYSIS",
                stance_probabilities=probs,
                stance_confidence=1.0,
                primary_stance_label="NEUTRAL_COMMENTARY",
                primary_stance_probabilities=primary_probs,
                primary_stance_confidence=1.0,
                style_tags=[],
                scalar_metrics={
                    "sarcasm_score": 0.0,
                    "contradiction_score": 0.0,
                    "exaggeration_score": 0.0,
                    "context_mismatch_score": 0.0,
                    "ridicule_score": 0.0,
                    "directness_score": 0.0,
                    "certainty_score": 0.0,
                    "hostility_score": 0.0,
                    "neutrality_score": 1.0,
                },
                sarcasm_gate_reason="empty_text",
                quoted_attack_detected=False,
                attack_endorsement_detected=False,
                attack_rejection_detected=False,
                contrast_rejection_detected=False,
                fairness_defense_score=0.0,
                reputation_defense_score=0.0,
                criticism_reference_score=0.0,
                context_change_score=0.0,
                evaluation_redirection_score=0.0,
                credibility_restoration_score=0.0,
                contrast_structure_score=0.0,
                causal_defense_score=0.0,
                credibility_defense_score=0.0,
                supportive_defense_strength=0.0,
                paragraph_stances=[],
                overall_stance={},
            )

        quote_flags = self._quote_awareness(lowered)
        fairness_defense_score = self.detect_fairness_defense_patterns(lowered)
        reputation_defense_score = self.detect_reputation_defense_patterns(lowered)
        criticism_reference_score = self.detect_criticism_reference_patterns(lowered)
        context_change_score = self.detect_context_change_patterns(lowered)
        evaluation_redirection_score = self.detect_evaluation_redirection_patterns(lowered)
        credibility_restoration_score = self.detect_credibility_restoration_patterns(lowered)
        contrast_structure_score = self.detect_contrast_structure_patterns(lowered)

        causal_defense_score = max(
            0.0,
            min(
                1.0,
                0.40 * criticism_reference_score
                + 0.30 * context_change_score
                + 0.30 * evaluation_redirection_score,
            ),
        )
        credibility_defense_score = max(
            0.0,
            min(1.0, 0.60 * credibility_restoration_score + 0.40 * contrast_structure_score),
        )
        sarcasm_features = self._sarcasm_features(lowered)
        sarcasm_score = float(sarcasm_features["sarcasm_score"])
        exaggeration_score = float(sarcasm_features["exaggeration_score"])
        contradiction_score = float(sarcasm_features["contradiction_score"])
        context_mismatch_score = float(sarcasm_features["context_mismatch_score"])
        praise_in_mocking_context_score = float(sarcasm_features["praise_in_mocking_context_score"])

        rule_based_supportive_strength = max(
            0.0,
            min(
                1.0,
                0.30 * fairness_defense_score
                + 0.20 * reputation_defense_score
                + 0.30 * causal_defense_score
                + 0.20 * credibility_defense_score,
            ),
        )

        semantic_similarities = self._semantic_similarities(text, embedder)
        semantic_supportive_similarity = float(semantic_similarities["semantic_supportive_similarity"])
        semantic_constructive_similarity = float(semantic_similarities["semantic_constructive_similarity"])
        semantic_balanced_similarity = float(semantic_similarities["semantic_balanced_similarity"])

        # Semantic subtype signals from benchmark centroids.
        semantic_criticism_reference_score = max(criticism_reference_score, semantic_constructive_similarity)
        semantic_explanation_depth_score = max(self._explanation_depth_score(text), semantic_balanced_similarity)
        fan_sincerity_score = max(
            self._signal_score(lowered, self.SUPPORTIVE_SIGNALS, normalizer=2.5),
            semantic_supportive_similarity + semantic_balanced_similarity + semantic_constructive_similarity,
        )

        fairness_defense_strength = semantic_supportive_similarity * semantic_criticism_reference_score
        credibility_defense_strength = semantic_supportive_similarity * semantic_explanation_depth_score
        fan_expression_strength = semantic_supportive_similarity * fan_sincerity_score

        semantic_strength = (
            0.4 * fairness_defense_strength
            + 0.4 * credibility_defense_strength
            + 0.2 * fan_expression_strength
        )
        semantic_strength = max(0.0, min(1.0, semantic_strength))
        semantic_supportive_strength = semantic_strength

        supportive_defense_strength = max(
            0.0,
            min(
                1.0,
                0.5 * semantic_strength + 0.5 * rule_based_supportive_strength,
            ),
        )

        if quote_flags["contrast_rejection_detected"] and fairness_defense_score >= 0.4:
            quote_flags["attack_rejection_detected"] = True

        supportive_score = self._signal_score(lowered, self.SUPPORTIVE_SIGNALS, normalizer=2.5)
        constructive_score = self._signal_score(lowered, self.CONSTRUCTIVE_SIGNALS, normalizer=2.5)
        suggestion_score = self._signal_score(lowered, self.SUGGESTION_SIGNALS, normalizer=3.0)
        complaint_score = self._signal_score(lowered, self.COMPLAINT_SIGNALS, normalizer=2.0)
        attack_score = self._signal_score(lowered, self.ATTACK_SIGNALS, normalizer=2.0)
        debate_balance = self._signal_score(lowered, ["on one hand", "on the other hand", "some believe", "others argue", "depends on"], normalizer=2.0)
        discourse_score = self._signal_score(lowered, self.CONTRAST_MARKERS + ["because", "as a result", "this means"], normalizer=4.0)

        frustration_marker_score = self._signal_score(
            lowered,
            [
                "again and again",
                "same confusion",
                "repeating mistakes",
                "frustrating to watch",
                "ignored repeatedly",
                "again and again ignored",
                "same issue again",
                "same mistake again",
                "no clear plan",
                "no long-term plan",
                "repeated confusion",
            ],
            normalizer=1.0,
        )
        if frustration_marker_score > 0.0:
            complaint_score = max(complaint_score, 0.65)

        direct_attack_selection_score = self._signal_score(
            lowered,
            [
                "clearly not performing",
                "management ignoring problems",
                "keeps selecting players who are not performing",
                "issues are being ignored",
                "problems are being ignored",
                "obvious issues ignored",
                "selectors ignoring",
                "management ignoring",
                "selection mistakes continue",
                "selection problems continue",
            ],
            normalizer=1.0,
        )
        if direct_attack_selection_score > 0.0:
            attack_score = max(attack_score, 0.65)

        if quote_flags["attack_rejection_detected"]:
            attack_score *= 0.25
            supportive_score = min(1.0, supportive_score + 0.25)

        supportive_score = min(1.0, supportive_score + 0.25 * supportive_defense_strength)

        if quote_flags["attack_endorsement_detected"]:
            attack_score = min(1.0, attack_score + 0.4)

        embedding_scores = self._embedding_scores(text, embedder)
        benchmark_probs = self._predict_benchmark_probs(text, embedder)

        raw_scores = {
            "SUPPORTIVE_DEFENSE": 0.55 * supportive_score + 0.3 * embedding_scores.get("SUPPORTIVE_DEFENSE", 0.0) + 0.15 * discourse_score,
            "CONSTRUCTIVE_CRITICISM": 0.55 * constructive_score + 0.3 * embedding_scores.get("CONSTRUCTIVE_CRITICISM", 0.0) + 0.15 * discourse_score,
            "NEUTRAL_ANALYSIS": 0.45 * max(0.0, 1.0 - (supportive_score + constructive_score + complaint_score + attack_score) / 3.0)
            + 0.4 * embedding_scores.get("NEUTRAL_ANALYSIS", 0.0)
            + 0.15 * (1.0 - debate_balance),
            "BALANCED_DEBATE": 0.55 * debate_balance + 0.3 * embedding_scores.get("BALANCED_DEBATE", 0.0) + 0.15 * discourse_score,
            "DISMISSIVE_COMPLAINT": 0.6 * complaint_score + 0.3 * embedding_scores.get("DISMISSIVE_COMPLAINT", 0.0) + 0.1 * (1.0 - discourse_score),
            "DIRECT_ATTACK": 0.6 * attack_score + 0.3 * embedding_scores.get("DIRECT_ATTACK", 0.0) + 0.1 * complaint_score,
            "SARCASTIC_CRITICISM": 0.45 * sarcasm_score + 0.25 * attack_score + 0.15 * complaint_score + 0.15 * (1.0 - supportive_score),
            "MIXED_STANCE": 0.35 * (supportive_score + constructive_score + debate_balance) / 3.0 + 0.35 * embedding_scores.get("MIXED_STANCE", 0.0) + 0.3 * discourse_score,
        }

        supportive_defense_probability = float(raw_scores.get("SUPPORTIVE_DEFENSE", 0.0))
        credibility_defense_probability = float(credibility_defense_score)
        attack_probability = float(raw_scores.get("DIRECT_ATTACK", 0.0))
        if sarcasm_score >= 0.4:
            supportive_defense_probability = max(0.0, supportive_defense_probability - 0.2)
            credibility_defense_probability = max(0.0, credibility_defense_probability - 0.1)
            attack_probability = min(1.0, attack_probability + 0.15)
            raw_scores["SUPPORTIVE_DEFENSE"] = supportive_defense_probability
            raw_scores["DIRECT_ATTACK"] = attack_probability
            credibility_defense_score = credibility_defense_probability

        explicit_supportive_context_score = max(
            supportive_score,
            rule_based_supportive_strength,
            fairness_defense_score,
            reputation_defense_score,
            causal_defense_score,
            credibility_defense_score,
        )
        stronger_explicit_supportive = explicit_supportive_context_score >= 0.72
        sarcasm_gate_triggered = False
        sarcasm_gate_reason = "score_below_threshold"
        if sarcasm_score >= self._sarcasm_threshold:
            if stronger_explicit_supportive:
                sarcasm_gate_reason = "blocked_by_stronger_explicit_supportive_context"
            else:
                sarcasm_gate_triggered = True
                sarcasm_gate_reason = "sarcasm_threshold_met_and_no_stronger_supportive_context"
                raw_scores["SUPPORTIVE_DEFENSE"] = max(0.0, float(raw_scores.get("SUPPORTIVE_DEFENSE", 0.0)) - 0.25)
                raw_scores["SARCASTIC_CRITICISM"] = max(float(raw_scores.get("SARCASTIC_CRITICISM", 0.0)), 0.55 + 0.35 * sarcasm_score)
                raw_scores["DIRECT_ATTACK"] = max(float(raw_scores.get("DIRECT_ATTACK", 0.0)), 0.25 + 0.5 * sarcasm_score)

        ridicule_score = max(
            0.0,
            min(
                1.0,
                0.4 * sarcasm_score + 0.3 * exaggeration_score + 0.3 * praise_in_mocking_context_score,
            ),
        )
        directness_score = max(0.0, min(1.0, 0.7 * attack_score + 0.3 * self._signal_score(lowered, ["must", "clearly", "obvious", "definitely"], normalizer=2.0)))
        certainty_score = max(0.0, min(1.0, self._signal_score(lowered, ["clearly", "definitely", "obviously", "surely", "always", "never"], normalizer=2.0)))
        hostility_score = max(0.0, min(1.0, 0.65 * attack_score + 0.35 * complaint_score))
        neutrality_score = max(
            0.0,
            min(
                1.0,
                0.3 * (1.0 - max(supportive_score, attack_score, complaint_score, sarcasm_score))
                + 0.7 * self._signal_score(lowered, ["analysts", "report", "according to", "some believe", "others believe"], normalizer=2.0),
            ),
        )

        if fairness_defense_score >= 0.4 or reputation_defense_score >= 0.4:
            raw_scores["SUPPORTIVE_DEFENSE"] = min(1.0, float(raw_scores.get("SUPPORTIVE_DEFENSE", 0.0)) + 0.25)

        if causal_defense_score >= 0.35:
            raw_scores["SUPPORTIVE_DEFENSE"] = min(1.0, float(raw_scores.get("SUPPORTIVE_DEFENSE", 0.0)) + 0.25)
        if credibility_defense_score >= 0.35:
            raw_scores["SUPPORTIVE_DEFENSE"] = min(1.0, float(raw_scores.get("SUPPORTIVE_DEFENSE", 0.0)) + 0.20)
        if evaluation_redirection_score >= 0.35 and credibility_defense_score >= 0.35:
            raw_scores["SUPPORTIVE_DEFENSE"] = min(1.0, float(raw_scores.get("SUPPORTIVE_DEFENSE", 0.0)) + 0.15)

        force_supportive_priority = (
            causal_defense_score >= 0.35
            and context_change_score >= 0.3
            and evaluation_redirection_score >= 0.3
        )
        if force_supportive_priority:
            raw_scores["SUPPORTIVE_DEFENSE"] = min(1.0, float(raw_scores.get("SUPPORTIVE_DEFENSE", 0.0)) + 0.20)

        blended_scores = {
            label: 0.35 * float(raw_scores.get(label, 0.0)) + 0.65 * float(benchmark_probs.get(label, 0.0))
            for label in self.STANCE_LABELS
        }

        if sarcasm_gate_triggered:
            stance_label = "SARCASTIC_CRITICISM"
            blended_scores["SARCASTIC_CRITICISM"] = max(blended_scores.get("SARCASTIC_CRITICISM", 0.0), 0.78)
        elif quote_flags["attack_endorsement_detected"]:
            stance_label = "DIRECT_ATTACK"
            blended_scores["DIRECT_ATTACK"] = max(blended_scores.get("DIRECT_ATTACK", 0.0), 0.9)
        elif attack_score >= 0.85:
            stance_label = "DIRECT_ATTACK"
            blended_scores["DIRECT_ATTACK"] = max(blended_scores.get("DIRECT_ATTACK", 0.0), 0.85)
        elif quote_flags["attack_rejection_detected"] and causal_defense_score >= 0.3:
            stance_label = "SUPPORTIVE_DEFENSE"
            blended_scores["SUPPORTIVE_DEFENSE"] = max(blended_scores.get("SUPPORTIVE_DEFENSE", 0.0), 0.8)
        else:
            ranked = sorted(blended_scores.items(), key=lambda item: item[1], reverse=True)
            stance_label = ranked[0][0] if ranked else "NEUTRAL_ANALYSIS"
            if ranked and ranked[0][1] < 0.28:
                stance_label = "MIXED_STANCE"

        pre_override_probs = self._normalize(blended_scores)
        pre_override_confidence = float(pre_override_probs.get(stance_label, 0.0))
        semantic_override_triggered = False
        semantic_precedence_override = (
            not sarcasm_gate_triggered
            and supportive_defense_strength >= 0.30
            and semantic_supportive_similarity >= semantic_constructive_similarity - 0.03
            and attack_score < 0.40
            and complaint_score < 0.40
        )
        if semantic_precedence_override:
            stance_label = "SUPPORTIVE_DEFENSE"

        semantic_compare_constructive = semantic_constructive_similarity - 0.002
        semantic_routing_override = (
            not semantic_override_triggered
            and not sarcasm_gate_triggered
            and not semantic_precedence_override
            and pre_override_confidence < 0.55
            and semantic_supportive_strength >= 0.25
            and supportive_defense_strength >= 0.25
            and attack_score < 0.40
            and complaint_score < 0.40
            and semantic_compare_constructive < semantic_supportive_similarity
        )
        if not semantic_override_triggered and semantic_routing_override:
            stance_label = "SUPPORTIVE_DEFENSE"
            print("SEMANTIC ROUTING OVERRIDE ACTIVATED")

        probs = self._normalize(blended_scores)
        stance_confidence = float(probs.get(stance_label, 0.0))
        if semantic_precedence_override and stance_label == "SUPPORTIVE_DEFENSE":
            stance_confidence = max(pre_override_confidence, stance_confidence, 0.55)
            semantic_override_triggered = True
        if semantic_routing_override and stance_label == "SUPPORTIVE_DEFENSE":
            stance_confidence = max(float(probs.get("SUPPORTIVE_DEFENSE", 0.0)), 0.55)
        if force_supportive_priority and stance_label == "SUPPORTIVE_DEFENSE":
            stance_confidence = max(float(probs.get("SUPPORTIVE_DEFENSE", 0.0)), 0.6)

        supportive_signal = max(
            0.0,
            min(
                1.0,
                0.5 * float(raw_scores.get("SUPPORTIVE_DEFENSE", 0.0))
                + 0.2 * supportive_defense_strength
                + 0.3 * semantic_supportive_similarity,
            ),
        )
        criticism_signal = max(
            float(raw_scores.get("DIRECT_ATTACK", 0.0)),
            float(raw_scores.get("DISMISSIVE_COMPLAINT", 0.0)),
            float(raw_scores.get("CONSTRUCTIVE_CRITICISM", 0.0)) * 0.85,
            0.7 * attack_score + 0.3 * complaint_score,
        )
        sarcastic_signal = max(float(raw_scores.get("SARCASTIC_CRITICISM", 0.0)), sarcasm_score)

        primary_stance_label, primary_stance_probabilities, primary_resolution_reason = self._resolve_primary_stance(
            supportive_signal=supportive_signal,
            criticism_signal=criticism_signal,
            sarcastic_signal=sarcastic_signal,
            neutrality_score=neutrality_score,
            sarcasm_score=sarcasm_score,
            contradiction_score=contradiction_score,
            context_mismatch_score=context_mismatch_score,
            sarcasm_gate_triggered=sarcasm_gate_triggered,
            stronger_explicit_supportive=stronger_explicit_supportive,
        )
        if (
            not sarcasm_gate_triggered
            and sarcasm_score < self._sarcasm_threshold
            and supportive_signal < 0.45
            and stance_label in {"DIRECT_ATTACK", "DISMISSIVE_COMPLAINT", "CONSTRUCTIVE_CRITICISM"}
        ):
            primary_stance_label = "DIRECT_CRITICISM"
            primary_resolution_reason = "legacy_critical_route_alignment"
        if (
            "but" in lowered
            and "still" in lowered
            and supportive_signal >= 0.3
            and criticism_signal >= 0.2
            and not sarcasm_gate_triggered
        ):
            primary_stance_label = "MIXED_OR_AMBIGUOUS"
            primary_resolution_reason = "contrastive_mixed_intent_pattern"
        style_tags = self._style_tags(
            text=text,
            sarcasm_score=sarcasm_score,
            exaggeration_score=exaggeration_score,
            contradiction_score=contradiction_score,
            context_mismatch_score=context_mismatch_score,
            directness_score=directness_score,
        )
        primary_stance_confidence = float(primary_stance_probabilities.get(primary_stance_label, 0.0))
        score_shifts = {
            "supportive_shift": round(float(raw_scores.get("SUPPORTIVE_DEFENSE", 0.0)) - supportive_score, 4),
            "attack_shift": round(float(raw_scores.get("DIRECT_ATTACK", 0.0)) - attack_score, 4),
            "credibility_shift": round(credibility_defense_score - float(self.detect_credibility_restoration_patterns(lowered)), 4),
        }

        if os.getenv("STANCE_DEBUG", "0") == "1":
            print(
                "STANCE DEBUG:",
                round(criticism_reference_score, 4),
                round(context_change_score, 4),
                round(evaluation_redirection_score, 4),
                round(causal_defense_score, 4),
                round(suggestion_score, 4),
                round(complaint_score, 4),
                round(attack_score, 4),
            )

            print("SEMANTIC SUPPORTIVE:", round(semantic_supportive_strength, 4))
            print(
                "SARCASM DEBUG:",
                round(sarcasm_score, 4),
                round(exaggeration_score, 4),
                round(contradiction_score, 4),
                round(context_mismatch_score, 4),
                round(praise_in_mocking_context_score, 4),
                round(ridicule_score, 4),
                round(directness_score, 4),
                round(certainty_score, 4),
                round(hostility_score, 4),
                round(neutrality_score, 4),
                f"gate={sarcasm_gate_triggered}",
                f"reason={sarcasm_gate_reason}",
                f"threshold={round(self._sarcasm_threshold, 4)}",
                f"primary={primary_stance_label}",
                f"primary_reason={primary_resolution_reason}",
                f"styles={style_tags}",
                f"legacy={stance_label}",
                f"score_shifts={score_shifts}",
            )

        return StanceResult(
            stance_label=stance_label,
            stance_probabilities=probs,
            stance_confidence=round(stance_confidence, 4),
            primary_stance_label=primary_stance_label,
            primary_stance_probabilities=primary_stance_probabilities,
            primary_stance_confidence=round(primary_stance_confidence, 4),
            style_tags=style_tags,
            scalar_metrics={
                "sarcasm_score": round(sarcasm_score, 4),
                "contradiction_score": round(contradiction_score, 4),
                "exaggeration_score": round(exaggeration_score, 4),
                "context_mismatch_score": round(context_mismatch_score, 4),
                "ridicule_score": round(ridicule_score, 4),
                "directness_score": round(directness_score, 4),
                "certainty_score": round(certainty_score, 4),
                "hostility_score": round(hostility_score, 4),
                "neutrality_score": round(neutrality_score, 4),
            },
            sarcasm_gate_reason=sarcasm_gate_reason,
            quoted_attack_detected=quote_flags["quoted_attack_detected"],
            attack_endorsement_detected=quote_flags["attack_endorsement_detected"],
            attack_rejection_detected=quote_flags["attack_rejection_detected"],
            contrast_rejection_detected=quote_flags["contrast_rejection_detected"],
            fairness_defense_score=round(fairness_defense_score, 4),
            reputation_defense_score=round(reputation_defense_score, 4),
            criticism_reference_score=round(criticism_reference_score, 4),
            context_change_score=round(context_change_score, 4),
            evaluation_redirection_score=round(evaluation_redirection_score, 4),
            credibility_restoration_score=round(credibility_restoration_score, 4),
            contrast_structure_score=round(contrast_structure_score, 4),
            causal_defense_score=round(causal_defense_score, 4),
            credibility_defense_score=round(credibility_defense_score, 4),
            supportive_defense_strength=round(supportive_defense_strength, 4),
            paragraph_stances=[],
            overall_stance={},
        )

    def _embedding_scores(self, text: str, embedder) -> Dict[str, float]:
        if embedder is None or not text.strip():
            return {label: 0.0 for label in self.STANCE_LABELS}

        text_vec = embedder.encode(text, convert_to_tensor=True)
        scores: Dict[str, float] = {}
        for label, examples in self.PROTOTYPES.items():
            vecs = embedder.encode(examples, convert_to_tensor=True)
            sims = util.cos_sim(text_vec, vecs)[0].cpu().numpy()
            top = float(np.max(sims))
            scores[label] = max(0.0, min(1.0, (top - 0.15) / 0.65))
        return scores

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        labels = [label for label in StanceDetector.STANCE_LABELS]
        values = np.array([max(0.0, float(scores.get(label, 0.0))) for label in labels], dtype=float)
        if float(values.sum()) <= 0.0:
            values = np.ones_like(values)
        probs = values / float(values.sum())
        return {label: round(float(prob), 4) for label, prob in zip(labels, probs)}

    def detect(self, text: str, embedder=None) -> StanceResult:
        paragraphs = self._split_paragraphs(text)
        if len(paragraphs) <= 1:
            single = self._detect_single(text=text, embedder=embedder)
            single.paragraph_stances = [
                {
                    "paragraph_index": 0,
                    "text": text.strip(),
                    "weight": 1.0,
                    "explanation_depth_score": round(self._explanation_depth_score(text), 4),
                    "stance_label": single.stance_label,
                    "primary_stance_label": single.primary_stance_label,
                    "stance_confidence": single.stance_confidence,
                    "primary_stance_confidence": single.primary_stance_confidence,
                    "stance_probabilities": single.stance_probabilities,
                    "primary_stance_probabilities": single.primary_stance_probabilities,
                    "style_tags": single.style_tags,
                    "scalar_metrics": single.scalar_metrics,
                    "supportive_defense_strength": single.supportive_defense_strength,
                    "criticism_reference_score": single.criticism_reference_score,
                    "context_change_score": single.context_change_score,
                    "evaluation_redirection_score": single.evaluation_redirection_score,
                    "credibility_restoration_score": single.credibility_restoration_score,
                    "contrast_structure_score": single.contrast_structure_score,
                    "causal_defense_score": single.causal_defense_score,
                    "credibility_defense_score": single.credibility_defense_score,
                }
            ]
            single.overall_stance = {
                "stance_label": single.stance_label,
                "stance_confidence": single.stance_confidence,
                "stance_probabilities": single.stance_probabilities,
                "primary_stance_label": single.primary_stance_label,
                "primary_stance_confidence": single.primary_stance_confidence,
                "primary_stance_probabilities": single.primary_stance_probabilities,
                "style_tags": single.style_tags,
                "scalar_metrics": single.scalar_metrics,
                "sarcasm_gate_reason": single.sarcasm_gate_reason,
            }
            return single

        paragraph_results = [self._detect_single(paragraph_text, embedder=embedder) for paragraph_text in paragraphs]
        paragraph_stances: List[Dict[str, Any]] = []
        weights: List[float] = []
        weighted_prob_sums = {label: 0.0 for label in self.STANCE_LABELS}
        weighted_primary_prob_sums = {label: 0.0 for label in self.PRIMARY_STANCE_LABELS}

        weighted_supportive_strength = 0.0
        weighted_fairness = 0.0
        weighted_reputation = 0.0
        weighted_criticism_reference = 0.0
        weighted_context_change = 0.0
        weighted_evaluation_redirection = 0.0
        weighted_credibility_restoration = 0.0
        weighted_contrast_structure = 0.0
        weighted_causal_defense = 0.0
        weighted_credibility_defense = 0.0

        quoted_attack_detected = False
        attack_endorsement_detected = False
        attack_rejection_detected = False
        contrast_rejection_detected = False

        for idx, (paragraph_text, paragraph_result) in enumerate(zip(paragraphs, paragraph_results)):
            paragraph_length = float(len(re.findall(r"[a-zA-Z']+", paragraph_text)))
            explanation_depth_score = self._explanation_depth_score(paragraph_text)
            weight = max(1.0, paragraph_length) * max(0.05, explanation_depth_score)
            weights.append(weight)

            for label in self.STANCE_LABELS:
                weighted_prob_sums[label] += weight * float(paragraph_result.stance_probabilities.get(label, 0.0))
            for label in self.PRIMARY_STANCE_LABELS:
                weighted_primary_prob_sums[label] += weight * float(paragraph_result.primary_stance_probabilities.get(label, 0.0))

            weighted_supportive_strength += weight * paragraph_result.supportive_defense_strength
            weighted_fairness += weight * paragraph_result.fairness_defense_score
            weighted_reputation += weight * paragraph_result.reputation_defense_score
            weighted_criticism_reference += weight * paragraph_result.criticism_reference_score
            weighted_context_change += weight * paragraph_result.context_change_score
            weighted_evaluation_redirection += weight * paragraph_result.evaluation_redirection_score
            weighted_credibility_restoration += weight * paragraph_result.credibility_restoration_score
            weighted_contrast_structure += weight * paragraph_result.contrast_structure_score
            weighted_causal_defense += weight * paragraph_result.causal_defense_score
            weighted_credibility_defense += weight * paragraph_result.credibility_defense_score

            quoted_attack_detected = quoted_attack_detected or paragraph_result.quoted_attack_detected
            attack_endorsement_detected = attack_endorsement_detected or paragraph_result.attack_endorsement_detected
            attack_rejection_detected = attack_rejection_detected or paragraph_result.attack_rejection_detected
            contrast_rejection_detected = contrast_rejection_detected or paragraph_result.contrast_rejection_detected

            paragraph_stances.append(
                {
                    "paragraph_index": idx,
                    "text": paragraph_text,
                    "weight": round(weight, 4),
                    "explanation_depth_score": round(explanation_depth_score, 4),
                    "stance_label": paragraph_result.stance_label,
                    "primary_stance_label": paragraph_result.primary_stance_label,
                    "stance_confidence": paragraph_result.stance_confidence,
                    "primary_stance_confidence": paragraph_result.primary_stance_confidence,
                    "stance_probabilities": paragraph_result.stance_probabilities,
                    "primary_stance_probabilities": paragraph_result.primary_stance_probabilities,
                    "style_tags": paragraph_result.style_tags,
                    "scalar_metrics": paragraph_result.scalar_metrics,
                    "supportive_defense_strength": paragraph_result.supportive_defense_strength,
                    "criticism_reference_score": paragraph_result.criticism_reference_score,
                    "context_change_score": paragraph_result.context_change_score,
                    "evaluation_redirection_score": paragraph_result.evaluation_redirection_score,
                    "credibility_restoration_score": paragraph_result.credibility_restoration_score,
                    "contrast_structure_score": paragraph_result.contrast_structure_score,
                    "causal_defense_score": paragraph_result.causal_defense_score,
                    "credibility_defense_score": paragraph_result.credibility_defense_score,
                }
            )

        total_weight = max(1e-6, float(np.sum(weights)))
        overall_probs = {label: round(max(0.0, weighted_prob_sums[label] / total_weight), 4) for label in self.STANCE_LABELS}
        overall_probs = self._normalize(overall_probs)

        overall_primary_probs = {
            label: round(max(0.0, weighted_primary_prob_sums[label] / total_weight), 4)
            for label in self.PRIMARY_STANCE_LABELS
        }
        primary_total = float(sum(overall_primary_probs.values()))
        if primary_total <= 0.0:
            overall_primary_probs = {label: round(1.0 / len(self.PRIMARY_STANCE_LABELS), 4) for label in self.PRIMARY_STANCE_LABELS}
        else:
            overall_primary_probs = {
                label: round(float(overall_primary_probs[label] / primary_total), 4)
                for label in self.PRIMARY_STANCE_LABELS
            }

        stance_label = max(overall_probs.items(), key=lambda item: item[1])[0]
        stance_confidence = float(overall_probs.get(stance_label, 0.0))
        primary_stance_label = max(overall_primary_probs.items(), key=lambda item: item[1])[0]
        primary_stance_confidence = float(overall_primary_probs.get(primary_stance_label, 0.0))

        style_count: Dict[str, int] = {}
        for paragraph_result in paragraph_results:
            for tag in paragraph_result.style_tags:
                style_count[tag] = style_count.get(tag, 0) + 1
        style_tags = [tag for tag, count in sorted(style_count.items(), key=lambda item: (-item[1], item[0])) if count >= 1]

        scalar_metrics_keys = [
            "sarcasm_score",
            "contradiction_score",
            "exaggeration_score",
            "context_mismatch_score",
            "ridicule_score",
            "directness_score",
            "certainty_score",
            "hostility_score",
            "neutrality_score",
        ]
        scalar_metrics = {}
        for key in scalar_metrics_keys:
            scalar_metrics[key] = round(
                float(np.mean([float(paragraph_result.scalar_metrics.get(key, 0.0)) for paragraph_result in paragraph_results])) if paragraph_results else 0.0,
                4,
            )
        sarcasm_gate_reason = "paragraph_aggregated"

        return StanceResult(
            stance_label=stance_label,
            stance_probabilities=overall_probs,
            stance_confidence=round(stance_confidence, 4),
            primary_stance_label=primary_stance_label,
            primary_stance_probabilities=overall_primary_probs,
            primary_stance_confidence=round(primary_stance_confidence, 4),
            style_tags=style_tags,
            scalar_metrics=scalar_metrics,
            sarcasm_gate_reason=sarcasm_gate_reason,
            quoted_attack_detected=quoted_attack_detected,
            attack_endorsement_detected=attack_endorsement_detected,
            attack_rejection_detected=attack_rejection_detected,
            contrast_rejection_detected=contrast_rejection_detected,
            fairness_defense_score=round(weighted_fairness / total_weight, 4),
            reputation_defense_score=round(weighted_reputation / total_weight, 4),
            criticism_reference_score=round(weighted_criticism_reference / total_weight, 4),
            context_change_score=round(weighted_context_change / total_weight, 4),
            evaluation_redirection_score=round(weighted_evaluation_redirection / total_weight, 4),
            credibility_restoration_score=round(weighted_credibility_restoration / total_weight, 4),
            contrast_structure_score=round(weighted_contrast_structure / total_weight, 4),
            causal_defense_score=round(weighted_causal_defense / total_weight, 4),
            credibility_defense_score=round(weighted_credibility_defense / total_weight, 4),
            supportive_defense_strength=round(weighted_supportive_strength / total_weight, 4),
            paragraph_stances=paragraph_stances,
            overall_stance={
                "stance_label": stance_label,
                "stance_confidence": round(stance_confidence, 4),
                "stance_probabilities": overall_probs,
                "primary_stance_label": primary_stance_label,
                "primary_stance_confidence": round(primary_stance_confidence, 4),
                "primary_stance_probabilities": overall_primary_probs,
                "style_tags": style_tags,
                "scalar_metrics": scalar_metrics,
                "sarcasm_gate_reason": sarcasm_gate_reason,
            },
        )

    def detect_as_dict(self, text: str, embedder=None) -> Dict[str, object]:
        result = self.detect(text=text, embedder=embedder)
        return {
            "stance_label": result.stance_label,
            "stance_probabilities": result.stance_probabilities,
            "stance_confidence": result.stance_confidence,
            "primary_stance_label": result.primary_stance_label,
            "primary_stance_probabilities": result.primary_stance_probabilities,
            "primary_stance_confidence": result.primary_stance_confidence,
            "style_tags": result.style_tags,
            "scalar_metrics": result.scalar_metrics,
            "sarcasm_gate_reason": result.sarcasm_gate_reason,
            "quoted_attack_detected": result.quoted_attack_detected,
            "attack_endorsement_detected": result.attack_endorsement_detected,
            "attack_rejection_detected": result.attack_rejection_detected,
            "contrast_rejection_detected": result.contrast_rejection_detected,
            "fairness_defense_score": result.fairness_defense_score,
            "reputation_defense_score": result.reputation_defense_score,
            "criticism_reference_score": result.criticism_reference_score,
            "context_change_score": result.context_change_score,
            "evaluation_redirection_score": result.evaluation_redirection_score,
            "credibility_restoration_score": result.credibility_restoration_score,
            "contrast_structure_score": result.contrast_structure_score,
            "causal_defense_score": result.causal_defense_score,
            "credibility_defense_score": result.credibility_defense_score,
            "supportive_defense_strength": result.supportive_defense_strength,
            "paragraph_stances": result.paragraph_stances,
            "overall_stance": result.overall_stance,
        }
