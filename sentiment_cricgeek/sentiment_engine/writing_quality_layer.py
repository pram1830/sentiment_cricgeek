from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import util


@dataclass
class WritingQualitySignals:
    coherence_score: float
    lexical_diversity_score: float
    sentence_variety_score: float
    repetition_penalty: float
    position_clarity_score: float
    counter_argument_score: float
    evidence_presence_score: float
    completeness_score: float
    information_density_score: float
    argument_logic_score: float

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


POSITION_MARKERS = [
    "i believe",
    "in my view",
    "this suggests",
    "the point is",
    "overall",
    "i think",
    "in conclusion",
]

COUNTER_ARGUMENT_MARKERS = [
    "however",
    "although",
    "on the other hand",
    "some argue",
    "but",
    "while others",
    "nevertheless",
    "yet",
]

ARGUMENT_LOGIC_MARKERS = [
    "because",
    "therefore",
    "since",
    "as a result",
    "which means",
    "thus",
    "hence",
    "consequently",
]

EVIDENCE_KEYWORDS = [
    "strike rate",
    "average",
    "economy",
    "runs",
    "wickets",
    "balls",
    "overs",
    "innings",
    "powerplay",
    "match",
    "series",
    "season",
]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _split_sentences(text: str) -> List[str]:
    parts = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _spacy_pos_tags(text: str) -> Optional[List[Tuple[str, str]]]:
    try:
        import spacy  # type: ignore

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [(token.text.lower(), token.pos_) for token in doc if token.is_alpha]
    except Exception:
        return None


def _ngram_counts(tokens: Sequence[str], n: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if len(tokens) < n:
        return counts
    for idx in range(0, len(tokens) - n + 1):
        key = " ".join(tokens[idx:idx + n])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _coherence_score(sentences: Sequence[str], embedder) -> float:
    if len(sentences) <= 1:
        return 0.5
    if embedder is None:
        return 0.5

    vectors = embedder.encode(list(sentences), convert_to_tensor=True)
    sims: List[float] = []
    for idx in range(len(sentences) - 1):
        sim = float(util.cos_sim(vectors[idx], vectors[idx + 1]).cpu().item())
        sims.append(_clip((sim + 1.0) / 2.0, 0.0, 1.0))

    if not sims:
        return 0.5
    return _clip(float(np.mean(sims)), 0.0, 1.0)


def _lexical_diversity_score(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    ttr = len(set(tokens)) / float(len(tokens))
    repeated_ratio = 1.0 - ttr
    richness_adjustment = 1.0 - _clip(repeated_ratio * 0.65, 0.0, 0.35)
    return _clip(ttr * richness_adjustment, 0.0, 1.0)


def _sentence_variety_score(sentences: Sequence[str]) -> float:
    if not sentences:
        return 0.0
    lengths = [len(_tokenize_words(sentence)) for sentence in sentences]
    if len(lengths) <= 1:
        return 0.35

    std = float(np.std(lengths))
    mean_len = float(np.mean(lengths))
    coeff_var = 0.0 if mean_len <= 0 else std / mean_len

    patterns = []
    for sentence in sentences:
        words = _tokenize_words(sentence)
        if not words:
            continue
        starter = words[0]
        has_clause = any(token in words for token in ["because", "however", "although", "while", "therefore"])
        has_numeric = any(bool(re.search(r"\d", token)) for token in re.findall(r"\S+", sentence))
        patterns.append((starter, has_clause, has_numeric, len(words) > 20))
    pattern_diversity = len(set(patterns)) / float(max(1, len(patterns)))

    return _clip(0.55 * _clip(coeff_var * 1.8, 0.0, 1.0) + 0.45 * pattern_diversity, 0.0, 1.0)


def _repetition_penalty(tokens: Sequence[str]) -> float:
    if len(tokens) < 6:
        return 0.0

    bigrams = _ngram_counts(tokens, 2)
    trigrams = _ngram_counts(tokens, 3)

    repeated_bigrams = sum(count - 1 for count in bigrams.values() if count > 1)
    repeated_trigrams = sum(count - 1 for count in trigrams.values() if count > 1)

    total_bigrams = max(1, sum(bigrams.values()))
    total_trigrams = max(1, sum(trigrams.values()))

    bigram_repeat_ratio = repeated_bigrams / float(total_bigrams)
    trigram_repeat_ratio = repeated_trigrams / float(total_trigrams)

    unigram_counts: Dict[str, int] = {}
    for token in tokens:
        unigram_counts[token] = unigram_counts.get(token, 0) + 1
    repeated_unigrams = sum(count - 1 for count in unigram_counts.values() if count > 1)
    unigram_repeat_ratio = repeated_unigrams / float(max(1, len(tokens)))

    penalty = 0.35 * unigram_repeat_ratio + 0.4 * bigram_repeat_ratio + 0.25 * trigram_repeat_ratio
    return _clip(penalty * 2.0, 0.0, 1.0)


def _position_clarity_score(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for marker in POSITION_MARKERS if marker in lowered)
    return _clip(hits / 2.0, 0.0, 1.0)


def _counter_argument_score(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for marker in COUNTER_ARGUMENT_MARKERS if marker in lowered)
    return _clip(hits / 2.0, 0.0, 1.0)


def _evidence_presence_score(text: str) -> float:
    lowered = text.lower()

    numeric_hits = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", lowered))
    date_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", lowered))
    cricket_hits = sum(1 for marker in EVIDENCE_KEYWORDS if marker in lowered)
    comparison_hits = len(re.findall(r"\b(better than|worse than|compared to|compared with)\b", lowered))
    match_ref_hits = len(re.findall(r"\b(last\s+\d+\s+(?:matches?|innings?)|this\s+season|between\s+20\d{2}\s+and\s+20\d{2})\b", lowered))

    ner_like_hits = 0
    for name_like in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text):
        if len(name_like.split()) >= 2:
            ner_like_hits += 1

    evidence_count = numeric_hits + date_hits + cricket_hits + comparison_hits + match_ref_hits + min(ner_like_hits, 3)
    return _clip(evidence_count / 8.0, 0.0, 1.0)


def _completeness_score(text: str) -> float:
    lowered = text.lower()
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0

    intro_markers = ["first", "to begin", "in this", "consider", "regarding", "overall"]
    argument_markers = ["because", "this means", "which shows", "therefore", "as a result", "suggests"]
    conclusion_markers = ["in conclusion", "overall", "therefore", "this shows", "to sum up", "finally"]

    first = sentences[0].lower()
    middle = " ".join(sentences[1:-1]).lower() if len(sentences) > 2 else ""
    last = sentences[-1].lower()

    intro_hit = 1.0 if any(marker in first for marker in intro_markers) or len(first.split()) >= 6 else 0.0
    argument_hit = 1.0 if any(marker in lowered for marker in argument_markers) else 0.0
    if middle and any(marker in middle for marker in argument_markers):
        argument_hit = 1.0
    conclusion_hit = 1.0 if any(marker in last for marker in conclusion_markers) else 0.0

    return _clip((intro_hit + argument_hit + conclusion_hit) / 3.0, 0.0, 1.0)


def _pos_content_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0

    tagged = _spacy_pos_tags(" ".join(tokens))
    if tagged is not None:
        content_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
        content_count = sum(1 for _, pos in tagged if pos in content_pos)
        return _clip(content_count / float(max(1, len(tagged))), 0.0, 1.0)

    try:
        import nltk  # type: ignore

        tagged = nltk.pos_tag(list(tokens))
        content_tags = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS"}
        content_count = sum(1 for _, tag in tagged if tag in content_tags)
        return _clip(content_count / float(len(tokens)), 0.0, 1.0)
    except Exception:
        # Deterministic fallback when a POS tagger is unavailable.
        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "in", "on", "at", "to", "of", "for", "with", "as", "by", "is", "are", "was", "were", "be", "been", "being", "that", "this", "it", "he", "she", "they", "we", "you", "i",
        }
        content_count = 0
        for token in tokens:
            if token in stopwords:
                continue
            if token.endswith(("ing", "ed", "ly", "ive", "ous", "al", "tion", "ment", "ness")):
                content_count += 1
                continue
            if len(token) >= 5:
                content_count += 1
        return _clip(content_count / float(len(tokens)), 0.0, 1.0)


def _argument_logic_score(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for marker in ARGUMENT_LOGIC_MARKERS if marker in lowered)
    return _clip(hits / 3.0, 0.0, 1.0)


def compute_writing_quality_component(signals: Dict[str, float]) -> float:
    writing_quality_index = (
        0.15 * float(signals.get("coherence_score", 0.0))
        + 0.10 * float(signals.get("lexical_diversity_score", 0.0))
        + 0.10 * float(signals.get("sentence_variety_score", 0.0))
        + 0.15 * float(signals.get("argument_logic_score", 0.0))
        + 0.15 * float(signals.get("evidence_presence_score", 0.0))
        + 0.10 * float(signals.get("counter_argument_score", 0.0))
        + 0.10 * float(signals.get("completeness_score", 0.0))
        + 0.15 * float(signals.get("information_density_score", 0.0))
        - 0.10 * float(signals.get("repetition_penalty", 0.0))
    )
    component = (writing_quality_index * 22.0) - 7.0
    return _clip(component, -10.0, 15.0)


def compute_writing_quality_signals(paragraph_texts: Sequence[str], embedder=None) -> Dict[str, Any]:
    clean_paragraphs = [text.strip() for text in paragraph_texts if text and text.strip()]
    if not clean_paragraphs:
        empty = WritingQualitySignals(
            coherence_score=0.0,
            lexical_diversity_score=0.0,
            sentence_variety_score=0.0,
            repetition_penalty=0.0,
            position_clarity_score=0.0,
            counter_argument_score=0.0,
            evidence_presence_score=0.0,
            completeness_score=0.0,
            information_density_score=0.0,
            argument_logic_score=0.0,
        )
        return {
            "aggregate": empty.as_dict(),
            "paragraphs": [],
        }

    per_paragraph: List[Dict[str, Any]] = []
    weights: List[float] = []

    for paragraph in clean_paragraphs:
        sentences = _split_sentences(paragraph)
        tokens = _tokenize_words(paragraph)

        signals = WritingQualitySignals(
            coherence_score=_coherence_score(sentences, embedder),
            lexical_diversity_score=_lexical_diversity_score(tokens),
            sentence_variety_score=_sentence_variety_score(sentences),
            repetition_penalty=_repetition_penalty(tokens),
            position_clarity_score=_position_clarity_score(paragraph),
            counter_argument_score=_counter_argument_score(paragraph),
            evidence_presence_score=_evidence_presence_score(paragraph),
            completeness_score=_completeness_score(paragraph),
            information_density_score=_pos_content_ratio(tokens),
            argument_logic_score=_argument_logic_score(paragraph),
        )

        weight = max(1.0, float(len(tokens)))
        weights.append(weight)
        paragraph_component = compute_writing_quality_component(signals.as_dict())
        per_paragraph.append(
            {
                "text": paragraph,
                "weight": round(weight, 4),
                "signals": signals.as_dict(),
                "writing_quality_component": round(paragraph_component, 4),
            }
        )

    total_weight = max(1e-6, float(sum(weights)))
    keys = list(per_paragraph[0]["signals"].keys())
    aggregate: Dict[str, float] = {}

    for key in keys:
        weighted_sum = 0.0
        for item, weight in zip(per_paragraph, weights):
            weighted_sum += weight * float(item["signals"][key])
        aggregate[key] = round(_clip(weighted_sum / total_weight, 0.0, 1.0), 4)

    aggregate_component = compute_writing_quality_component(aggregate)

    return {
        "aggregate": aggregate,
        "paragraphs": per_paragraph,
        "aggregate_component": round(aggregate_component, 4),
    }
