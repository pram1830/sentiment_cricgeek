from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
import textstat
from sentence_transformers import util

from .deterministic_rescue_layer import apply_deterministic_rescue
from .paragraph_splitter import ParagraphUnit
from .writing_quality_layer import compute_writing_quality_signals


COMPONENT_MAX = {
    "constructiveness": 40.0,
    "respectfulness": 20.0,
    "analytical_tone": 15.0,
    "clarity": 10.0,
    "fan_sincerity": 15.0,
}

SCORE_MIN = 20.0
SCORE_MAX = 95.0

STANCE_MIDPOINTS = {
    "SUPPORTIVE_DEFENSE": 82.0,
    "CONSTRUCTIVE_CRITICISM": 78.0,
    "BALANCED_DEBATE": 76.0,
    "NEUTRAL_ANALYSIS": 64.0,
    "MIXED_STANCE": 55.0,
    "DISMISSIVE_COMPLAINT": 40.0,
    "DIRECT_ATTACK": 25.0,
}

STANCE_BLEND_CLASSES = list(STANCE_MIDPOINTS.keys())

WRITER_PROFILE = {
    "Passionate Fan": {
        "constructiveness": 1.12,
        "respectfulness": 1.1,
        "analytical_tone": 0.9,
        "clarity": 1.0,
        "fan_sincerity": 1.22,
        "toxicity_strength": 1.2,
    },
    "Analyst": {
        "constructiveness": 1.1,
        "respectfulness": 1.02,
        "analytical_tone": 1.24,
        "clarity": 1.12,
        "fan_sincerity": 0.85,
        "toxicity_strength": 1.2,
    },
    "Storyteller": {
        "constructiveness": 0.95,
        "respectfulness": 1.08,
        "analytical_tone": 0.92,
        "clarity": 1.2,
        "fan_sincerity": 1.15,
        "toxicity_strength": 1.2,
    },
    "Debater": {
        "constructiveness": 1.1,
        "respectfulness": 1.12,
        "analytical_tone": 1.04,
        "clarity": 1.06,
        "fan_sincerity": 0.95,
        "toxicity_strength": 1.2,
    },
    "All-Rounder": {
        "constructiveness": 1.0,
        "respectfulness": 1.0,
        "analytical_tone": 1.0,
        "clarity": 1.0,
        "fan_sincerity": 1.0,
        "toxicity_strength": 1.15,
    },
}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalized_stance_probabilities(stance_result: Dict[str, Any]) -> Dict[str, float]:
    raw = stance_result.get("stance_probabilities", {}) if isinstance(stance_result.get("stance_probabilities", {}), dict) else {}
    probs = {label: _clip(float(raw.get(label, 0.0)), 0.0, 1.0) for label in STANCE_BLEND_CLASSES}
    total = float(sum(probs.values()))
    if total <= 0.0:
        return probs
    return {label: probs[label] / total for label in STANCE_BLEND_CLASSES}


def _compute_stance_blending(stance_result: Dict[str, Any]) -> Dict[str, Any]:
    probs = _normalized_stance_probabilities(stance_result)
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    dominant_stance = ranked[0][0] if ranked else "NEUTRAL_ANALYSIS"
    top_probability = float(ranked[0][1]) if ranked else 0.0
    second_probability = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    probability_gap = _clip(top_probability - second_probability, 0.0, 1.0)

    dominant_stance_component = float(STANCE_MIDPOINTS.get(dominant_stance, STANCE_MIDPOINTS["NEUTRAL_ANALYSIS"]))
    soft_stance_component = float(sum(probs[label] * STANCE_MIDPOINTS[label] for label in STANCE_BLEND_CLASSES))

    if probability_gap >= 0.15:
        final_stance_component = 0.65 * dominant_stance_component + 0.35 * soft_stance_component
    else:
        final_stance_component = 0.45 * dominant_stance_component + 0.55 * soft_stance_component

    stance_label = str(stance_result.get("stance_label", "NEUTRAL_ANALYSIS"))
    hard_override = stance_label in {"DIRECT_ATTACK", "DISMISSIVE_COMPLAINT"}
    if hard_override:
        final_stance_component = dominant_stance_component

    final_stance_component = _clip(float(final_stance_component), SCORE_MIN, SCORE_MAX)

    return {
        "stance_probabilities": probs,
        "dominant_stance": dominant_stance,
        "top_probability": top_probability,
        "second_probability": second_probability,
        "probability_gap": probability_gap,
        "dominant_stance_component": dominant_stance_component,
        "soft_stance_component": soft_stance_component,
        "final_stance_component": final_stance_component,
        "hard_override": hard_override,
    }


def _normalize_label(label: str) -> str:
    return label.lower().strip().replace("-", "_").replace(" ", "_")


def _keyword_ratio(text: str, keywords: List[str]) -> float:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for token in words if token in keywords)
    return hits / len(words)


def _phrase_ratio(text: str, phrases: List[str]) -> float:
    lowered = text.lower()
    if not phrases:
        return 0.0
    hits = sum(1 for phrase in phrases if phrase in lowered)
    return hits / len(phrases)


def _extract_toxicity_score(raw_output: Any) -> float:
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
        pairs = {_normalize_label(item.get("label", "")): float(item.get("score", 0.0)) for item in raw_output[0]}
        if "toxic" in pairs:
            return pairs["toxic"]
        if "non_toxic" in pairs:
            return 1.0 - pairs["non_toxic"]

    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
        label = _normalize_label(raw_output[0].get("label", ""))
        score = float(raw_output[0].get("score", 0.0))
        if "non" in label and "toxic" in label:
            return 1.0 - score
        if "toxic" in label:
            return score

    return 0.0


def _extract_emotion_scores(raw_output: Any) -> Dict[str, float]:
    default = {"anger": 0.0, "sadness": 0.0, "fear": 0.0, "joy": 0.0, "love": 0.0}
    items = raw_output[0] if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list) else raw_output
    if not isinstance(items, list):
        return default
    for item in items:
        label = _normalize_label(item.get("label", ""))
        if label in default:
            default[label] = float(item.get("score", 0.0))
    return default


def _extract_sentiment_scores(raw_output: Any) -> Dict[str, float]:
    default = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    items = raw_output[0] if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list) else raw_output
    if not isinstance(items, list):
        return default
    for item in items:
        label = _normalize_label(item.get("label", ""))
        score = float(item.get("score", 0.0))
        if "neg" in label:
            default["negative"] = score
        elif "neu" in label:
            default["neutral"] = score
        elif "pos" in label:
            default["positive"] = score
    return default


def _clarity_signal(paragraph: str) -> float:
    if not paragraph.strip():
        return 0.0
    readability = textstat.flesch_reading_ease(paragraph)
    readability_norm = _clip((readability - 15.0) / 75.0, 0.0, 1.0)
    avg_sentence_len = textstat.avg_sentence_length(paragraph)
    sentence_penalty = _clip((avg_sentence_len - 30.0) / 22.0, 0.0, 1.0)
    return _clip(0.78 * readability_norm + 0.22 * (1.0 - sentence_penalty), 0.0, 1.0)


def _semantic_similarity(embedder, text: str, anchors: List[str]) -> float:
    if embedder is None or not text.strip() or not anchors:
        return 0.0
    text_vec = embedder.encode(text, convert_to_tensor=True)
    anchor_vecs = embedder.encode(anchors, convert_to_tensor=True)
    sims = util.cos_sim(text_vec, anchor_vecs)[0].cpu().numpy()
    return _clip((float(np.max(sims)) - 0.15) / 0.65, 0.0, 1.0)


def _abuse_indicator(text: str) -> float:
    abuse_words = [
        "idiot",
        "moron",
        "stupid",
        "trash",
        "garbage",
        "useless",
        "pathetic",
        "loser",
        "shut up",
        "disgusting",
        "hate you",
    ]
    personal_attack_patterns = [r"\byou are\b", r"\bhe is\b", r"\bthey are\b", r"\bthat player is\b"]

    phrase_score = _phrase_ratio(text, abuse_words)
    pattern_hits = sum(1 for pat in personal_attack_patterns if re.search(pat, text.lower()) is not None)
    return _clip(0.7 * phrase_score + 0.3 * min(1.0, pattern_hits / 2.0), 0.0, 1.0)


def analyze_paragraphs(paragraphs: List[ParagraphUnit], models: Any, constructiveness_detector=None) -> Dict[str, Any]:
    if not paragraphs:
        empty_components = {k: 0.0 for k in COMPONENT_MAX}
        return {
            "paragraph_diagnostics": [],
            "mean_components": empty_components,
            "toxicity": {"mean": 0.0, "max": 0.0, "ratio": 0.0, "abuse_mean": 0.0},
            "signals": {"mean_negativity": 0.0},
        }

    constructive_phrases = [
        "could improve",
        "should improve",
        "should focus",
        "need to",
        "needs to",
        "would help",
        "suggest",
        "recommend",
        "next match",
        "learn from",
        "improve",
        "fix",
    ]
    reasoning_terms = ["because", "therefore", "however", "so that", "if", "then", "reason", "impact"]
    civility_terms = ["respect", "fair", "appreciate", "credit", "please", "calm", "balanced"]
    fan_terms = ["we", "our", "i", "us", "support", "fans", "team", "believe", "care"]
    debate_terms = ["on the other hand", "although", "while", "whereas", "counter", "rebuttal", "yet"]
    constructive_anchors = [
        "This paragraph gives respectful and practical suggestions for improving cricket performance.",
        "The writer is critical but constructive, with clear next-step recommendations.",
    ]
    sincerity_anchors = [
        "This sounds like a genuine fan who cares about the team and wants them to improve.",
        "Emotional but supportive cricket writing with sincere commitment.",
    ]
    debate_anchors = [
        "This paragraph balances two sides of an argument and reaches a fair conclusion.",
        "The writer compares opposing views and argues constructively.",
    ]

    para_component_scores: Dict[str, List[float]] = {k: [] for k in COMPONENT_MAX}
    para_toxicity: List[float] = []
    para_abuse: List[float] = []
    para_negativity: List[float] = []
    para_sent_positive: List[float] = []
    para_sent_neutral: List[float] = []
    para_sent_negative: List[float] = []
    constructive_pattern_hits: List[float] = []
    reasoning_scores: List[float] = []
    suggestion_scores: List[float] = []
    explanation_depth_scores: List[float] = []
    strategic_reasoning_scores: List[float] = []
    discourse_scores: List[float] = []
    debate_style_scores: List[float] = []
    respect_scores: List[float] = []
    paragraph_word_counts: List[float] = []
    paragraph_sentence_counts: List[float] = []
    paragraph_diagnostics: List[Dict[str, Any]] = []

    for paragraph in paragraphs:
        text = paragraph.text

        tox_raw = models.toxicity_classifier(text)
        model_toxic = _clip(_extract_toxicity_score(tox_raw), 0.0, 1.0)
        abuse_signal = _abuse_indicator(text)

        emo_scores = _extract_emotion_scores(models.emotion_classifier(text))
        sent_scores = _extract_sentiment_scores(models.sentiment_classifier(text))

        negativity = _clip(
            0.3 * emo_scores["anger"] + 0.2 * emo_scores["sadness"] + 0.15 * emo_scores["fear"] + 0.35 * sent_scores["negative"],
            0.0,
            1.0,
        )

        # Toxicity prioritizes explicit abuse/degradation cues over generic negativity.
        effective_toxicity = _clip(0.25 * model_toxic + 0.75 * abuse_signal, 0.0, 1.0)

        existing_constructive_signal = _clip(
            0.28 * _phrase_ratio(text, constructive_phrases)
            + 0.24 * _keyword_ratio(text, reasoning_terms)
            + 0.24 * _clip(_keyword_ratio(text, ["improve", "fix", "plan", "adjust", "change", "rotate", "recover", "support"]) * 14.0, 0.0, 1.0)
            + 0.24 * _semantic_similarity(models.embedder, text, constructive_anchors),
            0.0,
            1.0,
        )
        if constructiveness_detector is not None:
            detector_result = constructiveness_detector.detect(text=text, embedder=models.embedder)
            detector_signal = detector_result.constructiveness_score
            reasoning_marker_score = detector_result.reasoning_marker_score
            suggestion_score = detector_result.suggestion_score
            explanation_depth_score = detector_result.explanation_depth_score
            strategic_reasoning_score = detector_result.strategic_cricket_reasoning_score
            discourse_score = detector_result.discourse_score
            debate_style_score = detector_result.debate_style_score
        else:
            detector_signal = existing_constructive_signal
            reasoning_marker_score = existing_constructive_signal
            suggestion_score = existing_constructive_signal
            explanation_depth_score = existing_constructive_signal
            strategic_reasoning_score = existing_constructive_signal
            discourse_score = existing_constructive_signal
            debate_style_score = 0.0

        constructive_signal = _clip(
            0.5 * detector_signal
            + 0.25 * reasoning_marker_score
            + 0.15 * explanation_depth_score
            + 0.10 * discourse_score,
            0.0,
            1.0,
        )
        if explanation_depth_score >= 0.4:
            constructive_signal = max(constructive_signal, 0.55)
        analytical_signal = _clip(
            0.48 * _clip(_keyword_ratio(text, reasoning_terms + ["strategy", "tactic", "pattern", "selection", "role", "balance", "conditions"]) * 14.0, 0.0, 1.0)
            + 0.24 * _clip(_phrase_ratio(text, ["because", "therefore", "as a result", "which means", "instead of", "rather than"]) * 2.2, 0.0, 1.0)
            + 0.28 * strategic_reasoning_score,
            0.0,
            1.0,
        )
        debate_balance_signal = _clip(
            0.6 * _phrase_ratio(text, debate_terms) * 2.0 + 0.4 * _semantic_similarity(models.embedder, text, debate_anchors),
            0.0,
            1.0,
        )
        respect_signal = _clip(
            0.78 * (1.0 - effective_toxicity)
            + 0.12 * _clip(_keyword_ratio(text, civility_terms) * 20.0, 0.0, 1.0)
            + 0.1 * (1.0 - _abuse_indicator(text)),
            0.0,
            1.0,
        )
        clarity_signal = _clarity_signal(text)
        fan_sincerity_signal = _clip(
            0.52 * _clip(_keyword_ratio(text, fan_terms) * 10.0, 0.0, 1.0)
            + 0.23 * _clip((emo_scores["joy"] + emo_scores["love"] + 0.7 * sent_scores["positive"]), 0.0, 1.0)
            + 0.15 * constructive_signal
            + 0.1 * _semantic_similarity(models.embedder, text, sincerity_anchors),
            0.0,
            1.0,
        )

        # Misclassification resilience: clear reasoning + explanation depth + non-toxic tone should not under-score.
        if (
            reasoning_marker_score >= 0.14
            and explanation_depth_score >= 0.2
            and respect_signal >= 0.65
            and effective_toxicity <= 0.12
        ):
            constructive_signal = max(constructive_signal, 0.65)

        # Constructive fan protection: actionable suggestions with respectful tone should not be treated as weak constructiveness.
        if suggestion_score >= 0.5 and respect_signal >= 0.72 and effective_toxicity <= 0.12:
            constructive_signal = max(constructive_signal, 0.58)

        # Storyteller protection: narrative continuity + explanation arc + low toxicity + moderate sincerity.
        narrative_continuity = discourse_score >= 0.45 and len(paragraph.sentences) >= 4
        explanation_arc = explanation_depth_score >= 0.3 and discourse_score >= 0.35
        moderate_sincerity = fan_sincerity_signal >= 0.3
        if narrative_continuity and explanation_arc and effective_toxicity <= 0.12 and moderate_sincerity:
            constructive_signal = max(constructive_signal, 0.6)

        # Balanced debate recognizer should modestly improve constructive confidence.
        if debate_style_score >= 0.4 and respect_signal >= 0.68 and effective_toxicity <= 0.15:
            constructive_signal = _clip(constructive_signal + 0.05 * debate_style_score, 0.0, 1.0)

        constructive_pattern = (
            ((reasoning_marker_score >= 0.12) or (analytical_signal >= 0.45) or (explanation_depth_score >= 0.5))
            and ((suggestion_score >= 0.15) or (detector_signal >= 0.48))
            and (respect_signal >= 0.68)
            and (effective_toxicity <= 0.15)
        )

        # Constructive emotional writing should not be punished for being emotional.
        emotional_constructive_bonus = _clip(negativity * constructive_signal * 0.28, 0.0, 0.2)
        coherence_support = _clip(0.55 * respect_signal + 0.45 * clarity_signal, 0.0, 1.0)
        constructive_base = _clip(
            0.44 * constructive_signal
            + 0.24 * coherence_support
            + 0.2 * analytical_signal
            + 0.12 * emotional_constructive_bonus,
            0.0,
            1.0,
        )

        constructiveness = COMPONENT_MAX["constructiveness"] * constructive_base
        respectfulness = COMPONENT_MAX["respectfulness"] * respect_signal
        analytical = COMPONENT_MAX["analytical_tone"] * _clip(0.85 * analytical_signal + 0.15 * debate_balance_signal, 0.0, 1.0)
        clarity = COMPONENT_MAX["clarity"] * clarity_signal
        fan_sincerity = COMPONENT_MAX["fan_sincerity"] * fan_sincerity_signal

        para_component_scores["constructiveness"].append(constructiveness)
        para_component_scores["respectfulness"].append(respectfulness)
        para_component_scores["analytical_tone"].append(analytical)
        para_component_scores["clarity"].append(clarity)
        para_component_scores["fan_sincerity"].append(fan_sincerity)
        para_toxicity.append(effective_toxicity)
        para_abuse.append(abuse_signal)
        para_negativity.append(negativity)
        para_sent_positive.append(float(sent_scores.get("positive", 0.0)))
        para_sent_neutral.append(float(sent_scores.get("neutral", 0.0)))
        para_sent_negative.append(float(sent_scores.get("negative", 0.0)))
        constructive_pattern_hits.append(1.0 if constructive_pattern else 0.0)
        reasoning_scores.append(reasoning_marker_score)
        suggestion_scores.append(suggestion_score)
        explanation_depth_scores.append(explanation_depth_score)
        strategic_reasoning_scores.append(strategic_reasoning_score)
        discourse_scores.append(discourse_score)
        debate_style_scores.append(debate_style_score)
        respect_scores.append(respect_signal)
        paragraph_word_counts.append(float(len(re.findall(r"[a-zA-Z']+", text))))
        paragraph_sentence_counts.append(float(len(paragraph.sentences)))

        paragraph_diagnostics.append(
            {
                "paragraph_index": paragraph.index,
                "sentence_count": len(paragraph.sentences),
                "text": text,
                "model_toxicity": round(model_toxic, 4),
                "abuse_signal": round(abuse_signal, 4),
                "effective_toxicity": round(effective_toxicity, 4),
                "negativity": round(negativity, 4),
                "signals": {
                    "constructive": round(constructive_signal, 4),
                    "constructive_existing": round(existing_constructive_signal, 4),
                    "constructive_detector": round(detector_signal, 4),
                    "reasoning_marker_score": round(reasoning_marker_score, 4),
                    "suggestion_score": round(suggestion_score, 4),
                    "explanation_depth_score": round(explanation_depth_score, 4),
                    "strategic_cricket_reasoning_score": round(strategic_reasoning_score, 4),
                    "discourse_score": round(discourse_score, 4),
                    "debate_style_score": round(debate_style_score, 4),
                    "constructive_pattern": constructive_pattern,
                    "respect": round(respect_signal, 4),
                    "analytical": round(analytical_signal, 4),
                    "debate_balance": round(debate_balance_signal, 4),
                    "clarity": round(clarity_signal, 4),
                    "fan_sincerity": round(fan_sincerity_signal, 4),
                },
                "paragraph_scores": {
                    "constructiveness": round(constructiveness, 2),
                    "respectfulness": round(respectfulness, 2),
                    "analytical_tone": round(analytical, 2),
                    "clarity": round(clarity, 2),
                    "fan_sincerity": round(fan_sincerity, 2),
                },
            }
        )

    mean_components = {k: float(np.mean(v)) if v else 0.0 for k, v in para_component_scores.items()}
    tox_mean = float(np.mean(para_toxicity))
    tox_max = float(np.max(para_toxicity))
    tox_ratio = float(np.mean([1.0 if value > 0.55 else 0.0 for value in para_toxicity]))
    writing_quality = compute_writing_quality_signals([paragraph.text for paragraph in paragraphs], embedder=models.embedder)

    return {
        "paragraph_diagnostics": paragraph_diagnostics,
        "mean_components": mean_components,
        "toxicity": {
            "mean": tox_mean,
            "max": tox_max,
            "ratio": tox_ratio,
            "abuse_mean": float(np.mean(para_abuse)),
        },
        "signals": {
            "mean_negativity": float(np.mean(para_negativity)),
            "mean_sentiment_positive": float(np.mean(para_sent_positive)),
            "mean_sentiment_neutral": float(np.mean(para_sent_neutral)),
            "mean_sentiment_negative": float(np.mean(para_sent_negative)),
            "constructive_pattern_ratio": float(np.mean(constructive_pattern_hits)),
            "mean_reasoning_marker_score": float(np.mean(reasoning_scores)),
            "mean_suggestion_score": float(np.mean(suggestion_scores)),
            "mean_explanation_depth_score": float(np.mean(explanation_depth_scores)),
            "mean_strategic_cricket_reasoning_score": float(np.mean(strategic_reasoning_scores)),
            "mean_discourse_score": float(np.mean(discourse_scores)),
            "mean_debate_style_score": float(np.mean(debate_style_scores)),
            "mean_respect_score": float(np.mean(respect_scores)),
            "mean_paragraph_word_count": float(np.mean(paragraph_word_counts)),
            "mean_paragraph_sentence_count": float(np.mean(paragraph_sentence_counts)),
        },
        "writing_quality": writing_quality,
    }


def apply_adaptive_scoring(
    paragraph_analysis: Dict[str, Any],
    writer_type: str,
    writer_type_probabilities: Dict[str, float],
) -> Dict[str, Any]:
    profile = WRITER_PROFILE.get(writer_type, WRITER_PROFILE["All-Rounder"])
    mean_components = paragraph_analysis["mean_components"]
    writer_conf = float(writer_type_probabilities.get(writer_type, 0.0)) if isinstance(writer_type_probabilities, dict) else 0.0

    component_scores = {}
    baseline_constructiveness = float(mean_components.get("constructiveness", 0.0))
    constructiveness_conf_multiplier = 1.0 + 0.08 * writer_conf
    constructiveness_score = _clip(baseline_constructiveness * constructiveness_conf_multiplier, baseline_constructiveness, 40.0)
    component_scores["constructiveness"] = round(constructiveness_score, 2)

    for component in ["respectfulness", "analytical_tone", "clarity", "fan_sincerity"]:
        maximum = COMPONENT_MAX[component]
        score = mean_components.get(component, 0.0) * profile[component]
        component_scores[component] = round(_clip(score, 0.0, maximum), 2)

    return {
        "writer_type": writer_type,
        "writer_type_probabilities": writer_type_probabilities,
        "adaptive_profile": profile,
        "component_scores": component_scores,
        "paragraph_diagnostics": paragraph_analysis["paragraph_diagnostics"],
        "toxicity": paragraph_analysis["toxicity"],
        "signals": paragraph_analysis["signals"],
        "writing_quality": paragraph_analysis.get("writing_quality", {}),
        "stance": paragraph_analysis.get("stance", {}),
        "stance_blending": paragraph_analysis.get("stance_blending", {}),
    }


def apply_stance_aware_weighting(paragraph_analysis: Dict[str, Any], stance_result: Dict[str, Any]) -> Dict[str, Any]:
    weighted = dict(paragraph_analysis)
    mean_components = dict(paragraph_analysis.get("mean_components", {}))
    signals = paragraph_analysis.get("signals", {})

    stance_label = str(stance_result.get("stance_label", "NEUTRAL_ANALYSIS"))
    mean_respect = float(signals.get("mean_respect_score", 0.0))

    if stance_label == "SUPPORTIVE_DEFENSE" and mean_respect > 0.7:
        mean_components["fan_sincerity"] = _clip(float(mean_components.get("fan_sincerity", 0.0)) + 0.35, 0.0, COMPONENT_MAX["fan_sincerity"])
        mean_components["constructiveness"] = _clip(float(mean_components.get("constructiveness", 0.0)) + 0.25, 0.0, COMPONENT_MAX["constructiveness"])

    if stance_label == "CONSTRUCTIVE_CRITICISM":
        mean_components["constructiveness"] = _clip(float(mean_components.get("constructiveness", 0.0)) + 0.30, 0.0, COMPONENT_MAX["constructiveness"])
        mean_components["analytical_tone"] = _clip(float(mean_components.get("analytical_tone", 0.0)) + 0.20, 0.0, COMPONENT_MAX["analytical_tone"])

    if stance_label == "BALANCED_DEBATE":
        mean_components["analytical_tone"] = _clip(float(mean_components.get("analytical_tone", 0.0)) + 1.5, 0.0, COMPONENT_MAX["analytical_tone"])
        mean_components["constructiveness"] = _clip(float(mean_components.get("constructiveness", 0.0)) + 2.0, 0.0, COMPONENT_MAX["constructiveness"])

    if stance_label == "DISMISSIVE_COMPLAINT":
        mean_components["constructiveness"] = _clip(float(mean_components.get("constructiveness", 0.0)) * 0.82, 0.0, COMPONENT_MAX["constructiveness"])

    if stance_label == "DIRECT_ATTACK" and not bool(stance_result.get("attack_rejection_detected", False)):
        mean_components["constructiveness"] = _clip(float(mean_components.get("constructiveness", 0.0)) * 0.7, 0.0, COMPONENT_MAX["constructiveness"])
        mean_components["respectfulness"] = _clip(float(mean_components.get("respectfulness", 0.0)) * 0.65, 0.0, COMPONENT_MAX["respectfulness"])

    weighted["mean_components"] = mean_components
    weighted["stance"] = stance_result
    weighted["stance_blending"] = _compute_stance_blending(stance_result)
    return weighted


def apply_toxicity_penalty(adaptive_result: Dict[str, Any], writer_type: str) -> Dict[str, Any]:
    toxicity = adaptive_result["toxicity"]
    profile = WRITER_PROFILE.get(writer_type, WRITER_PROFILE["All-Rounder"])
    diagnostics = adaptive_result.get("paragraph_diagnostics", [])
    stance = adaptive_result.get("stance", {})
    stance_blending = adaptive_result.get("stance_blending", {}) if isinstance(adaptive_result.get("stance_blending", {}), dict) else {}
    writing_quality = adaptive_result.get("writing_quality", {}) if isinstance(adaptive_result.get("writing_quality", {}), dict) else {}
    writing_quality_aggregate = writing_quality.get("aggregate", {}) if isinstance(writing_quality.get("aggregate", {}), dict) else {}
    attack_rejection_detected = bool(stance.get("attack_rejection_detected", False))
    attack_endorsement_detected = bool(stance.get("attack_endorsement_detected", False))
    stance_label = str(stance.get("stance_label", "NEUTRAL_ANALYSIS"))
    primary_stance_label = str(stance.get("primary_stance_label", ""))
    supportive_defense_strength = float(stance.get("supportive_defense_strength", 0.0))
    stance_confidence = float(stance.get("stance_confidence", 0.0))
    causal_defense_score = float(stance.get("causal_defense_score", 0.0))
    fairness_defense_score = float(stance.get("fairness_defense_score", 0.0))
    scalar_metrics = stance.get("scalar_metrics", {}) if isinstance(stance.get("scalar_metrics", {}), dict) else {}
    sarcasm_score = float(scalar_metrics.get("sarcasm_score", 0.0))
    exaggeration_score = float(scalar_metrics.get("exaggeration_score", 0.0))
    contradiction_score = float(scalar_metrics.get("contradiction_score", 0.0))
    context_mismatch_score = float(scalar_metrics.get("context_mismatch_score", 0.0))
    ridicule_score = float(scalar_metrics.get("ridicule_score", 0.0))
    neutrality_score = float(scalar_metrics.get("neutrality_score", 0.0))
    direct_attack_prob = float(stance.get("stance_probabilities", {}).get("DIRECT_ATTACK", 0.0))
    dismissive_complaint_prob = float(stance.get("stance_probabilities", {}).get("DISMISSIVE_COMPLAINT", 0.0))
    mean_respect = float(adaptive_result.get("signals", {}).get("mean_respect_score", 0.0))

    detector_mean = 0.0
    if diagnostics:
        detector_mean = float(
            np.mean([
                float(item.get("signals", {}).get("constructive_detector", 0.0))
                for item in diagnostics
            ])
        )

    # Max-toxicity override means one toxic paragraph can strongly impact final score.
    toxicity_intensity = _clip(
        0.2 * toxicity["mean"] + 0.2 * toxicity["max"] + 0.15 * toxicity["ratio"] + 0.45 * toxicity["abuse_mean"],
        0.0,
        1.0,
    )
    fairness_override_allowed = (
        stance_label == "SUPPORTIVE_DEFENSE"
        and supportive_defense_strength >= 0.4
        and mean_respect >= 0.7
        and toxicity_intensity < 0.25
        and direct_attack_prob <= 0.25
        and dismissive_complaint_prob <= 0.35
    )
    intent_supportive_override_allowed = (
        stance_label == "SUPPORTIVE_DEFENSE"
        and causal_defense_score >= 0.35
        and not attack_endorsement_detected
    )
    if attack_rejection_detected:
        toxicity_intensity = _clip(toxicity_intensity * 0.2, 0.0, 1.0)
    if fairness_override_allowed:
        toxicity_intensity = _clip(toxicity_intensity * 0.2, 0.0, 1.0)
    if intent_supportive_override_allowed:
        toxicity_intensity = _clip(toxicity_intensity * 0.2, 0.0, 1.0)
    abuse_boost = _clip(0.65 + 0.7 * toxicity["abuse_mean"], 0.65, 1.35)
    penalty_strength = profile["toxicity_strength"] * abuse_boost

    toxicity_penalty = -25.0 * _clip((toxicity_intensity**1.22) * penalty_strength, 0.0, 1.0)
    if attack_rejection_detected:
        toxicity_penalty = max(toxicity_penalty, -5.0)
    if fairness_override_allowed:
        toxicity_penalty = max(toxicity_penalty, -5.0)
    if intent_supportive_override_allowed:
        toxicity_penalty = max(toxicity_penalty, -1.0)
    toxicity_penalty = round(toxicity_penalty, 2)

    component_scores = dict(adaptive_result["component_scores"])
    dominant_stance_component = float(stance_blending.get("dominant_stance_component", STANCE_MIDPOINTS["NEUTRAL_ANALYSIS"]))
    final_stance_component = float(stance_blending.get("final_stance_component", dominant_stance_component))
    hard_override = bool(stance_blending.get("hard_override", False))
    toxicity_penalty_active = toxicity_penalty < 0.0
    stance_blend_adjustment = 0.0

    if not hard_override and not toxicity_penalty_active:
        stance_blend_adjustment = 0.85 * (final_stance_component - dominant_stance_component)

    print(
        "STANCE BLEND DEBUG:",
        stance_blending.get("stance_probabilities", {}),
        round(float(stance_blending.get("top_probability", 0.0)), 4),
        round(float(stance_blending.get("second_probability", 0.0)), 4),
        round(float(stance_blending.get("probability_gap", 0.0)), 4),
        stance_blending.get("dominant_stance", "NEUTRAL_ANALYSIS"),
        round(float(stance_blending.get("soft_stance_component", 0.0)), 4),
        round(final_stance_component, 4),
    )

    component_scores["toxicity_penalty"] = toxicity_penalty
    component_scores["stance_blend_adjustment"] = round(stance_blend_adjustment, 2)

    def _recompute_total() -> float:
        return (
            component_scores["constructiveness"]
            + component_scores["respectfulness"]
            + component_scores["analytical_tone"]
            + component_scores["clarity"]
            + component_scores["fan_sincerity"]
            + toxicity_penalty
        )

    total = _recompute_total()

    # Archetype floor calibration for coherent, respectful, low-toxicity writing.
    if writer_type == "Analyst":
        quality_score = (
            0.35 * (component_scores["constructiveness"] / 40.0)
            + 0.15 * (component_scores["respectfulness"] / 20.0)
            + 0.35 * (component_scores["analytical_tone"] / 15.0)
            + 0.15 * (component_scores["clarity"] / 10.0)
        )
    elif writer_type == "Storyteller":
        quality_score = (
            0.28 * (component_scores["constructiveness"] / 40.0)
            + 0.22 * (component_scores["respectfulness"] / 20.0)
            + 0.2 * (component_scores["clarity"] / 10.0)
            + 0.3 * (component_scores["fan_sincerity"] / 15.0)
        )
    elif writer_type == "Debater":
        quality_score = (
            0.34 * (component_scores["constructiveness"] / 40.0)
            + 0.28 * (component_scores["respectfulness"] / 20.0)
            + 0.26 * (component_scores["analytical_tone"] / 15.0)
            + 0.12 * (component_scores["clarity"] / 10.0)
        )
    else:
        quality_score = (
            0.32 * (component_scores["constructiveness"] / 40.0)
            + 0.22 * (component_scores["respectfulness"] / 20.0)
            + 0.18 * (component_scores["clarity"] / 10.0)
            + 0.18 * (component_scores["fan_sincerity"] / 15.0)
            + 0.1 * (component_scores["analytical_tone"] / 15.0)
        )

    anchor_floor = {
        "Passionate Fan": 75.0,
        "Storyteller": 75.0,
        "Analyst": 80.0,
        "Debater": 80.0,
        "All-Rounder": 60.0,
    }.get(writer_type, 60.0)

    supportive_fairness_gate = (
        stance_label == "SUPPORTIVE_DEFENSE"
        and attack_rejection_detected
        and supportive_defense_strength >= 0.45
        and fairness_defense_score >= 0.5
        and mean_respect >= 0.75
        and toxicity["abuse_mean"] < 0.2
        and toxicity["mean"] < 0.2
        and direct_attack_prob <= 0.2
        and dismissive_complaint_prob <= 0.3
    )
    if supportive_fairness_gate:
        anchor_floor = max(anchor_floor, 72.0)

    constructive_pattern_ratio = float(adaptive_result.get("signals", {}).get("constructive_pattern_ratio", 0.0))
    mean_reasoning = float(adaptive_result.get("signals", {}).get("mean_reasoning_marker_score", 0.0))
    mean_suggestion = float(adaptive_result.get("signals", {}).get("mean_suggestion_score", 0.0))
    mean_explanation_depth = float(adaptive_result.get("signals", {}).get("mean_explanation_depth_score", 0.0))
    mean_strategic_reasoning = float(adaptive_result.get("signals", {}).get("mean_strategic_cricket_reasoning_score", 0.0))
    mean_discourse = float(adaptive_result.get("signals", {}).get("mean_discourse_score", 0.0))
    mean_respect = float(adaptive_result.get("signals", {}).get("mean_respect_score", 0.0))
    mean_paragraph_word_count = float(adaptive_result.get("signals", {}).get("mean_paragraph_word_count", 0.0))
    mean_paragraph_sentence_count = float(adaptive_result.get("signals", {}).get("mean_paragraph_sentence_count", 0.0))
    floor_trigger = (
        (constructive_pattern_ratio >= 0.35)
        or (writer_type == "Analyst" and component_scores["analytical_tone"] >= 7.0)
        or (writer_type == "Debater" and component_scores["analytical_tone"] >= 6.5 and component_scores["respectfulness"] >= 13.0)
        or (writer_type == "Storyteller" and component_scores["clarity"] >= 7.0 and component_scores["fan_sincerity"] >= 7.0)
        or (writer_type == "Passionate Fan" and component_scores["fan_sincerity"] >= 8.0 and component_scores["respectfulness"] >= 14.0)
        or supportive_fairness_gate
    )

    extended_floor_trigger = floor_trigger or (
        writer_type in {"Analyst", "Debater"}
        and toxicity["abuse_mean"] < 0.08
        and constructive_pattern_ratio >= 0.2
        and quality_score >= 0.34
    ) or (
        writer_type == "Storyteller"
        and toxicity["abuse_mean"] < 0.08
        and constructive_pattern_ratio >= 0.2
        and quality_score >= 0.36
        and component_scores["clarity"] >= 6.0
    )

    if (toxicity["abuse_mean"] < 0.12 or supportive_fairness_gate) and extended_floor_trigger and total < anchor_floor:
        uplift = anchor_floor - total
        uplift_weights = {
            "Passionate Fan": {"constructiveness": 0.52, "fan_sincerity": 0.33, "clarity": 0.15},
            "Storyteller": {"clarity": 0.45, "fan_sincerity": 0.35, "constructiveness": 0.2},
            "Analyst": {"analytical_tone": 0.42, "constructiveness": 0.33, "clarity": 0.25},
            "Debater": {"analytical_tone": 0.35, "constructiveness": 0.35, "respectfulness": 0.2, "clarity": 0.1},
            "All-Rounder": {"constructiveness": 0.3, "respectfulness": 0.2, "analytical_tone": 0.2, "clarity": 0.15, "fan_sincerity": 0.15},
        }.get(writer_type, {"constructiveness": 0.3, "respectfulness": 0.2, "analytical_tone": 0.2, "clarity": 0.15, "fan_sincerity": 0.15})

        for component, weight in uplift_weights.items():
            component_scores[component] = round(
                _clip(component_scores[component] + uplift * weight, 0.0, COMPONENT_MAX[component]),
                2,
            )

        # If weighted uplift hits caps, spread any remainder across available component headroom.
        subtotal = _recompute_total()
        remainder = max(0.0, anchor_floor - subtotal)
        if remainder > 0.0:
            for component in ["constructiveness", "respectfulness", "analytical_tone", "clarity", "fan_sincerity"]:
                if remainder <= 0.0:
                    break
                room = COMPONENT_MAX[component] - component_scores[component]
                if room <= 0.0:
                    continue
                add = min(room, remainder)
                component_scores[component] = round(component_scores[component] + add, 2)
                remainder -= add

        total = _recompute_total()

    rescue_result = apply_deterministic_rescue(
        writer_type=writer_type,
        component_scores=component_scores,
        total=total,
        toxicity=toxicity,
        signals={
            "constructive_pattern_ratio": constructive_pattern_ratio,
            "mean_reasoning_marker_score": mean_reasoning,
            "mean_suggestion_score": mean_suggestion,
            "mean_explanation_depth_score": mean_explanation_depth,
            "mean_strategic_cricket_reasoning_score": mean_strategic_reasoning,
            "mean_discourse_score": mean_discourse,
            "mean_respect_score": mean_respect,
            "mean_paragraph_word_count": mean_paragraph_word_count,
            "mean_paragraph_sentence_count": mean_paragraph_sentence_count,
            "detector_mean": detector_mean,
            "stance_label": stance_label,
            "supportive_defense_strength": supportive_defense_strength,
        },
        paragraph_diagnostics=diagnostics,
    )
    component_scores = rescue_result["component_scores"]
    total = rescue_result["total"]

    if writer_type == "Analyst" and mean_explanation_depth >= 0.4 and mean_respect >= 0.7:
        component_scores["constructiveness"] = round(max(component_scores["constructiveness"], 24.0), 2)
        total = _recompute_total()

    complaint_phrases = [
        "same script",
        "nothing ever changes",
        "more of the same",
        "another frustrating",
        "every week",
        "supporters deserve stronger",
        "recurring pattern",
        "same mistake again",
        "nothing changes",
        "always happens",
        "management never learns",
        "waste of opportunity",
        "repeated failure without explanation",
    ]
    reasoning_terms = ["because", "therefore", "however", "while", "although", "whereas", "if", "then", "evidence", "approach"]
    suggestion_terms = ["should", "could", "need to", "needs to", "improve", "fix", "adjust", "recommend"]
    text_blob = " ".join(str(item.get("text", "")).lower() for item in diagnostics)
    repeated_phrase_hits = sum(1 for phrase in complaint_phrases if text_blob.count(phrase) >= 1)
    complaint_style_repetition = repeated_phrase_hits >= 1
    lexical_reasoning_hits = sum(1 for term in reasoning_terms if term in text_blob)
    lexical_suggestion_hits = sum(1 for term in suggestion_terms if term in text_blob)
    low_reasoning = mean_reasoning < 0.1 and lexical_reasoning_hits <= 1
    low_suggestion = mean_suggestion < 0.1
    low_depth = mean_explanation_depth < 0.18 or lexical_suggestion_hits == 0
    low_discourse = mean_discourse < 0.2
    low_constructiveness = component_scores["constructiveness"] < 18.0
    low_empathy_sincerity = component_scores["fan_sincerity"] < 6.0
    weak_analytical_tone = component_scores["analytical_tone"] < 4.5

    # Complaint caps apply only when complaint-like repetition and weak reasoning/suggestion/depth are all present.
    if (
        toxicity["abuse_mean"] < 0.12
        and low_reasoning
        and low_suggestion
        and low_depth
        and low_discourse
        and low_constructiveness
        and low_empathy_sincerity
        and complaint_style_repetition
        and total > 58.0
    ):
        total = 58.0

    if (
        toxicity["abuse_mean"] < 0.12
        and complaint_style_repetition
        and low_suggestion
        and weak_analytical_tone
        and low_empathy_sincerity
        and lexical_reasoning_hits <= 2
        and total > 60.0
    ):
        total = 60.0

    # Complaint false-boost suppressor: apply a multiplier only when complaint repetition exists without reasoning/suggestion/depth/discourse.
    if (
        complaint_style_repetition
        and mean_reasoning < 0.1
        and mean_suggestion < 0.1
        and mean_explanation_depth < 0.2
        and mean_discourse < 0.2
        and toxicity["abuse_mean"] < 0.12
    ):
        component_scores["constructiveness"] = round(component_scores["constructiveness"] * 0.78, 2)
        total = _recompute_total()

    # Strong abuse should suppress optimistic totals even if other components remain high.
    if toxicity["abuse_mean"] >= 0.2 and not attack_rejection_detected:
        if toxicity_penalty > -20.0:
            toxicity_penalty = -20.0
        component_scores["toxicity_penalty"] = toxicity_penalty
        total = _recompute_total()

    if toxicity["abuse_mean"] >= 0.35 and not attack_rejection_detected:
        if toxicity_penalty > -25.0:
            toxicity_penalty = -25.0
        component_scores["toxicity_penalty"] = toxicity_penalty
        total = _recompute_total()

    # Toxic attack cap: clearly abusive competence attacks must stay in low ranges.
    if toxicity["abuse_mean"] >= 0.2 and total > 35.0 and not attack_rejection_detected:
        total = 35.0

    if stance_blend_adjustment != 0.0:
        total = _clip(total + stance_blend_adjustment, SCORE_MIN, SCORE_MAX)

    sarcasm_attack_like = (
        primary_stance_label == "SARCASTIC_ATTACK"
        or stance_label in {"SARCASTIC_CRITICISM", "DIRECT_ATTACK", "DISMISSIVE_COMPLAINT"}
    )
    sarcasm_credibility_penalty = 0.0
    if sarcasm_attack_like:
        base_penalty = (
            12.0 * sarcasm_score
            + 9.0 * ridicule_score
            + 7.0 * contradiction_score
            + 7.0 * context_mismatch_score
            + 4.0 * exaggeration_score
            - 4.0 * neutrality_score
        )
        if sarcasm_score >= 0.5 and ridicule_score >= 0.5:
            base_penalty += 8.0
        if str(stance.get("sarcasm_gate_reason", "")).startswith("sarcasm_threshold_met") and primary_stance_label == "SARCASTIC_ATTACK":
            base_penalty += 6.0
        confidence_scale = 0.8 + 0.4 * _clip(stance_confidence, 0.0, 1.0)
        sarcasm_credibility_penalty = _clip(base_penalty * confidence_scale, 0.0, 42.0)

    if stance_label == "SUPPORTIVE_DEFENSE" or primary_stance_label == "SUPPORTIVE_DEFENSE":
        sarcasm_credibility_penalty = 0.0

    if sarcasm_credibility_penalty > 0.0:
        total = max(0.0, total - sarcasm_credibility_penalty)
    component_scores["sarcasm_credibility_penalty"] = round(-sarcasm_credibility_penalty, 2)

    coherence_score = float(writing_quality_aggregate.get("coherence_score", 0.0))
    lexical_diversity_score = float(writing_quality_aggregate.get("lexical_diversity_score", 0.0))
    sentence_variety_score = float(writing_quality_aggregate.get("sentence_variety_score", 0.0))
    repetition_penalty = float(writing_quality_aggregate.get("repetition_penalty", 0.0))
    position_clarity_score = float(writing_quality_aggregate.get("position_clarity_score", 0.0))
    counter_argument_score = float(writing_quality_aggregate.get("counter_argument_score", 0.0))
    evidence_presence_score = float(writing_quality_aggregate.get("evidence_presence_score", 0.0))
    completeness_score = float(writing_quality_aggregate.get("completeness_score", 0.0))
    information_density_score = float(writing_quality_aggregate.get("information_density_score", 0.0))
    argument_logic_score = float(writing_quality_aggregate.get("argument_logic_score", 0.0))

    # Writing-quality contribution is applied in sentiment_pipeline before calibration.
    writing_quality_component = 0.0
    component_scores["writing_quality_component"] = 0.0

    final_score = round(_clip(total, SCORE_MIN, SCORE_MAX), 2)
    if (
        stance_label == "SUPPORTIVE_DEFENSE"
        and supportive_defense_strength >= 0.25
        and supportive_defense_strength < 0.40
    ):
        final_score = round(min(82.0, final_score + (5.0 * supportive_defense_strength)), 2)

    return {
        **adaptive_result,
        "component_scores": component_scores,
        "components": {
            "constructiveness": {
                "score": component_scores["constructiveness"],
                "max": 40.0,
                "explanation": "Rewards actionable improvements and constructive criticism, including emotional but respectful critique.",
            },
            "respectfulness": {
                "score": component_scores["respectfulness"],
                "max": 20.0,
                "explanation": "Measures civility and absence of abuse; disagreement or disappointment alone is not treated as toxicity.",
            },
            "analytical_tone": {
                "score": component_scores["analytical_tone"],
                "max": 15.0,
                "explanation": "Tracks reasoning and structure but does not force statistics for fan or storyteller styles.",
            },
            "clarity": {
                "score": component_scores["clarity"],
                "max": 10.0,
                "explanation": "Captures readability and coherent flow, including narrative style.",
            },
            "fan_sincerity": {
                "score": component_scores["fan_sincerity"],
                "max": 15.0,
                "explanation": "Rewards authentic supporter voice and team-focused intent.",
            },
            "toxicity_penalty": {
                "score": toxicity_penalty,
                "max_negative": -25.0,
                "explanation": "Strong paragraph-weighted penalty with max-toxicity override for abuse and personal attacks.",
            },
            "sarcasm_credibility_penalty": {
                "score": round(-sarcasm_credibility_penalty, 2),
                "max_negative": -42.0,
                "explanation": "Penalizes sarcastic ridicule and contradiction in attack-like stance outputs to avoid inflated quality scores.",
            },
            "writing_quality_component": {
                "score": round(writing_quality_component, 2),
                "min": -10.0,
                "max": 15.0,
                "explanation": "Deterministic rubric-like writing quality adjustment from coherence, evidence, argument logic, density, and repetition.",
            },
            "stance_blend_adjustment": {
                "score": round(stance_blend_adjustment, 2),
                "min": -70.0,
                "max": 70.0,
                "explanation": "Probability-aware stance smoothing adjustment computed from dominant and soft stance anchors.",
            },
        },
        "final_score": final_score,
        "score_out_of_100": final_score,
        "stance_label": stance.get("stance_label", "NEUTRAL_ANALYSIS"),
        "stance_confidence": stance.get("stance_confidence", 0.0),
        "supportive_defense_strength": supportive_defense_strength,
        "fairness_defense_score": float(stance.get("fairness_defense_score", 0.0)),
        "reputation_defense_score": float(stance.get("reputation_defense_score", 0.0)),
        "contrast_rejection_detected": bool(stance.get("contrast_rejection_detected", False)),
        "quoted_attack_detected": bool(stance.get("quoted_attack_detected", False)),
        "attack_endorsement_detected": bool(stance.get("attack_endorsement_detected", False)),
        "attack_rejection_detected": attack_rejection_detected,
        "writing_quality_breakdown": {
            "coherence_score": round(coherence_score, 4),
            "lexical_diversity_score": round(lexical_diversity_score, 4),
            "sentence_variety_score": round(sentence_variety_score, 4),
            "repetition_penalty": round(repetition_penalty, 4),
            "position_clarity_score": round(position_clarity_score, 4),
            "counter_argument_score": round(counter_argument_score, 4),
            "evidence_presence_score": round(evidence_presence_score, 4),
            "completeness_score": round(completeness_score, 4),
            "information_density_score": round(information_density_score, 4),
            "argument_logic_score": round(argument_logic_score, 4),
        },
        "stance_blending_debug": {
            "stance_probabilities": stance_blending.get("stance_probabilities", {}),
            "top_probability": round(float(stance_blending.get("top_probability", 0.0)), 4),
            "second_probability": round(float(stance_blending.get("second_probability", 0.0)), 4),
            "probability_gap": round(float(stance_blending.get("probability_gap", 0.0)), 4),
            "dominant_stance": stance_blending.get("dominant_stance", "NEUTRAL_ANALYSIS"),
            "dominant_stance_component": round(float(stance_blending.get("dominant_stance_component", 0.0)), 4),
            "soft_stance_component": round(float(stance_blending.get("soft_stance_component", 0.0)), 4),
            "final_stance_component": round(float(stance_blending.get("final_stance_component", 0.0)), 4),
        },
        "rescue_layer": {
            "activated": rescue_result["activated"],
            "reason": rescue_result["reason"],
            "rescued_outlier": rescue_result["rescued_outlier"],
            "confidence_adjustment": rescue_result["confidence_adjustment"],
            "analytical_boost_applied": rescue_result["analytical_boost_applied"],
        },
    }
