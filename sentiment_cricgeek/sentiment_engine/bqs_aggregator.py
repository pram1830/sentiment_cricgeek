from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re


SCORE_MIN = 20.0
SCORE_MAX = 95.0

ARCHETYPE_KEYS = {
    "analyst_score": "Analyst",
    "fan_score": "Passionate Fan",
    "storyteller_score": "Storyteller",
    "debater_score": "Debater",
}

ARCHETYPE_WEIGHT_PROFILES = {
    "analyst_score": {
        "stat_accuracy_component": 0.30,
        "stance_component": 0.25,
        "writing_quality_component": 0.20,
        "argument_logic_component": 0.15,
        "information_density_component": 0.10,
    },
    "fan_score": {
        "stance_component": 0.40,
        "writing_quality_component": 0.25,
        "toxicity_component": 0.15,
        "engagement_style_component": 0.10,
        "structure_component": 0.10,
    },
    "storyteller_score": {
        "writing_quality_component": 0.35,
        "coherence_component": 0.20,
        "stance_component": 0.20,
        "information_density_component": 0.15,
        "structure_component": 0.10,
    },
    "debater_score": {
        "argument_logic_component": 0.30,
        "counter_argument_component": 0.20,
        "stance_component": 0.20,
        "writing_quality_component": 0.15,
        "stat_accuracy_component": 0.15,
    },
}

STANCE_BANDS = {
    "DIRECT_ATTACK": (20.0, 45.0),
    "DISMISSIVE_COMPLAINT": (30.0, 55.0),
    "NEUTRAL_ANALYSIS": (40.0, 70.0),
    "BALANCED_DEBATE": (60.0, 80.0),
    "CONSTRUCTIVE_CRITICISM": (65.0, 85.0),
    "SUPPORTIVE_DEFENSE": (70.0, 95.0),
}

STANCE_ANCHORS = {
    "SUPPORTIVE_DEFENSE": 82.0,
    "CONSTRUCTIVE_CRITICISM": 78.0,
    "BALANCED_DEBATE": 76.0,
    "NEUTRAL_ANALYSIS": 64.0,
    "MIXED_STANCE": 55.0,
    "DISMISSIVE_COMPLAINT": 40.0,
    "DIRECT_ATTACK": 25.0,
}

POS_ANCHOR = 78.0
NEUTRAL_ANCHOR = 62.0
NEG_ANCHOR = 40.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_archetype_scores(writer_probs: Dict[str, float]) -> Dict[str, float]:
    raw = {
        archetype_key: float(writer_probs.get(writer_name, 0.0))
        for archetype_key, writer_name in ARCHETYPE_KEYS.items()
    }
    total = sum(max(0.0, value) for value in raw.values())
    if total <= 0.0:
        return {key: 0.25 for key in raw}
    return {key: max(0.0, value) / total for key, value in raw.items()}


def _normalize_distribution(raw: Dict[str, float], labels: List[str]) -> Tuple[Dict[str, float], List[str]]:
    probs = {label: _clip(float(raw.get(label, 0.0)), 0.0, 1.0) for label in labels}
    total = float(sum(probs.values()))
    warnings: List[str] = []
    if total <= 0.0:
        warnings.append("missing_probability_vector")
        return probs, warnings
    if abs(total - 1.0) > 1e-6:
        warnings.append("distribution_renormalized")
    return {label: probs[label] / total for label in labels}, warnings


def _top_two_gap(distribution: Dict[str, float]) -> Tuple[float, float, str]:
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return 0.0, 0.0, ""
    top = float(ranked[0][1])
    second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    return top, second, _clip(top - second, 0.0, 1.0)


def _to_score_range(unit_value: float) -> float:
    return SCORE_MIN + (SCORE_MAX - SCORE_MIN) * _clip(unit_value, 0.0, 1.0)


def _component_to_score(component: float, low: float, high: float) -> float:
    if high <= low:
        return SCORE_MIN
    normalized = _clip((component - low) / (high - low), 0.0, 1.0)
    return _to_score_range(normalized)


def _contains_any(text: str, phrases: List[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _attack_severity(blog_text: str) -> int:
    text = blog_text.lower()
    insult_keywords = [
        "useless",
        "fraud",
        "idiot",
        "moron",
        "pathetic",
        "trash",
        "garbage",
        "disgrace",
        "clueless",
    ]
    humiliation_markers = [
        "disgrace to cricket",
        "embarrassment",
        "shameful",
        "humiliating",
        "joke",
    ]
    group_dismissal_patterns = [
        r"anyone who supports",
        r"people who support",
        r"fans who support",
    ]

    severity = 0
    if _contains_any(text, insult_keywords):
        severity += 1
    if _contains_any(text, humiliation_markers):
        severity += 1
    if any(re.search(pattern, text) is not None for pattern in group_dismissal_patterns):
        severity += 1
    return severity


def _dismissive_ceiling(blog_text: str) -> float:
    text = blog_text.lower()
    suggestion_markers = [
        "should",
        "could",
        "needs to",
        "need to",
        "improve",
        "reconsider",
        "adjust",
        "recommend",
        "better",
    ]
    if _contains_any(text, suggestion_markers):
        return 50.0
    return 40.0


def _is_dismissive_text(blog_text: str) -> bool:
    text = blog_text.lower()
    dismissive_markers = [
        "same script",
        "nothing changes",
        "management never learns",
        "frustrating",
        "poorly handled",
        "repeated mistakes",
        "no clear plan",
    ]
    return _contains_any(text, dismissive_markers)


def _soft_stance_components(stance_result: Dict[str, Any]) -> Tuple[Dict[str, float], float, float, float, float, str, List[str]]:
    raw_probs = stance_result.get("stance_probabilities", {}) if isinstance(stance_result.get("stance_probabilities", {}), dict) else {}
    labels = list(STANCE_ANCHORS.keys())
    stance_probs, warnings = _normalize_distribution(raw_probs, labels)

    top, second, gap = _top_two_gap(stance_probs)
    ranked = sorted(stance_probs.items(), key=lambda item: item[1], reverse=True)
    dominant_label = ranked[0][0] if ranked else "NEUTRAL_ANALYSIS"
    dominant_stance_component = STANCE_ANCHORS.get(dominant_label, STANCE_ANCHORS["NEUTRAL_ANALYSIS"])
    soft_stance_component = sum(stance_probs[label] * STANCE_ANCHORS[label] for label in labels)

    if gap >= 0.15:
        final_stance_component = 0.65 * dominant_stance_component + 0.35 * soft_stance_component
    else:
        final_stance_component = 0.45 * dominant_stance_component + 0.55 * soft_stance_component

    return stance_probs, dominant_stance_component, soft_stance_component, top, second, dominant_label, warnings + [f"stance_gap={gap:.4f}"]


def _soft_sentiment_component(signals: Dict[str, Any]) -> Tuple[Dict[str, float], float, float, float, List[str]]:
    labels = ["positive", "neutral", "negative"]
    raw_sentiment = {
        "positive": float(signals.get("mean_sentiment_positive", 0.0)),
        "neutral": float(signals.get("mean_sentiment_neutral", 0.0)),
        "negative": float(signals.get("mean_sentiment_negative", 0.0)),
    }
    sentiment_probs, warnings = _normalize_distribution(raw_sentiment, labels)

    if sum(sentiment_probs.values()) <= 0.0:
        mean_negativity = _clip(float(signals.get("mean_negativity", 0.5)), 0.0, 1.0)
        sentiment_probs = {
            "positive": _clip(1.0 - mean_negativity, 0.0, 1.0),
            "neutral": 0.0,
            "negative": mean_negativity,
        }
        sentiment_probs, _ = _normalize_distribution(sentiment_probs, labels)
        warnings.append("sentiment_probability_fallback_from_mean_negativity")

    top, second, gap = _top_two_gap(sentiment_probs)
    sentiment_soft_component = (
        sentiment_probs["positive"] * POS_ANCHOR
        + sentiment_probs["neutral"] * NEUTRAL_ANCHOR
        + sentiment_probs["negative"] * NEG_ANCHOR
    )
    return sentiment_probs, sentiment_soft_component, top, second, warnings + [f"sentiment_gap={gap:.4f}"]


def _blended_archetype_weights(archetype_scores: Dict[str, float]) -> Dict[str, float]:
    keys = {
        "stance_component",
        "stat_accuracy_component",
        "writing_quality_component",
        "argument_logic_component",
        "information_density_component",
        "toxicity_component",
        "engagement_style_component",
        "structure_component",
        "coherence_component",
        "counter_argument_component",
    }
    blended = {key: 0.0 for key in keys}
    for archetype_key, confidence in archetype_scores.items():
        profile = ARCHETYPE_WEIGHT_PROFILES.get(archetype_key, {})
        for component_key, weight in profile.items():
            blended[component_key] += confidence * float(weight)
    return blended


def _stance_strength(stance_label: str, stance_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    stance_probs = stance_result.get("stance_probabilities", {}) if isinstance(stance_result.get("stance_probabilities", {}), dict) else {}
    supportive_strength = float(stance_result.get("supportive_defense_strength", 0.0))
    constructive_strength = float(stance_probs.get("CONSTRUCTIVE_CRITICISM", 0.0)) + 0.7 * float(stance_probs.get("BALANCED_DEBATE", 0.0))
    attack_strength = float(stance_probs.get("DIRECT_ATTACK", 0.0))
    dismissive_strength = float(stance_probs.get("DISMISSIVE_COMPLAINT", 0.0))

    base_map = {
        "SUPPORTIVE_DEFENSE": 0.84,
        "CONSTRUCTIVE_CRITICISM": 0.76,
        "BALANCED_DEBATE": 0.70,
        "NEUTRAL_ANALYSIS": 0.58,
        "MIXED_STANCE": 0.52,
        "DISMISSIVE_COMPLAINT": 0.40,
        "DIRECT_ATTACK": 0.20,
        "SARCASTIC_CRITICISM": 0.32,
    }
    base = float(base_map.get(stance_label, 0.55))

    stance_strength = _clip(
        base
        + 0.14 * supportive_strength
        + 0.08 * constructive_strength
        - 0.20 * attack_strength
        - 0.12 * dismissive_strength,
        0.0,
        1.0,
    )

    return stance_strength, {
        "supportive_defense_strength": round(supportive_strength, 4),
        "constructive_strength": round(constructive_strength, 4),
        "attack_strength": round(attack_strength, 4),
        "dismissive_strength": round(dismissive_strength, 4),
    }


def _stats_shape(stats_verification: Dict[str, Any]) -> float:
    if not bool(stats_verification.get("stats_found", False)):
        return 0.0
    if bool(stats_verification.get("stats_verified", False)):
        return 4.0
    accuracy = float(stats_verification.get("stat_accuracy_score", 0.0))
    if accuracy >= 0.5:
        return 2.0
    return -6.0


def aggregate_bqs(
    *,
    stance_result: Dict[str, Any],
    stats_verification: Dict[str, Any],
    writing_quality_breakdown: Dict[str, float],
    component_scores: Dict[str, float],
    writer_type_probabilities: Dict[str, float],
    signals: Dict[str, Any] | None = None,
    blog_text: str = "",
) -> Dict[str, Any]:
    signal_map = signals if isinstance(signals, dict) else {}
    archetype_scores = _normalize_archetype_scores(writer_type_probabilities)
    archetype_weights = _blended_archetype_weights(archetype_scores)

    stance_label = str(stance_result.get("stance_label", "NEUTRAL_ANALYSIS"))
    stance_strength_value, stance_strength_parts = _stance_strength(stance_label, stance_result)
    stance_probs, dominant_stance_component, soft_stance_component, stance_top, stance_second, dominant_stance_label, stance_warnings = _soft_stance_components(stance_result)
    stance_gap = _clip(stance_top - stance_second, 0.0, 1.0)

    if stance_gap >= 0.15:
        final_stance_component = 0.65 * dominant_stance_component + 0.35 * soft_stance_component
    else:
        final_stance_component = 0.45 * dominant_stance_component + 0.55 * soft_stance_component

    # Preserve probability-based stance while adding deterministic stance-strength shaping.
    stance_strength_score = _to_score_range(stance_strength_value)
    final_stance_component = 0.8 * final_stance_component + 0.2 * stance_strength_score
    final_stance_component += (
        20.0 * float(stance_strength_parts.get("supportive_defense_strength", 0.0))
        + 10.0 * float(stance_strength_parts.get("constructive_strength", 0.0))
        - 16.0 * float(stance_strength_parts.get("attack_strength", 0.0))
        - 12.0 * float(stance_strength_parts.get("dismissive_strength", 0.0))
    )
    final_stance_component = _clip(final_stance_component, SCORE_MIN, SCORE_MAX)

    stat_accuracy = _clip(float(stats_verification.get("stat_accuracy_score", 0.0)), 0.0, 1.0)
    lexical_diversity = _clip(float(writing_quality_breakdown.get("lexical_diversity_score", 0.0)), 0.0, 1.0)
    sentence_variety = _clip(float(writing_quality_breakdown.get("sentence_variety_score", 0.0)), 0.0, 1.0)
    repetition_penalty = _clip(float(writing_quality_breakdown.get("repetition_penalty", 0.0)), 0.0, 1.0)

    writing_quality_component_raw = float(component_scores.get("writing_quality_component", 0.0))
    writing_quality_score = _component_to_score(writing_quality_component_raw, -10.0, 15.0)

    argument_logic = _clip(float(writing_quality_breakdown.get("argument_logic_score", 0.0)), 0.0, 1.0)
    information_density = _clip(float(writing_quality_breakdown.get("information_density_score", 0.0)), 0.0, 1.0)
    coherence = _clip(float(writing_quality_breakdown.get("coherence_score", 0.0)), 0.0, 1.0)
    counter_argument = _clip(float(writing_quality_breakdown.get("counter_argument_score", 0.0)), 0.0, 1.0)
    structure = _clip(float(writing_quality_breakdown.get("completeness_score", 0.0)), 0.0, 1.0)

    toxicity_penalty = abs(float(component_scores.get("toxicity_penalty", 0.0)))
    toxicity_score = _clip(toxicity_penalty / 25.0, 0.0, 1.0)
    toxicity_component = 1.0 - toxicity_score
    engagement_style = _clip(float(component_scores.get("fan_sincerity", 0.0)) / 15.0, 0.0, 1.0)

    sentiment_probs, sentiment_soft_component, sentiment_top, sentiment_second, sentiment_warnings = _soft_sentiment_component(signal_map)
    sentiment_gap = _clip(sentiment_top - sentiment_second, 0.0, 1.0)

    stat_accuracy_score_component = _to_score_range(stat_accuracy)

    component_map = {
        "stance_component": _clip((final_stance_component - SCORE_MIN) / (SCORE_MAX - SCORE_MIN), 0.0, 1.0),
        "stat_accuracy_component": stat_accuracy,
        "writing_quality_component": _clip((writing_quality_score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN), 0.0, 1.0),
        "argument_logic_component": argument_logic,
        "information_density_component": information_density,
        "toxicity_component": toxicity_component,
        "engagement_style_component": engagement_style,
        "structure_component": structure,
        "coherence_component": coherence,
        "counter_argument_component": counter_argument,
    }

    archetype_fusion = sum(archetype_weights[key] * component_map[key] for key in archetype_weights)

    originality_component = _clip(
        0.5 * lexical_diversity + 0.3 * sentence_variety - 0.2 * repetition_penalty,
        0.0,
        1.0,
    )
    originality_score_component = _to_score_range(originality_component)

    final_core_component = (
        0.45 * final_stance_component
        + 0.25 * sentiment_soft_component
        + 0.15 * writing_quality_score
        + 0.10 * stat_accuracy_score_component
        + 0.05 * originality_score_component
    )

    blended_average = 0.5 * (soft_stance_component + sentiment_soft_component)
    confidence_gap_factor = min(stance_gap, sentiment_gap)
    confidence_adjustment = 1.0
    if confidence_gap_factor < 0.10:
        confidence_adjustment = 0.98
        final_core_component = 0.90 * final_core_component + 0.10 * blended_average

    archetype_component = _to_score_range(archetype_fusion)
    final_score_raw = (0.80 * final_core_component + 0.20 * archetype_component) * confidence_adjustment

    final_score_raw += _stats_shape(stats_verification)

    toxicity_adjustment = "none"
    if toxicity_score > 0.75:
        final_score_raw = min(final_score_raw, 30.0)
        toxicity_adjustment = "cap_30"
    elif toxicity_score > 0.6:
        final_score_raw = min(final_score_raw, 40.0)
        toxicity_adjustment = "cap_40"

    attack_severity_score = 0
    attack_ceiling_applied = False
    final_ceiling_value: float | None = None

    attack_like_text = _attack_severity(blog_text) >= 1
    if stance_label == "DIRECT_ATTACK" or attack_like_text:
        attack_severity_score = _attack_severity(blog_text)
        ceiling_map = {
            0: 40.0,
            1: 30.0,
            2: 25.0,
            3: 20.0,
        }
        final_ceiling_value = ceiling_map.get(min(attack_severity_score, 3), 40.0)
        final_score_raw = min(final_score_raw, final_ceiling_value)
        attack_ceiling_applied = True

    dismissive_like = stance_label == "DISMISSIVE_COMPLAINT" or _is_dismissive_text(blog_text)
    if dismissive_like and not attack_ceiling_applied:
        final_ceiling_value = _dismissive_ceiling(blog_text)
        final_score_raw = min(final_score_raw, final_ceiling_value)
        attack_ceiling_applied = True

    calibration_band = STANCE_BANDS.get(stance_label, (SCORE_MIN, SCORE_MAX))
    if attack_ceiling_applied and final_ceiling_value is not None:
        # Preserve strict safety ceiling; do not allow band minimums to lift toxic content.
        final_score_raw = min(final_score_raw, final_ceiling_value)
    else:
        final_score_raw = _clip(final_score_raw, calibration_band[0], calibration_band[1])
    final_bqs_score = round(_clip(final_score_raw, SCORE_MIN, SCORE_MAX), 2)

    archetype_detected = max(archetype_scores.items(), key=lambda item: item[1])[0]

    print(
        "BQS SOFT FUSION DEBUG:",
        stance_probs,
        sentiment_probs,
        round(stance_top, 4),
        round(stance_second, 4),
        round(stance_gap, 4),
        round(dominant_stance_component, 4),
        round(soft_stance_component, 4),
        round(sentiment_soft_component, 4),
        round(final_core_component, 4),
        final_bqs_score,
    )

    return {
        "final_bqs_score": final_bqs_score,
        "archetype_detected": archetype_detected,
        "stance_label": stance_label,
        "stance_strength": round(stance_strength_value, 4),
        "stance_strength_parts": stance_strength_parts,
        "stat_accuracy_component": round(stat_accuracy, 4),
        "writing_quality_component": round(_clip((writing_quality_score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN), 0.0, 1.0), 4),
        "originality_component": round(originality_component, 4),
        "toxicity_adjustment": toxicity_adjustment,
        "confidence_gap_factor": round(confidence_gap_factor, 4),
        "confidence_adjustment": round(confidence_adjustment, 4),
        "attack_severity_score": attack_severity_score,
        "attack_ceiling_applied": attack_ceiling_applied,
        "final_ceiling_value": final_ceiling_value,
        "soft_fusion_debug": {
            "stance_probabilities": {k: round(v, 4) for k, v in stance_probs.items()},
            "sentiment_probabilities": {k: round(v, 4) for k, v in sentiment_probs.items()},
            "top1_probability": round(stance_top, 4),
            "second_probability": round(stance_second, 4),
            "probability_gap": round(stance_gap, 4),
            "dominant_stance": dominant_stance_label,
            "dominant_stance_component": round(dominant_stance_component, 4),
            "soft_stance_component": round(soft_stance_component, 4),
            "sentiment_soft_component": round(sentiment_soft_component, 4),
            "final_stance_component": round(final_stance_component, 4),
            "final_core_component": round(final_core_component, 4),
            "attack_severity_score": attack_severity_score,
            "attack_ceiling_applied": attack_ceiling_applied,
            "ceiling_value": final_ceiling_value,
            "warnings": stance_warnings + sentiment_warnings,
        },
        "calibration_band": {
            "min": calibration_band[0],
            "max": calibration_band[1],
        },
        "archetype_scores": {k: round(v, 4) for k, v in archetype_scores.items()},
        "archetype_weights": {k: round(v, 4) for k, v in archetype_weights.items()},
        "fusion_debug": {
            "final_core_component": round(final_core_component, 4),
            "archetype_fusion": round(archetype_fusion, 4),
            "archetype_component": round(archetype_component, 4),
            "toxicity_score": round(toxicity_score, 4),
        },
    }
