from __future__ import annotations

import re
from typing import Any, Dict, List


ALLOWED_WRITERS = {"Analyst", "Debater", "Storyteller", "All-Rounder"}
REASONING_TERMS = [
    "because",
    "therefore",
    "however",
    "while",
    "although",
    "whereas",
    "if",
    "then",
    "shows",
    "suggests",
    "approach",
    "problem",
    "structure",
    "evidence",
    "argues",
    "argument",
    "compare",
    "comparing",
    "balanced view",
    "both sides",
    "middle ground",
    "on the other hand",
    "next steps",
    "resilience",
    "adjustment",
]
SUGGESTION_TERMS = [
    "should",
    "could",
    "need to",
    "needs to",
    "better approach",
    "adjust",
    "improve",
    "fix",
    "next steps",
]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _is_complaint_like(paragraph_diagnostics: List[Dict[str, Any]], signals: Dict[str, float], component_scores: Dict[str, float]) -> bool:
    text_blob = " ".join(str(item.get("text", "")).lower() for item in paragraph_diagnostics)
    complaint_patterns = [
        r"same script",
        r"nothing ever changes",
        r"more of the same",
        r"another frustrating",
        r"every week",
        r"recurring pattern",
        r"tired of",
        r"frustrating",
        r"disappointed",
    ]
    repeated = any(len(re.findall(pattern, text_blob)) >= 1 for pattern in complaint_patterns)
    lexical_reasoning_hits = sum(1 for term in REASONING_TERMS if term in text_blob)
    lexical_suggestion_hits = sum(1 for term in SUGGESTION_TERMS if term in text_blob)
    complaint_tone_hits = sum(1 for pattern in complaint_patterns if re.search(pattern, text_blob) is not None)
    return (
        repeated
        and complaint_tone_hits >= 1
        and float(signals.get("mean_suggestion_score", 0.0)) < 0.12
        and lexical_suggestion_hits <= 1
        and float(signals.get("mean_reasoning_marker_score", 0.0)) < 0.3
        and lexical_reasoning_hits <= 2
        and float(signals.get("mean_discourse_score", 0.0)) < 0.25
    )


def apply_deterministic_rescue(
    writer_type: str,
    component_scores: Dict[str, float],
    total: float,
    toxicity: Dict[str, float],
    signals: Dict[str, float],
    paragraph_diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    result = {
        "component_scores": dict(component_scores),
        "total": float(total),
        "activated": False,
        "rescued_outlier": False,
        "analytical_boost_applied": False,
        "confidence_adjustment": 0.0,
        "reason": "not_eligible",
    }

    if writer_type not in ALLOWED_WRITERS:
        result["reason"] = "writer_not_eligible"
        return result

    low_toxicity = float(toxicity.get("mean", 0.0)) < 0.18 and float(toxicity.get("abuse_mean", 0.0)) < 0.08
    respectful = float(signals.get("mean_respect_score", 0.0)) >= 0.65
    text_blob = " ".join(str(item.get("text", "")).lower() for item in paragraph_diagnostics)
    lexical_reasoning_hits = sum(1 for term in REASONING_TERMS if term in text_blob)
    lexical_suggestion_hits = sum(1 for term in SUGGESTION_TERMS if term in text_blob)

    has_reasoning = (
        float(signals.get("mean_reasoning_marker_score", 0.0)) >= 0.08
        or lexical_reasoning_hits >= 1
        or float(signals.get("mean_discourse_score", 0.0)) >= 0.35
        or float(component_scores.get("analytical_tone", 0.0)) >= 6.5
    )
    has_suggestion_or_depth = (
        float(signals.get("mean_suggestion_score", 0.0)) >= 0.12
        or float(signals.get("mean_explanation_depth_score", 0.0)) >= 0.2
        or float(signals.get("mean_discourse_score", 0.0)) >= 0.4
        or lexical_suggestion_hits >= 2
    )
    sustained_reasoning_length = (
        float(signals.get("mean_paragraph_word_count", 0.0)) >= 70.0
        and float(signals.get("mean_paragraph_sentence_count", 0.0)) >= 4.0
    )

    strategic_reasoning = float(signals.get("mean_strategic_cricket_reasoning_score", 0.0))

    # Analyst fast gate: strategic cricket reasoning should directly unlock rescue under low toxicity and respectful tone.
    analyst_strategic_gate = (
        writer_type == "Analyst"
        and low_toxicity
        and respectful
        and strategic_reasoning >= 0.2
    )

    simple_reasoning_gate = (
        low_toxicity
        and float(toxicity.get("mean", 0.0)) < 0.05
        and float(signals.get("mean_respect_score", 0.0)) > 0.7
        and (
            float(signals.get("mean_reasoning_marker_score", 0.0)) >= 0.4
            or float(signals.get("mean_explanation_depth_score", 0.0)) >= 0.4
        )
    )

    if not (
        analyst_strategic_gate
        or simple_reasoning_gate
        or (low_toxicity and respectful and has_reasoning and has_suggestion_or_depth and sustained_reasoning_length)
    ):
        result["reason"] = "signal_gate_failed"
        return result

    if _is_complaint_like(paragraph_diagnostics, signals, result["component_scores"]):
        result["reason"] = "complaint_like_blocked"
        return result

    result["activated"] = True
    result["reason"] = "simple_reasoning_detected" if simple_reasoning_gate else "constructive_rescue"

    if simple_reasoning_gate:
        analytical_uplift = 0.35
        constructiveness_uplift = 0.25
        result["component_scores"]["analytical_tone"] = round(
            _clip(float(result["component_scores"].get("analytical_tone", 0.0)) + analytical_uplift, 0.0, 15.0),
            2,
        )
        result["component_scores"]["constructiveness"] = round(
            _clip(float(result["component_scores"].get("constructiveness", 0.0)) + constructiveness_uplift, 0.0, 40.0),
            2,
        )
        result["total"] = (
            float(result["component_scores"].get("constructiveness", 0.0))
            + float(result["component_scores"].get("respectfulness", 0.0))
            + float(result["component_scores"].get("analytical_tone", 0.0))
            + float(result["component_scores"].get("clarity", 0.0))
            + float(result["component_scores"].get("fan_sincerity", 0.0))
            + float(result["component_scores"].get("toxicity_penalty", 0.0))
        )

    stance_label = str(signals.get("stance_label", ""))
    supportive_defense_strength = float(signals.get("supportive_defense_strength", 0.0))
    if (
        stance_label == "SUPPORTIVE_DEFENSE"
        and supportive_defense_strength >= 0.4
        and float(signals.get("mean_explanation_depth_score", 0.0)) >= 0.3
        and float(signals.get("mean_respect_score", 0.0)) >= 0.7
    ):
        result["component_scores"]["constructiveness"] = round(
            _clip(float(result["component_scores"].get("constructiveness", 0.0)) + 0.2, 0.0, 40.0),
            2,
        )
        result["component_scores"]["analytical_tone"] = round(
            _clip(float(result["component_scores"].get("analytical_tone", 0.0)) + 0.15, 0.0, 15.0),
            2,
        )
        result["reason"] = "fairness_defense_detected"
        result["total"] = (
            float(result["component_scores"].get("constructiveness", 0.0))
            + float(result["component_scores"].get("respectfulness", 0.0))
            + float(result["component_scores"].get("analytical_tone", 0.0))
            + float(result["component_scores"].get("clarity", 0.0))
            + float(result["component_scores"].get("fan_sincerity", 0.0))
            + float(result["component_scores"].get("toxicity_penalty", 0.0))
        )

    # Controlled long-form analytical boost: small bump to correct systematic under-scoring.
    if (
        (float(signals.get("mean_reasoning_marker_score", 0.0)) >= 0.2 or lexical_reasoning_hits >= 4)
        and float(signals.get("mean_explanation_depth_score", 0.0)) >= 0.25
        and float(signals.get("constructive_pattern_ratio", 0.0)) >= 0.15
        and float(signals.get("mean_respect_score", 0.0)) >= 0.72
        and float(toxicity.get("mean", 0.0)) < 0.12
    ):
        result["component_scores"]["constructiveness"] = round(
            _clip(float(result["component_scores"].get("constructiveness", 0.0)) + 2.0, 0.0, 40.0),
            2,
        )
        result["total"] += 2.0
        result["analytical_boost_applied"] = True

    # Minimum floors for constructive respectful analytical writing.
    if float(result["component_scores"].get("constructiveness", 0.0)) < 24.0:
        delta = 24.0 - float(result["component_scores"].get("constructiveness", 0.0))
        result["component_scores"]["constructiveness"] = 24.0
        result["total"] += delta

    target_floor = 73.0 if writer_type in {"Analyst", "Debater"} else 68.0
    if analyst_strategic_gate:
        target_floor = max(target_floor, 80.0)
    if result["total"] < target_floor:
        result["total"] = target_floor
        result["rescued_outlier"] = True

    result["confidence_adjustment"] = 0.08
    return result
