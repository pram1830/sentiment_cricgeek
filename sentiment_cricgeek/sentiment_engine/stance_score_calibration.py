from __future__ import annotations

from typing import Any, Dict, Tuple


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


STANCE_SCORE_BANDS: Dict[str, Tuple[float, float]] = {
    "SUPPORTIVE_DEFENSE": (70.0, 85.0),
    "CONSTRUCTIVE_CRITICISM": (62.0, 82.0),
    "NEUTRAL_ANALYSIS": (45.0, 65.0),
    "BALANCED_DEBATE": (55.0, 75.0),
    "DISMISSIVE_COMPLAINT": (25.0, 50.0),
    "DIRECT_ATTACK": (0.0, 25.0),
    "MIXED_STANCE": (35.0, 68.0),
}


def _outside_band_deviation(score: float, band: Tuple[float, float]) -> float:
    low, high = band
    if score < low:
        return low - score
    if score > high:
        return score - high
    return 0.0


def _pull_into_band(score: float, band: Tuple[float, float], strength: float) -> float:
    low, high = band
    if score < low:
        return score + strength * (low - score)
    if score > high:
        return score - strength * (score - high)
    return score


def apply_stance_score_calibration(result: Dict[str, Any]) -> Dict[str, Any]:
    calibrated = dict(result)
    stance_label = str(calibrated.get("stance_label", "NEUTRAL_ANALYSIS"))
    stance_confidence = float(calibrated.get("stance_confidence", 0.0))
    supportive_defense_strength = float(calibrated.get("supportive_defense_strength", 0.0))
    score = float(calibrated.get("final_score", calibrated.get("score_out_of_100", 0.0)))

    band = STANCE_SCORE_BANDS.get(stance_label, (0.0, 100.0))
    low, high = band

    new_score = score
    reason = "no_change"

    if stance_label == "SUPPORTIVE_DEFENSE":
        if supportive_defense_strength >= 0.25 and supportive_defense_strength < 0.40:
            new_score = max(score, 72.0, 70.0 + 5.0 * supportive_defense_strength)
            new_score = min(new_score, 82.0)
            reason = "supportive_mid_strength_passthrough"
            print("CALIBRATION SUPPORTIVE MID-STRENGTH PASS-THROUGH ACTIVE")
        elif stance_confidence >= 0.55:
            new_score = _clip(score, low, high)
            reason = "supportive_confident_band_map"
        elif stance_confidence >= 0.35:
            new_score = _pull_into_band(score, (low, high), 0.9)
            reason = "supportive_medium_conf_pull"
    elif stance_label == "CONSTRUCTIVE_CRITICISM":
        if stance_confidence >= 0.5:
            new_score = _clip(score, low, high)
            reason = "constructive_confident_band_map"
        elif stance_confidence >= 0.32:
            new_score = _pull_into_band(score, (low, high), 0.85)
            reason = "constructive_medium_conf_pull"
    elif stance_label == "DISMISSIVE_COMPLAINT":
        new_score = min(score, high)
        reason = "dismissive_cap"
    elif stance_label == "DIRECT_ATTACK":
        new_score = min(score, high)
        reason = "direct_attack_cap"
    elif stance_label in {"NEUTRAL_ANALYSIS", "BALANCED_DEBATE", "MIXED_STANCE"}:
        if stance_confidence >= 0.45:
            new_score = _clip(score, low, high)
            reason = "neutral_family_band_map"
        elif stance_confidence >= 0.25:
            new_score = _pull_into_band(score, (low, high), 0.85)
            reason = "neutral_family_medium_conf_pull"

    new_score = round(_clip(new_score, 0.0, 100.0), 2)
    calibrated["final_score"] = new_score
    calibrated["score_out_of_100"] = new_score

    if "component_scores" not in calibrated:
        calibrated["component_scores"] = {}

    calibrated["stance_calibration"] = {
        "applied": new_score != score,
        "reason": reason,
        "input_score": round(score, 2),
        "output_score": new_score,
        "stance_label": stance_label,
        "stance_confidence": round(stance_confidence, 4),
        "target_band": [low, high],
        "outside_band_deviation_before": round(_outside_band_deviation(score, band), 2),
        "outside_band_deviation_after": round(_outside_band_deviation(new_score, band), 2),
    }

    return calibrated
