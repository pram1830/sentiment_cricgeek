from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset

try:
    from sentiment_engine.constructiveness_detector import ConstructivenessDetector
    from sentiment_engine.model_loader import ModelLoader
    from sentiment_engine.sentiment_pipeline import SentimentPipeline
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from sentiment_engine.constructiveness_detector import ConstructivenessDetector
    from sentiment_engine.model_loader import ModelLoader
    from sentiment_engine.sentiment_pipeline import SentimentPipeline


SPORTS_KEYWORDS = {
    "cricket",
    "football",
    "basketball",
    "match",
    "team",
    "player",
    "selection",
    "captain",
    "bowling",
    "batting",
    "inning",
    "innings",
    "coach",
    "goal",
    "tournament",
    "league",
}

REASONING_MARKERS = ["because", "therefore", "so that", "as a result", "which means", "reason"]
SUGGESTION_MARKERS = ["should", "could", "recommend", "suggest", "needs to", "would help", "better to"]
COMPARISON_MARKERS = ["however", "while", "although", "on the other hand", "compared", "rather than"]
ABUSIVE_TERMS = [
    "idiot",
    "moron",
    "stupid",
    "trash",
    "garbage",
    "pathetic",
    "loser",
    "useless",
    "hate you",
    "clown",
]


def _contains_sports_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in SPORTS_KEYWORDS)


def _extract_text(row: Dict[str, Any]) -> Optional[str]:
    for key in ["text", "tweet", "content", "sentence"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_label(row: Dict[str, Any]) -> Any:
    for key in ["label", "sentiment", "labels"]:
        if key in row:
            return row[key]
    return None


def _map_to_sentiment(label: Any) -> str:
    # tweet_eval/sentiment default map: 0 negative, 1 neutral, 2 positive
    if isinstance(label, int):
        if label == 0:
            return "negative"
        if label == 1:
            return "neutral"
        if label == 2:
            return "positive"
    label_text = str(label).lower()
    if "pos" in label_text:
        return "positive"
    if "neu" in label_text:
        return "neutral"
    return "negative"


def _heuristic_constructive_target(text: str) -> int:
    lowered = text.lower()
    has_reasoning = any(marker in lowered for marker in REASONING_MARKERS)
    has_suggestion = any(marker in lowered for marker in SUGGESTION_MARKERS)
    has_comparison = any(marker in lowered for marker in COMPARISON_MARKERS)
    has_explanation_connector = bool(re.search(r"\bif\b|\bthen\b|\bso\b|\bthus\b", lowered))

    signal_count = sum([has_reasoning, has_suggestion, has_comparison, has_explanation_connector])
    return 1 if signal_count >= 2 else 0


def _heuristic_toxic_target(text: str) -> int:
    lowered = text.lower()
    direct_abuse = any(term in lowered for term in ABUSIVE_TERMS)
    personal_attack = bool(re.search(r"\byou are\b|\bhe is\b|\bthey are\b", lowered))
    return 1 if (direct_abuse or personal_attack) else 0


def _extract_toxicity_from_classifier(output: Any) -> float:
    if isinstance(output, list) and output and isinstance(output[0], list):
        pairs = {str(item.get("label", "")).lower().replace("-", "_"): float(item.get("score", 0.0)) for item in output[0]}
        if "toxic" in pairs:
            return pairs["toxic"]
        if "non_toxic" in pairs:
            return 1.0 - pairs["non_toxic"]

    if isinstance(output, list) and output and isinstance(output[0], dict):
        label = str(output[0].get("label", "")).lower()
        score = float(output[0].get("score", 0.0))
        if "toxic" in label and "non" not in label:
            return score
        if "non" in label and "toxic" in label:
            return 1.0 - score

    return 0.0


def _load_sports_rows(minimum: int = 100, target: int = 220) -> List[Tuple[str, Any]]:
    print("Loading dataset")
    dataset = load_dataset("tweet_eval", "sentiment")

    print("Filtering sports samples")
    rows: List[Tuple[str, Any]] = []
    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]
        for row in split:
            text = _extract_text(row)
            if not text:
                continue
            if not _contains_sports_keyword(text):
                continue
            rows.append((text, _extract_label(row)))
            if len(rows) >= target:
                break
        if len(rows) >= target:
            break

    if len(rows) < minimum:
        raise RuntimeError(f"Only found {len(rows)} sports rows; need at least {minimum}.")

    random.seed(7)
    random.shuffle(rows)
    return rows[: max(minimum, min(target, len(rows)))]


def run_validation(sample_count: int = 120) -> Dict[str, Any]:
    rows = _load_sports_rows(minimum=100, target=max(140, sample_count))
    rows = rows[:sample_count]

    pipeline = SentimentPipeline()
    models = ModelLoader().load()
    constructiveness_detector = ConstructivenessDetector()

    print("Running constructiveness detection")
    print("Running toxicity detection")

    construct_correct = 0
    toxicity_correct = 0
    toxicity_fp = 0
    toxicity_fn = 0

    predictions: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    deviations: List[float] = []
    writer_counter: Counter = Counter()

    for idx, (text, raw_label) in enumerate(rows, start=1):
        mapped_sentiment = _map_to_sentiment(raw_label)
        constructive_target = _heuristic_constructive_target(text)
        toxic_target = _heuristic_toxic_target(text)

        detector_result = constructiveness_detector.detect(text=text, embedder=models.embedder)
        constructive_pred = 1 if detector_result.constructiveness_score >= 0.5 else 0
        construct_correct += int(constructive_pred == constructive_target)

        tox_output = models.toxicity_classifier(text)
        toxicity_score = _extract_toxicity_from_classifier(tox_output)
        toxic_pred = 1 if toxicity_score >= 0.5 else 0
        toxicity_correct += int(toxic_pred == toxic_target)

        if toxic_pred == 1 and toxic_target == 0:
            toxicity_fp += 1
        if toxic_pred == 0 and toxic_target == 1:
            toxicity_fn += 1

        score_result = pipeline.score(text, enable_logs=False)
        writer_type = score_result["writer_type"]
        writer_counter[writer_type] += 1

        sentiment_base = {"positive": 78.0, "neutral": 66.0, "negative": 58.0}[mapped_sentiment]
        expected_estimate = sentiment_base + 10.0 * constructive_target - 22.0 * toxic_target
        expected_estimate = max(0.0, min(100.0, expected_estimate))
        deviation = abs(score_result["final_score"] - expected_estimate)
        deviations.append(deviation)

        sample_pred = {
            "id": idx,
            "text": text,
            "dataset_sentiment": mapped_sentiment,
            "writer_type": writer_type,
            "final_score": score_result["final_score"],
            "constructiveness_score": round(detector_result.constructiveness_score, 4),
            "constructive_target": constructive_target,
            "constructive_prediction": constructive_pred,
            "toxicity_score": round(float(toxicity_score), 4),
            "toxicity_target": toxic_target,
            "toxicity_prediction": toxic_pred,
            "confidence_estimates": {
                "constructiveness": round(detector_result.confidence, 4),
                "toxicity": round(min(1.0, abs(float(toxicity_score) - 0.5) * 2.0), 4),
            },
            "score_deviation_estimate": round(float(deviation), 4),
        }
        predictions.append(sample_pred)

        if (constructive_pred != constructive_target) or (toxic_pred != toxic_target):
            errors.append(sample_pred)

    total = len(rows)
    construct_acc = (construct_correct / total) * 100.0
    tox_acc = (toxicity_correct / total) * 100.0

    toxic_count = sum(_heuristic_toxic_target(t) for t, _ in rows)
    non_toxic_count = total - toxic_count

    fp_rate = (toxicity_fp / non_toxic_count) * 100.0 if non_toxic_count > 0 else 0.0
    fn_rate = (toxicity_fn / toxic_count) * 100.0 if toxic_count > 0 else 0.0

    report = {
        "metrics": {
            "total_evaluated_samples": total,
            "constructiveness_detection_accuracy_percent": round(construct_acc, 2),
            "toxicity_detection_accuracy_percent": round(tox_acc, 2),
            "false_positive_toxicity_percent": round(fp_rate, 2),
            "false_negative_toxicity_percent": round(fn_rate, 2),
            "toxicity_target_count": toxic_count,
            "non_toxicity_target_count": non_toxic_count,
            "average_score_deviation_estimate": round(float(np.mean(deviations)) if deviations else 0.0, 2),
            "writer_type_detection_distribution": dict(writer_counter),
        },
        "sample_predictions": predictions[:40],
        "error_examples": errors[:30],
        "confidence_estimates": {
            "average_constructiveness_confidence": round(float(np.mean([p["confidence_estimates"]["constructiveness"] for p in predictions])), 4),
            "average_toxicity_confidence": round(float(np.mean([p["confidence_estimates"]["toxicity"] for p in predictions])), 4),
        },
    }

    return report


def main() -> None:
    report = run_validation(sample_count=120)

    print("Generating validation report")
    metrics = report["metrics"]
    print(f"Total evaluated samples: {metrics['total_evaluated_samples']}")
    print(f"Constructiveness detection accuracy %: {metrics['constructiveness_detection_accuracy_percent']}")
    print(f"Toxicity detection accuracy %: {metrics['toxicity_detection_accuracy_percent']}")
    print(f"False positive toxicity %: {metrics['false_positive_toxicity_percent']}")
    print(f"False negative toxicity %: {metrics['false_negative_toxicity_percent']}")
    print(f"Average score deviation estimate: {metrics['average_score_deviation_estimate']}")
    print(f"Writer-type detection distribution across dataset: {metrics['writer_type_detection_distribution']}")

    out_path = os.path.join(os.getcwd(), "sports_validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
