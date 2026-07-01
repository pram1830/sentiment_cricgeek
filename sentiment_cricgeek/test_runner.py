from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Tuple

from sentiment_engine.sentiment_pipeline import SentimentPipeline


DELIMITER = "===BLOG==="
BENCHMARK_DIR = Path(__file__).resolve().parent / "sentiment_engine" / "benchmarks"
DEFAULT_BENCHMARK_PATH = (
    BENCHMARK_DIR / "cricgeek_stance_benchmark_extended.json"
    if (BENCHMARK_DIR / "cricgeek_stance_benchmark_extended.json").exists()
    else BENCHMARK_DIR / "cricgeek_stance_benchmark_200.json"
)


def _read_multiline_input() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    print("Paste one or more blog texts. Separate each blog with ===BLOG===.")
    print("End input with Ctrl+Z then Enter (Windows) or Ctrl+D (Unix).")
    lines: List[str] = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines).strip()


def _split_blog_blocks(raw_text: str) -> List[str]:
    blocks = [block.strip() for block in raw_text.split(DELIMITER)]
    return [block for block in blocks if block]


def _print_blog_result(index: int, result: Dict[str, Any]) -> None:
    component_scores = result.get("component_scores", {})

    print(f"\nBLOG {index} RESULT")
    print(f"final_score: {result.get('final_score', 0.0)}")
    print(f"stance_label: {result.get('stance_label', 'NEUTRAL_ANALYSIS')}")
    print(f"primary_stance_label: {result.get('primary_stance_label', 'NEUTRAL_COMMENTARY')}")
    print(f"stance_confidence: {result.get('stance_confidence', 0.0)}")
    print(f"primary_stance_confidence: {result.get('primary_stance_confidence', 0.0)}")
    print(f"style_tags: {result.get('style_tags', [])}")
    print(f"scalar_metrics: {result.get('scalar_metrics', {})}")
    print(f"sarcasm_gate_reason: {result.get('sarcasm_gate_reason', '')}")
    print(f"supportive_defense_strength: {result.get('supportive_defense_strength', 0.0)}")
    print(f"criticism_reference_score: {result.get('criticism_reference_score', 0.0)}")
    print(f"context_change_score: {result.get('context_change_score', 0.0)}")
    print(f"evaluation_redirection_score: {result.get('evaluation_redirection_score', 0.0)}")
    print(f"credibility_restoration_score: {result.get('credibility_restoration_score', 0.0)}")
    print(f"contrast_structure_score: {result.get('contrast_structure_score', 0.0)}")
    print(f"causal_defense_score: {result.get('causal_defense_score', 0.0)}")
    print(f"credibility_defense_score: {result.get('credibility_defense_score', 0.0)}")
    print(f"attack_rejection_detected: {result.get('attack_rejection_detected', False)}")
    print(f"attack_endorsement_detected: {result.get('attack_endorsement_detected', False)}")
    print(f"toxicity_penalty: {component_scores.get('toxicity_penalty', 0.0)}")


def _print_summary_table(results: List[Dict[str, Any]]) -> None:
    print("\nSUMMARY COMPARISON")
    print("BLOG | LEGACY_STANCE | PRIMARY_STANCE | SCORE | SUPPORTIVE_STRENGTH | TOXICITY")
    for idx, result in enumerate(results, start=1):
        stance = str(result.get("stance_label", "NEUTRAL_ANALYSIS"))
        primary_stance = str(result.get("primary_stance_label", "NEUTRAL_COMMENTARY"))
        score = float(result.get("final_score", 0.0))
        supportive = float(result.get("supportive_defense_strength", 0.0))
        toxicity = float(result.get("component_scores", {}).get("toxicity_penalty", 0.0))
        print(f"{idx} | {stance} | {primary_stance} | {score:.2f} | {supportive:.4f} | {toxicity:.2f}")


def _load_benchmark(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and item.get("text") and item.get("expected_stance")]


def _score_deviation(score: float, band: Tuple[float, float]) -> float:
    low, high = band
    if score < low:
        return low - score
    if score > high:
        return score - high
    return 0.0


def _run_benchmark(benchmark_path: Path) -> None:
    benchmark = _load_benchmark(benchmark_path)
    if not benchmark:
        print(f"No benchmark samples found at: {benchmark_path}")
        return

    pipeline = SentimentPipeline()
    per_stance_totals: Dict[str, int] = {}
    per_stance_correct: Dict[str, int] = {}
    per_stance_deviation: Dict[str, float] = {}
    sample_rows: List[Dict[str, Any]] = []

    print(f"Running benchmark: {benchmark_path}")
    print("IDX | EXPECTED | PREDICTED | CONF | SCORE | ERROR")

    for idx, sample in enumerate(benchmark, start=1):
        text = str(sample.get("text", ""))
        expected_stance = str(sample.get("expected_stance", "NEUTRAL_ANALYSIS"))
        expected_score_band_raw = sample.get("expected_score_band", [0.0, 100.0])
        expected_toxicity_band_raw = sample.get("expected_toxicity_band", [0.0, 1.0])
        expected_supportive_band_raw = sample.get("expected_supportive_strength_band", [0.0, 1.0])

        # LOOCV: remove current sample from the stance benchmark training set.
        train_samples = [row for j, row in enumerate(benchmark) if j != (idx - 1)]
        detector = pipeline.stance_detector

        def _patched_loader(self, rows=train_samples):
            return rows

        detector._load_benchmark_samples = types.MethodType(_patched_loader, detector)
        detector._benchmark_ready = False
        detector._centroid_embeddings = {}
        detector._centroid_features = {}
        detector._benchmark_embeddings = None
        detector._benchmark_labels = []

        expected_score_band = (float(expected_score_band_raw[0]), float(expected_score_band_raw[1]))
        expected_toxicity_band = (float(expected_toxicity_band_raw[0]), float(expected_toxicity_band_raw[1]))
        expected_supportive_band = (float(expected_supportive_band_raw[0]), float(expected_supportive_band_raw[1]))

        result = pipeline.score(text, enable_logs=False)

        predicted_stance = str(result.get("stance_label", "NEUTRAL_ANALYSIS"))
        confidence = float(result.get("stance_confidence", 0.0))
        score = float(result.get("final_score", 0.0))
        toxicity = abs(float(result.get("component_scores", {}).get("toxicity_penalty", 0.0)) / 25.0)
        supportive_strength = float(result.get("supportive_defense_strength", 0.0))

        score_dev = _score_deviation(score, expected_score_band)
        score_hit = score_dev == 0.0
        toxicity_hit = expected_toxicity_band[0] <= toxicity <= expected_toxicity_band[1]
        supportive_hit = expected_supportive_band[0] <= supportive_strength <= expected_supportive_band[1]
        stance_hit = predicted_stance == expected_stance

        if not stance_hit:
            error_type = "stance_mismatch"
        elif not score_hit:
            error_type = "score_band_miss"
        elif not toxicity_hit:
            error_type = "toxicity_band_miss"
        elif not supportive_hit:
            error_type = "supportive_band_miss"
        else:
            error_type = "ok"

        per_stance_totals[expected_stance] = per_stance_totals.get(expected_stance, 0) + 1
        per_stance_correct[expected_stance] = per_stance_correct.get(expected_stance, 0) + (1 if stance_hit else 0)
        per_stance_deviation[expected_stance] = per_stance_deviation.get(expected_stance, 0.0) + score_dev
        sample_rows.append(
            {
                "index": idx,
                "expected_stance": expected_stance,
                "predicted_stance": predicted_stance,
                "confidence": round(confidence, 4),
                "score": round(score, 2),
                "expected_score_band": [expected_score_band[0], expected_score_band[1]],
                "score_deviation": round(score_dev, 2),
                "toxicity": round(toxicity, 4),
                "expected_toxicity_band": [expected_toxicity_band[0], expected_toxicity_band[1]],
                "supportive_strength": round(supportive_strength, 4),
                "expected_supportive_strength_band": [expected_supportive_band[0], expected_supportive_band[1]],
                "error_type": error_type,
            }
        )

        print(f"{idx} | {expected_stance} | {predicted_stance} | {confidence:.4f} | {score:.2f} | {error_type}")

    print("\nOVERALL ACCURACY BY STANCE")
    print("STANCE | ACCURACY | COUNT")
    total_predictions = 0
    total_correct = 0
    for stance in sorted(per_stance_totals.keys()):
        total = per_stance_totals[stance]
        correct = per_stance_correct.get(stance, 0)
        accuracy = (correct / total) * 100.0 if total else 0.0
        total_predictions += total
        total_correct += correct
        print(f"{stance} | {accuracy:.2f}% | {total}")

    overall_accuracy = (total_correct / total_predictions) * 100.0 if total_predictions else 0.0
    print(f"\nOVERALL ACCURACY: {overall_accuracy:.2f}%")

    print("\nAVERAGE SCORE DEVIATION BY STANCE")
    print("STANCE | AVG_DEVIATION")
    summary_rows: List[Dict[str, Any]] = []
    for stance in sorted(per_stance_totals.keys()):
        total = per_stance_totals[stance]
        avg_dev = per_stance_deviation.get(stance, 0.0) / total if total else 0.0
        correct = per_stance_correct.get(stance, 0)
        accuracy = (correct / total) * 100.0 if total else 0.0
        summary_rows.append(
            {
                "stance": stance,
                "count": total,
                "accuracy_percent": round(accuracy, 2),
                "average_score_deviation": round(avg_dev, 2),
            }
        )
        print(f"{stance} | {avg_dev:.2f}")

    report = {
        "benchmark_path": str(benchmark_path),
        "evaluation_mode": "leave_one_out_cross_validation",
        "total_samples": len(sample_rows),
        "overall_accuracy_percent": round(overall_accuracy, 2),
        "samples": sample_rows,
        "summary_by_stance": summary_rows,
    }
    benchmark_name = benchmark_path.name.lower()
    out_name = "stance_benchmark_report_loocv_extended.json" if "extended" in benchmark_name else "stance_benchmark_report_loocv.json"
    out_path = Path(__file__).resolve().parent / out_name
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark report: {out_path}")


def _run_sarcasm_tests() -> None:
    test_cases: List[Dict[str, Any]] = [
        {
            "name": "direct_praise",
            "text": "He handled pressure well, adapted to role changes, and played with clear team-first intent.",
            "expected_primary": ["SUPPORTIVE_DEFENSE"],
            "required_style": ["DIRECT"],
            "forbidden_primary": ["SARCASTIC_ATTACK"],
        },
        {
            "name": "direct_criticism",
            "text": "The selection has repeated mistakes and the approach is clearly not working.",
            "expected_primary": ["DIRECT_CRITICISM"],
            "required_style": ["DIRECT"],
        },
        {
            "name": "sarcastic_praise_means_criticism",
            "text": "His batting is a masterclass in suspense... visionary, really.",
            "expected_primary": ["SARCASTIC_ATTACK", "DIRECT_CRITICISM"],
            "required_style": ["SARCASTIC"],
            "forbidden_primary": ["SUPPORTIVE_DEFENSE"],
        },
        {
            "name": "hyperbolic_mockery",
            "text": "What a legendary strategy from a completely different geological era of cricket.",
            "expected_primary": ["SARCASTIC_ATTACK"],
            "required_style": ["SARCASTIC", "HYPERBOLIC"],
            "forbidden_primary": ["SUPPORTIVE_DEFENSE"],
        },
        {
            "name": "rhetorical_sarcasm",
            "text": "Brilliant planning, right? Is this how modern aggression is supposed to look?",
            "expected_primary": ["SARCASTIC_ATTACK", "MIXED_OR_AMBIGUOUS"],
            "required_style": ["RHETORICAL"],
            "forbidden_primary": ["SUPPORTIVE_DEFENSE"],
        },
        {
            "name": "neutral_reporting",
            "text": "Some analysts said role flexibility improved balance while others focused on strike rotation trends.",
            "expected_primary": ["NEUTRAL_COMMENTARY", "MIXED_OR_AMBIGUOUS"],
            "required_style": ["REPORTED"],
        },
        {
            "name": "mixed_opinion",
            "text": "He has improved composure and role clarity, but same mistake again keeps appearing and management never learns.",
            "expected_primary": ["MIXED_OR_AMBIGUOUS", "DIRECT_CRITICISM"],
        },
        {
            "name": "quoted_third_person_commentary",
            "text": "People say he is finished, but others believe his role changed and context matters now.",
            "expected_primary": ["MIXED_OR_AMBIGUOUS", "NEUTRAL_COMMENTARY", "SUPPORTIVE_DEFENSE"],
            "required_style": ["REPORTED", "QUOTED"],
        },
        {
            "name": "mocking_intent_wording_variant",
            "text": "A visionary innings again, because making boundaries optional is apparently elite modern strategy.",
            "expected_primary": ["SARCASTIC_ATTACK", "DIRECT_CRITICISM"],
            "required_style": ["SARCASTIC"],
            "forbidden_primary": ["SUPPORTIVE_DEFENSE"],
        },
    ]

    pipeline = SentimentPipeline()
    passed = 0

    print("Running sarcasm regression tests")
    print("NAME | PRIMARY | STYLES | CONF | STATUS")
    for case in test_cases:
        result = pipeline.score(str(case["text"]))
        predicted_primary = str(result.get("primary_stance_label", "NEUTRAL_COMMENTARY"))
        styles = [str(tag) for tag in result.get("style_tags", [])]
        confidence = float(result.get("primary_stance_confidence", 0.0))

        allowed = set(str(label) for label in case["expected_primary"])
        forbidden = set(str(label) for label in case.get("forbidden_primary", []))
        required_style = set(str(label) for label in case.get("required_style", []))

        ok = (predicted_primary in allowed) and (predicted_primary not in forbidden)
        if ok and required_style:
            ok = required_style.issubset(set(styles))
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"{case['name']} | {predicted_primary} | {styles} | {confidence:.4f} | {status}")

    total = len(test_cases)
    print(f"\nSarcasm tests passed: {passed}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CricGeek sentiment/stance runner")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark dataset regression")
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH), help="Path to benchmark JSON")
    parser.add_argument("--sarcasm-tests", action="store_true", help="Run sarcasm-focused regression tests")
    args = parser.parse_args()

    if args.benchmark:
        _run_benchmark(Path(args.benchmark_path))
        return

    if args.sarcasm_tests:
        _run_sarcasm_tests()
        return

    raw_text = _read_multiline_input()
    if not raw_text:
        print("No blog input received.")
        return

    blog_blocks = _split_blog_blocks(raw_text)
    if not blog_blocks:
        print("No valid blog blocks found.")
        return

    pipeline = SentimentPipeline()
    results: List[Dict[str, Any]] = []

    for idx, blog_text in enumerate(blog_blocks, start=1):
        result = pipeline.score(blog_text, enable_logs=False)
        results.append(result)
        _print_blog_result(idx, result)

    _print_summary_table(results)


if __name__ == "__main__":
    main()