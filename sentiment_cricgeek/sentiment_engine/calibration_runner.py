from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, List

from .calibration_bank import CalibrationSample, get_calibration_samples
from .sentiment_pipeline import SentimentPipeline


def _midpoint(score_range):
    return (score_range[0] + score_range[1]) / 2.0


def run_calibration() -> Dict:
    samples = get_calibration_samples()
    pipeline = SentimentPipeline()

    total = len(samples)
    correct_writer = 0
    deviations_by_type: Dict[str, List[float]] = defaultdict(list)
    off_by_15: List[Dict] = []

    for sample in samples:
        result = pipeline.score(sample.text, enable_logs=False)
        predicted_type = result["writer_type"]
        score = float(result["final_score"])

        if predicted_type == sample.expected_writer_type:
            correct_writer += 1

        expected_mid = _midpoint(sample.expected_score_range)
        deviation = abs(score - expected_mid)
        deviations_by_type[sample.expected_writer_type].append(deviation)

        if deviation > 15.0:
            off_by_15.append(
                {
                    "sample_id": sample.sample_id,
                    "expected_writer_type": sample.expected_writer_type,
                    "predicted_writer_type": predicted_type,
                    "expected_range": sample.expected_score_range,
                    "predicted_score": round(score, 2),
                    "deviation": round(deviation, 2),
                }
            )

    writer_accuracy = (correct_writer / total) * 100.0 if total else 0.0
    avg_dev_by_type = {k: round(mean(v), 2) if v else 0.0 for k, v in deviations_by_type.items()}

    return {
        "total_samples": total,
        "writer_type_accuracy": round(writer_accuracy, 2),
        "average_score_deviation_by_archetype": avg_dev_by_type,
        "off_by_more_than_15": off_by_15,
        "summary": {
            "status": "needs_tuning" if off_by_15 else "calibrated",
            "message": "Calibration is acceptable." if not off_by_15 else "Some samples exceed calibration tolerance and need tuning.",
        },
    }


def main() -> None:
    report = run_calibration()

    print(f"Total calibration samples: {report['total_samples']}")
    print(f"Writer type accuracy: {report['writer_type_accuracy']:.2f}%")

    print("\nAverage score deviation by archetype:")
    for archetype, deviation in report["average_score_deviation_by_archetype"].items():
        print(f"- {archetype}: {deviation:.2f}")

    print("\nParagraphs off by more than 15 points:")
    if not report["off_by_more_than_15"]:
        print("- None")
    else:
        for item in report["off_by_more_than_15"]:
            print(
                f"- {item['sample_id']}: expected {item['expected_range']}, "
                f"got {item['predicted_score']}, deviation {item['deviation']}, "
                f"writer {item['predicted_writer_type']}"
            )

    print("\nCalibration summary:")
    print(f"- Status: {report['summary']['status']}")
    print(f"- Message: {report['summary']['message']}")


if __name__ == "__main__":
    main()
