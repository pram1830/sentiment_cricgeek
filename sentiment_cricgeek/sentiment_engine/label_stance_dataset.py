from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


STANCE_BANDS: Dict[str, Dict[str, List[float]]] = {
    "SUPPORTIVE_DEFENSE": {
        "score": [70.0, 85.0],
        "toxicity": [0.0, 0.12],
        "supportive": [0.45, 1.0],
    },
    "CONSTRUCTIVE_CRITICISM": {
        "score": [62.0, 82.0],
        "toxicity": [0.0, 0.18],
        "supportive": [0.1, 0.5],
    },
    "NEUTRAL_ANALYSIS": {
        "score": [45.0, 65.0],
        "toxicity": [0.0, 0.15],
        "supportive": [0.0, 0.35],
    },
    "BALANCED_DEBATE": {
        "score": [55.0, 75.0],
        "toxicity": [0.0, 0.22],
        "supportive": [0.05, 0.45],
    },
    "DISMISSIVE_COMPLAINT": {
        "score": [25.0, 50.0],
        "toxicity": [0.05, 0.35],
        "supportive": [0.0, 0.2],
    },
    "DIRECT_ATTACK": {
        "score": [0.0, 25.0],
        "toxicity": [0.2, 1.0],
        "supportive": [0.0, 0.1],
    },
    "MIXED_STANCE": {
        "score": [35.0, 68.0],
        "toxicity": [0.0, 0.35],
        "supportive": [0.05, 0.4],
    },
}


def _signal_score(text: str, patterns: List[str], normalizer: float) -> float:
    hits = sum(1 for p in patterns if p in text)
    return max(0.0, min(1.0, hits / normalizer))


def _label_text(text: str) -> Tuple[str, float]:
    lowered = text.lower()

    supportive = _signal_score(
        lowered,
        [
            "unfair",
            "context matters",
            "judge based on",
            "earlier performances",
            "different phase",
            "recent improvement",
            "still a quality player",
            "makes more sense to evaluate",
        ],
        3.0,
    )
    constructive = _signal_score(
        lowered,
        ["should", "could", "needs to", "improve", "fix", "adjust", "recommend"],
        3.0,
    )
    complaint = _signal_score(
        lowered,
        ["again and again", "same confusion", "repeating mistakes", "frustrating to watch", "nothing changes"],
        2.0,
    )
    attack = _signal_score(
        lowered,
        [
            "clearly not performing",
            "management ignoring problems",
            "useless",
            "trash",
            "not performing",
        ],
        2.0,
    )
    debate = _signal_score(
        lowered,
        ["some say", "others think", "on the other hand", "however", "while"],
        2.0,
    )

    if attack >= 0.5:
        return "DIRECT_ATTACK", 0.85
    if supportive >= 0.45 and complaint >= 0.25:
        return "MIXED_STANCE", 0.65
    if supportive >= 0.45:
        return "SUPPORTIVE_DEFENSE", 0.75
    if complaint >= 0.5:
        return "DISMISSIVE_COMPLAINT", 0.7
    if constructive >= 0.45:
        return "CONSTRUCTIVE_CRITICISM", 0.7
    if debate >= 0.45:
        return "BALANCED_DEBATE", 0.7
    return "NEUTRAL_ANALYSIS", 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description="Label filtered cricket tweets into benchmark stance schema")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "cricket_twitter_filtered_samples.json"),
        help="Input filtered samples json",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "cricket_twitter_labeled_stance_samples.json"),
        help="Output labeled benchmark-style json",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return

    raw = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print("Input must be a JSON list")
        return

    labeled: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        stance, confidence = _label_text(text)
        bands = STANCE_BANDS[stance]

        labeled.append(
            {
                "id": idx,
                "text": text,
                "expected_stance": stance,
                "expected_score_band": bands["score"],
                "expected_toxicity_band": bands["toxicity"],
                "expected_supportive_strength_band": bands["supportive"],
                "label_confidence": round(confidence, 4),
                "source": item.get("source", "twitter_like"),
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labeled, indent=2), encoding="utf-8")
    print(f"Saved {len(labeled)} labeled samples to {out_path}")


if __name__ == "__main__":
    main()
