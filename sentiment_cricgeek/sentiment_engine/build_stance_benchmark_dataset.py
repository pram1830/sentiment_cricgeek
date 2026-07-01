from __future__ import annotations

import json
import random
from pathlib import Path


random.seed(42)

COUNTS = {
    "SUPPORTIVE_DEFENSE": 30,
    "CONSTRUCTIVE_CRITICISM": 30,
    "NEUTRAL_ANALYSIS": 28,
    "BALANCED_DEBATE": 28,
    "DISMISSIVE_COMPLAINT": 28,
    "DIRECT_ATTACK": 28,
    "MIXED_STANCE": 28,
}

BANDS = {
    "SUPPORTIVE_DEFENSE": {"score": [70, 85], "toxicity": [0.0, 0.12], "supportive": [0.45, 1.0]},
    "CONSTRUCTIVE_CRITICISM": {"score": [62, 82], "toxicity": [0.0, 0.18], "supportive": [0.10, 0.50]},
    "NEUTRAL_ANALYSIS": {"score": [45, 65], "toxicity": [0.0, 0.15], "supportive": [0.00, 0.35]},
    "BALANCED_DEBATE": {"score": [55, 75], "toxicity": [0.0, 0.22], "supportive": [0.05, 0.45]},
    "DISMISSIVE_COMPLAINT": {"score": [25, 50], "toxicity": [0.05, 0.35], "supportive": [0.00, 0.20]},
    "DIRECT_ATTACK": {"score": [0, 25], "toxicity": [0.20, 1.0], "supportive": [0.00, 0.10]},
    "MIXED_STANCE": {"score": [35, 68], "toxicity": [0.0, 0.35], "supportive": [0.05, 0.40]},
}

PLAYERS = ["Padikkal", "Samson", "Rinku", "Gill", "Parag", "Jurel", "Iyer", "Jaiswal"]
ROLES = ["finisher", "anchor", "powerplay batter", "middle-order stabilizer", "left-hander in tough phases"]
CONTEXTS = ["new batting role", "different phase of career", "clearer team balance", "improved support from non-striker", "better matchups"]
METRICS = ["strike rate", "dot-ball rate", "boundary percentage", "risk profile"]


def build_sample(stance: str) -> str:
    player = random.choice(PLAYERS)
    role = random.choice(ROLES)
    context = random.choice(CONTEXTS)
    metric = random.choice(METRICS)

    if stance == "SUPPORTIVE_DEFENSE":
        return (
            f"Many people still judge {player} based on earlier performances, but the {context} changed his output. "
            f"It makes more sense to evaluate current progress instead of old {metric} numbers. "
            f"He is still a quality player when used as a {role}."
        )
    if stance == "CONSTRUCTIVE_CRITICISM":
        return (
            f"{player} could improve against pace in overs 7-12, and the team should adjust field-aware strike rotation. "
            f"A clearer plan for his {role} would reduce pressure, and management needs to align matchups better."
        )
    if stance == "NEUTRAL_ANALYSIS":
        return (
            f"{player} faced mixed conditions and his output tracked expected variance for a {role}. "
            f"The innings had moderate control and average acceleration, with no strong evidence for a strategic shift."
        )
    if stance == "BALANCED_DEBATE":
        return (
            f"Some analysts argue {player} slows the middle overs, while others note he stabilizes collapse risk. "
            f"Both views hold under different conditions, so role-context should guide selection decisions."
        )
    if stance == "DISMISSIVE_COMPLAINT":
        return (
            f"Again and again we see the same confusion around {player}. Repeating mistakes keep happening and it is frustrating to watch. "
            "Nothing really changes in team calls."
        )
    if stance == "DIRECT_ATTACK":
        return (
            f"{player} is clearly not performing and management is ignoring problems by keeping players who are not performing. "
            "This is useless selection and should stop now."
        )
    return (
        f"{player} looked uncomfortable early and some criticism is valid, but there are signs of adaptation in the {context}. "
        "He should improve shot selection, yet dropping him immediately may be premature."
    )


def main() -> None:
    samples = []
    for stance, count in COUNTS.items():
        for _ in range(count):
            band = BANDS[stance]
            samples.append(
                {
                    "id": len(samples) + 1,
                    "text": build_sample(stance),
                    "expected_stance": stance,
                    "expected_score_band": band["score"],
                    "expected_toxicity_band": band["toxicity"],
                    "expected_supportive_strength_band": band["supportive"],
                }
            )

    out_path = Path(__file__).resolve().parent / "benchmarks" / "cricgeek_stance_benchmark_200.json"
    out_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
