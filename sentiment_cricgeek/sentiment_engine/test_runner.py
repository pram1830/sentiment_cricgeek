from __future__ import annotations

import json
import sys

from .sentiment_pipeline import SentimentPipeline


def _read_multiline_input() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    print("Paste your cricket blog text. End input with Ctrl+Z then Enter (Windows) or Ctrl+D (Unix):")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines).strip()


def main() -> None:
    blog_text = _read_multiline_input()
    if not blog_text:
        print("No blog input received.")
        return

    pipeline = SentimentPipeline()
    result = pipeline.score(blog_text, enable_logs=True)

    component_scores = result["component_scores"]
    probabilities = result["writer_type_probabilities"]

    print(f"\nDetected Writer Type: {result['writer_type']}")
    print(f"\nSentiment Score: {result['final_score']:.2f} / 100")

    print(f"\nConstructiveness: {component_scores['constructiveness']:.2f} / 40")
    print(f"Respectfulness: {component_scores['respectfulness']:.2f} / 20")
    print(f"Analytical tone: {component_scores['analytical_tone']:.2f} / 15")
    print(f"Clarity: {component_scores['clarity']:.2f} / 10")
    print(f"Fan sincerity: {component_scores['fan_sincerity']:.2f} / 15")
    print(f"Toxicity penalty: {component_scores['toxicity_penalty']:.2f}")

    print("\nWriter Type Probabilities:")
    for key, value in probabilities.items():
        if isinstance(value, (float, int)):
            print(f"- {key}: {float(value):.4f}")
        else:
            print(f"- {key}: {value}")

    print("\nParagraph Diagnostics:")
    for item in result.get("paragraph_diagnostics", []):
        print(
            f"- P{item['paragraph_index']}: tox={item['effective_toxicity']:.3f}, "
            f"neg={item['negativity']:.3f}, construct={item['paragraph_scores']['constructiveness']:.2f}, "
            f"respect={item['paragraph_scores']['respectfulness']:.2f}"
        )

    print("\nFinal Component Scores:")
    print(result["component_scores"])

    print("\nJSON output:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
