from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


PLAYER_KEYWORDS = [
    "kohli",
    "rohit",
    "rahul",
    "gill",
    "jaiswal",
    "iyer",
    "rinku",
    "samson",
    "padikkal",
    "parag",
    "jurel",
    "bumrah",
    "siraj",
    "hardik",
    "jadeja",
]

SELECTION_KEYWORDS = [
    "selection",
    "selected",
    "playing xi",
    "bench",
    "drop",
    "picked",
    "team",
    "captaincy",
    "management",
]

CRITICISM_KEYWORDS = [
    "not performing",
    "struggling",
    "failed",
    "poor",
    "bad",
    "again and again",
    "repeating mistakes",
]

ROLE_KEYWORDS = [
    "role",
    "position",
    "powerplay",
    "middle overs",
    "finisher",
    "anchor",
]

FAIRNESS_DEFENSE_KEYWORDS = [
    "unfair",
    "context matters",
    "judge based on",
    "earlier performances",
    "different phase",
    "recent improvement",
    "still a quality player",
]

DEBATE_KEYWORDS = [
    "some say",
    "others think",
    "on the other hand",
    "however",
    "while",
]

TEXT_COLUMNS_CANDIDATES = ["text", "tweet", "content", "body", "full_text", "message"]


def _is_hashtag_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return bool(re.fullmatch(r"(?:#\w+\s*)+", stripped))


def _is_emoji_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    cleaned = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF\s]", "", stripped)
    return cleaned == ""


def _is_retweet_without_commentary(text: str) -> bool:
    lowered = text.lower().strip()
    return lowered.startswith("rt @") and ":" not in lowered


def _contains_topic_keywords(text: str) -> bool:
    lowered = text.lower()
    groups = [
        PLAYER_KEYWORDS,
        SELECTION_KEYWORDS,
        CRITICISM_KEYWORDS,
        ROLE_KEYWORDS,
        FAIRNESS_DEFENSE_KEYWORDS,
        DEBATE_KEYWORDS,
    ]
    return any(any(token in lowered for token in group) for group in groups)


def _normalize_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _row_to_text(row: Dict[str, str], forced_column: str | None = None) -> str:
    if forced_column and forced_column in row:
        return str(row.get(forced_column, "") or "")
    for col in TEXT_COLUMNS_CANDIDATES:
        if col in row and row[col]:
            return str(row[col])
    return ""


def _load_local_csv(path: Path, text_column: str | None = None) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield _row_to_text(row, forced_column=text_column)


def _load_hf_dataset(name: str, split: str, text_column: str | None = None) -> Iterable[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return []

    dataset = load_dataset(name, split=split)
    for row in dataset:
        if text_column and text_column in row:
            yield str(row.get(text_column, "") or "")
        else:
            for col in TEXT_COLUMNS_CANDIDATES:
                if col in row and row[col]:
                    yield str(row[col])
                    break


def collect_samples(
    input_csv: Path | None,
    hf_dataset: str | None,
    hf_split: str,
    text_column: str | None,
) -> List[Dict[str, str]]:
    raw_texts: List[str] = []

    if input_csv is not None and input_csv.exists():
        raw_texts.extend(_load_local_csv(input_csv, text_column=text_column))
    elif hf_dataset:
        raw_texts.extend(_load_hf_dataset(hf_dataset, hf_split, text_column=text_column))

    seen = set()
    collected: List[Dict[str, str]] = []

    for text in raw_texts:
        normalized = _normalize_text(str(text))
        if not normalized:
            continue
        if _is_hashtag_only(normalized):
            continue
        if _is_emoji_only(normalized):
            continue
        if _is_retweet_without_commentary(normalized):
            continue
        if not _contains_topic_keywords(normalized):
            continue

        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)

        collected.append({"text": normalized, "source": "twitter_like"})

    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and filter cricket twitter-like samples")
    parser.add_argument("--input-csv", default="", help="Path to local csv file")
    parser.add_argument("--hf-dataset", default="", help="Optional HuggingFace dataset name")
    parser.add_argument("--hf-split", default="train", help="Dataset split if using --hf-dataset")
    parser.add_argument("--text-column", default="", help="Optional explicit text column")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "cricket_twitter_filtered_samples.json"),
        help="Output json path",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv) if args.input_csv else None
    text_column = args.text_column or None
    hf_dataset = args.hf_dataset or None

    samples = collect_samples(
        input_csv=input_csv,
        hf_dataset=hf_dataset,
        hf_split=args.hf_split,
        text_column=text_column,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} filtered samples to {out_path}")


if __name__ == "__main__":
    main()
