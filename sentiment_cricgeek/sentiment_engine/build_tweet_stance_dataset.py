from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_engine.constructiveness_detector import ConstructivenessDetector
from sentiment_engine.model_loader import ModelLoader
from sentiment_engine.paragraph_splitter import split_into_paragraphs
from sentiment_engine.scoring_rules import (
    analyze_paragraphs,
    apply_adaptive_scoring,
    apply_stance_aware_weighting,
    apply_toxicity_penalty,
)
from sentiment_engine.stance_detector import StanceDetector
from sentiment_engine.writer_dna_classifier import WriterDNAClassifier


RANDOM_SEED = 42

SOURCE_FILES = [
    "TwExtract-akakrcb6-315.csv",
    "TwExtract-ashwinravi99-159.csv",
    "TwExtract-ErikaMorris79-310.csv",
    "TwExtract-Im__Arfan-308.csv",
    "TwExtract-prasannalara-304.csv",
]

CRICKET_TERMS = {
    "cricket",
    "ipl",
    "bcci",
    "icc",
    "odi",
    "t20",
    "test match",
    "world cup",
    "wicket",
    "bowler",
    "batter",
    "batsman",
    "batting",
    "bowling",
    "selection",
    "selectors",
    "team selection",
    "captain",
    "innings",
    "powerplay",
    "run chase",
    "over rate",
    "management",
    "coach",
    "rcb",
    "csk",
    "mi",
    "kkr",
    "rr",
    "srh",
    "dc",
    "pbks",
    "india",
    "pakistan",
    "australia",
    "england",
    "new zealand",
    "south africa",
    "jadeja",
    "kohli",
    "rohit",
    "rahul",
    "ashwin",
    "bumrah",
    "siraj",
    "gill",
    "jaiswal",
    "padikkal",
    "samson",
    "rinku",
    "iyer",
    "parag",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HASHTAG_TOKEN_RE = re.compile(r"^#\w+$")
MENTION_RE = re.compile(r"@\w+")


def _find_text_key(headers: List[str]) -> str:
    normalized = {h.strip().strip('"').lower(): h for h in headers}
    for key in ["text", "full_text", "tweet", "content"]:
        if key in normalized:
            return normalized[key]
    return headers[0]


def _read_csv_rows(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not headers:
            return []
        text_key = _find_text_key(headers)
        rows: List[str] = []
        for row in reader:
            value = row.get(text_key, "")
            rows.append(str(value) if value is not None else "")
        return rows


def _is_empty_text(text: str) -> bool:
    return not text.strip()


def _is_url_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    cleaned = URL_RE.sub("", stripped).strip()
    return cleaned == ""


def _is_hashtag_only(text: str) -> bool:
    tokens = [tok for tok in text.strip().split() if tok]
    if not tokens:
        return False
    return all(HASHTAG_TOKEN_RE.match(tok) is not None for tok in tokens)


def _is_emoji_only(text: str) -> bool:
    # Remove URLs/mentions/hashtags and punctuation-like separators.
    cleaned = URL_RE.sub("", text)
    cleaned = MENTION_RE.sub("", cleaned)
    cleaned = re.sub(r"#\w+", "", cleaned)
    cleaned = re.sub(r"[\s\W_]+", "", cleaned, flags=re.UNICODE)
    return cleaned == ""


def _is_retweet_without_commentary(text: str) -> bool:
    stripped = text.strip()
    return stripped.lower().startswith("rt @")


def _is_cricket_related(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in CRICKET_TERMS)


def _clean_tweets(raw_rows: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()

    for text in raw_rows:
        normalized = text.strip()
        if _is_empty_text(normalized):
            continue
        if _is_url_only(normalized):
            continue
        if _is_hashtag_only(normalized):
            continue
        if _is_emoji_only(normalized):
            continue
        if _is_retweet_without_commentary(normalized):
            continue
        if not _is_cricket_related(normalized):
            continue

        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)

    return deduped


def _estimate_bands(score: float, toxicity_mean: float, supportive_strength: float) -> Tuple[List[float], List[float], List[float]]:
    score_band = [round(max(0.0, score - 6.0), 2), round(min(100.0, score + 6.0), 2)]
    toxicity_band = [round(max(0.0, toxicity_mean - 0.08), 4), round(min(1.0, toxicity_mean + 0.08), 4)]
    supportive_band = [
        round(max(0.0, supportive_strength - 0.15), 4),
        round(min(1.0, supportive_strength + 0.15), 4),
    ]
    return score_band, toxicity_band, supportive_band


def _stratified_split(rows: List[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = str(row["stance_label"])
        by_class.setdefault(label, []).append(row)

    random.seed(seed)
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for label, bucket in by_class.items():
        random.shuffle(bucket)
        n = len(bucket)
        split_idx = max(1, int(round(n * train_ratio))) if n > 1 else 1
        split_idx = min(split_idx, n - 1) if n > 1 else 1
        train.extend(bucket[:split_idx])
        test.extend(bucket[split_idx:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "text",
        "stance_label",
        "confidence",
        "expected_score_band",
        "expected_toxicity_band",
        "expected_supportive_strength_band",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "text": row["text"],
                "stance_label": row["stance_label"],
                "confidence": row["confidence"],
                "expected_score_band": json.dumps(row["expected_score_band"]),
                "expected_toxicity_band": json.dumps(row["expected_toxicity_band"]),
                "expected_supportive_strength_band": json.dumps(row["expected_supportive_strength_band"]),
            })


def _class_distribution(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for row in rows:
        label = str(row["stance_label"])
        dist[label] = dist.get(label, 0) + 1
    return dict(sorted(dist.items(), key=lambda item: item[0]))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    tweets_dir = root / "tweets"
    output_dir = root / "sentiment_engine" / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_texts: List[str] = []
    for name in SOURCE_FILES:
        csv_path = tweets_dir / name
        if not csv_path.exists():
            continue
        raw_texts.extend(_read_csv_rows(csv_path))

    cleaned = _clean_tweets(raw_texts)
    if not cleaned:
        print("No cleaned cricket-related tweets available.")
        return

    model_loader = ModelLoader()
    models = model_loader.load()
    detector = StanceDetector()
    constructiveness_detector = ConstructivenessDetector()
    dna_classifier = WriterDNAClassifier()

    dataset_rows: List[Dict[str, Any]] = []
    for i, text in enumerate(cleaned, start=1):
        if i % 100 == 0:
            print(f"Processed {i}/{len(cleaned)} tweets")

        stance_result = detector.detect_as_dict(text=text, embedder=models.embedder)
        stance_label = str(stance_result.get("stance_label", "NEUTRAL_ANALYSIS"))
        stance_confidence = float(stance_result.get("stance_confidence", 0.0))
        supportive_defense_strength = float(stance_result.get("supportive_defense_strength", 0.0))
        criticism_reference_score = float(stance_result.get("criticism_reference_score", 0.0))
        context_change_score = float(stance_result.get("context_change_score", 0.0))
        evaluation_redirection_score = float(stance_result.get("evaluation_redirection_score", 0.0))
        credibility_restoration_score = float(stance_result.get("credibility_restoration_score", 0.0))
        contrast_structure_score = float(stance_result.get("contrast_structure_score", 0.0))
        causal_defense_score = float(stance_result.get("causal_defense_score", 0.0))
        credibility_defense_score = float(stance_result.get("credibility_defense_score", 0.0))

        paragraphs = split_into_paragraphs(text)
        paragraph_analysis = analyze_paragraphs(
            paragraphs=paragraphs,
            models=models,
            constructiveness_detector=constructiveness_detector,
        )
        stance_weighted = apply_stance_aware_weighting(paragraph_analysis, stance_result)

        dna_result = dna_classifier.classify(text, embedder=models.embedder)
        adaptive_result = apply_adaptive_scoring(
            paragraph_analysis=stance_weighted,
            writer_type=dna_result.writer_type,
            writer_type_probabilities=dna_result.writer_type_probabilities,
        )
        scored = apply_toxicity_penalty(adaptive_result, writer_type=dna_result.writer_type)

        final_score = float(scored.get("final_score", 0.0))
        toxicity_mean = float(scored.get("toxicity", {}).get("mean", 0.0))
        supportive_strength = supportive_defense_strength
        score_band, toxicity_band, supportive_band = _estimate_bands(final_score, toxicity_mean, supportive_strength)

        dataset_rows.append(
            {
                "text": text,
                "stance_label": stance_label,
                "confidence": stance_confidence,
                "expected_score_band": score_band,
                "expected_toxicity_band": toxicity_band,
                "expected_supportive_strength_band": supportive_band,
                "_debug_signals": {
                    "criticism_reference_score": criticism_reference_score,
                    "context_change_score": context_change_score,
                    "evaluation_redirection_score": evaluation_redirection_score,
                    "credibility_restoration_score": credibility_restoration_score,
                    "contrast_structure_score": contrast_structure_score,
                    "causal_defense_score": causal_defense_score,
                    "credibility_defense_score": credibility_defense_score,
                },
            }
        )

    train_rows, test_rows = _stratified_split(dataset_rows, train_ratio=0.8, seed=RANDOM_SEED)

    train_path = output_dir / "cricgeek_stance_train.csv"
    test_path = output_dir / "cricgeek_stance_test.csv"
    manifest_path = output_dir / "cricgeek_split_manifest.json"

    _write_csv(train_path, train_rows)
    _write_csv(test_path, test_rows)

    manifest = {
        "total_rows": len(dataset_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "class_distribution_train": _class_distribution(train_rows),
        "class_distribution_test": _class_distribution(test_rows),
        "random_seed_used": RANDOM_SEED,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved train split: {train_path}")
    print(f"Saved test split: {test_path}")
    print(f"Saved manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
