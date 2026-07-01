from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_engine.stance_detector import StanceDetector


TARGET_PER_CLASS = 150
STANCE_LABELS = [
    "SUPPORTIVE_DEFENSE",
    "CONSTRUCTIVE_CRITICISM",
    "BALANCED_DEBATE",
    "NEUTRAL_ANALYSIS",
    "DISMISSIVE_COMPLAINT",
    "DIRECT_ATTACK",
    "MIXED_STANCE",
]


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _clone_with_variation(sample: Dict[str, Any], offset: int) -> Dict[str, Any]:
    text = str(sample.get("text", "")).strip()
    # Mild synthetic perturbations for augmentation only when needed for balance.
    variants = [
        text,
        text + " This should be judged with role context.",
        text + " Selection context matters for interpretation.",
        "In this match context, " + text,
    ]
    new_text = variants[offset % len(variants)]

    return {
        "id": -1,
        "text": new_text,
        "expected_stance": sample.get("expected_stance", "NEUTRAL_ANALYSIS"),
        "expected_score_band": sample.get("expected_score_band", [45.0, 65.0]),
        "expected_toxicity_band": sample.get("expected_toxicity_band", [0.0, 0.15]),
        "expected_supportive_strength_band": sample.get("expected_supportive_strength_band", [0.0, 0.35]),
        "source": "synthetic_balance",
    }


def _rebalance(samples: List[Dict[str, Any]], target_per_class: int) -> List[Dict[str, Any]]:
    by_stance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        stance = str(sample.get("expected_stance", "NEUTRAL_ANALYSIS"))
        by_stance[stance].append(sample)

    rebalanced: List[Dict[str, Any]] = []
    random.seed(42)

    for stance in STANCE_LABELS:
        bucket = by_stance.get(stance, [])
        if not bucket:
            continue

        while len(bucket) < target_per_class:
            seed = random.choice(bucket)
            bucket.append(_clone_with_variation(seed, len(bucket)))

        if len(bucket) > target_per_class:
            random.shuffle(bucket)
            bucket = bucket[:target_per_class]

        rebalanced.extend(bucket)

    for idx, sample in enumerate(rebalanced, start=1):
        sample["id"] = idx

    return rebalanced


def _dedupe_and_strip_synthetic(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("source", "")) != "synthetic_balance"]
    seen = set()
    unique: List[Dict[str, Any]] = []
    for row in filtered:
        key = (str(row.get("expected_stance", "")), str(row.get("text", "")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _rebuild_cache(dataset_path: Path) -> None:
    detector = StanceDetector()
    rows = _load_json_list(dataset_path)
    print("Benchmark dataset size:", len(rows))
    if not rows:
        print("No rows available for cache rebuild")
        return

    # Load model once per rebuild run.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    texts = [str(item.get("text", "")) for item in rows]
    labels = [str(item.get("expected_stance", "NEUTRAL_ANALYSIS")) for item in rows]

    embeddings_list: List[np.ndarray] = []
    feature_list: List[np.ndarray] = []
    for i, text in enumerate(texts, start=1):
        if i % 50 == 0:
            print("Processing sample", i)
        emb = model.encode(text, convert_to_tensor=True).cpu().numpy()
        embeddings_list.append(emb)
        feature_list.append(detector._feature_vector(text))  # type: ignore[attr-defined]

    embeddings = np.stack(embeddings_list, axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    embeddings = embeddings / norms
    features = np.stack(feature_list, axis=0)

    detector._benchmark_ready = False  # type: ignore[attr-defined]
    detector._centroid_embeddings = {}  # type: ignore[attr-defined]
    detector._centroid_features = {}  # type: ignore[attr-defined]
    detector._benchmark_embeddings = embeddings  # type: ignore[attr-defined]
    detector._benchmark_labels = labels  # type: ignore[attr-defined]

    for stance in STANCE_LABELS:
        idxs = [idx for idx, value in enumerate(labels) if value == stance]
        if not idxs:
            continue
        emb_subset = embeddings[idxs]
        feat_subset = features[idxs]
        centroid_emb = np.mean(emb_subset, axis=0)
        centroid_norm = np.linalg.norm(centroid_emb)
        if centroid_norm > 0:
            centroid_emb = centroid_emb / centroid_norm
        detector._centroid_embeddings[stance] = centroid_emb  # type: ignore[attr-defined]
        detector._centroid_features[stance] = np.mean(feat_subset, axis=0)  # type: ignore[attr-defined]

    detector._benchmark_ready = True  # type: ignore[attr-defined]

    cache_path = dataset_path.with_suffix(".embedding_cache.npz")
    np.savez_compressed(cache_path, embeddings=embeddings, labels=np.array(labels, dtype=object))
    print(f"Saved embedding cache artifact: {cache_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build extended hybrid stance benchmark dataset")
    parser.add_argument(
        "--base",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "cricgeek_stance_benchmark_200.json"),
    )
    parser.add_argument(
        "--twitter-labeled",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "cricket_twitter_labeled_stance_samples.json"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "benchmarks" / "cricgeek_stance_benchmark_extended.json"),
    )
    parser.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--rebuild-cache", action="store_true", default=False)
    args = parser.parse_args()

    base_samples = _load_json_list(Path(args.base))
    twitter_samples = _load_json_list(Path(args.twitter_labeled))

    merged = _dedupe_and_strip_synthetic(base_samples + twitter_samples)
    if not merged:
        print("No samples available to merge.")
        return

    extended = _rebalance(merged, target_per_class=args.target_per_class)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(extended, indent=2), encoding="utf-8")
    print(f"Saved extended benchmark: {out_path} with {len(extended)} samples")

    counts: Dict[str, int] = defaultdict(int)
    for row in extended:
        counts[str(row.get("expected_stance", "NEUTRAL_ANALYSIS"))] += 1
    print("Class balance:")
    for stance in STANCE_LABELS:
        print(f"- {stance}: {counts.get(stance, 0)}")

    if args.rebuild_cache:
        _rebuild_cache(out_path)
        print("Rebuilt embedding benchmark cache")


if __name__ == "__main__":
    main()
