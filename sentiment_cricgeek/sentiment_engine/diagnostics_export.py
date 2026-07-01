from __future__ import annotations

import csv
from typing import Any, Dict, List


CSV_COLUMNS = [
    "sample_id",
    "text",
    "expected_writer_type",
    "predicted_writer_type",
    "expected_constructive",
    "expected_toxic",
    "expected_score_range",
    "predicted_final_score",
    "constructiveness_signal",
    "reasoning_marker_score",
    "suggestion_score",
    "explanation_depth_score",
    "toxicity_score",
    "toxicity_penalty",
    "writer_confidence",
    "paragraph_count",
    "word_count",
    "deviation_from_expected_midpoint",
]


def export_diagnostics_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})
