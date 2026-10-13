#!/usr/bin/env python3
"""Score Qwen-style ReVA prediction JSONL files.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
    # TODO(student): use regex with DOTALL, strip whitespace, and return a string.
    match = re.search(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else text.strip()

def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""
    # TODO(student): normalize case and robustly parse A-H from short or verbose output.
    answer = extract_answer(text).strip()

    match = re.fullmatch(
        r"[\(\[]?([A-H])[\)\].]?",
        answer,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    match = re.search(
        r"\b(?:final\s+answer|(?:the\s+)?answer|(?:the\s+)?option|choice)\s*(?::|is|=)?\s*[\(\[]?([A-H])\b",
        answer,
        flags=re.IGNORECASE,
    )

    return match.group(1).upper() if match else ""

def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""
    # TODO(student): read every prediction *.json line-by-line and index by item["id"].
    predictions = {}

    for file_path in sorted(output_dir.glob("*.json")):
        if file_path.name == "result.json":
            continue

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)
                predictions[str(item["id"])] = item

    return predictions

def score_predictions(preds: dict[str, dict[str, Any]], gt_list: list[dict[str, Any]]) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""
    # TODO(student): compare predictions against every GT item, count missing predictions
    # as incorrect, and report Completed plus Total/subcategory accuracy statistics.
    results = {}
    completed = 0
    correct = 0
    by_type = defaultdict(lambda: [0, 0])

    for gt in gt_list:
        qa_id = str(gt["id"])
        prediction = preds.get(qa_id)

        if prediction is not None:
            completed += 1

        raw_pred = prediction.get("pred", "") if prediction else ""
        pred_letter = extract_letter(str(raw_pred or ""))
        correct_letter = str(gt["answer"]).strip().upper()
        acc = int(pred_letter == correct_letter)

        results[qa_id] = {
            **gt,
            "pred": raw_pred,
            "pred_letter": pred_letter,
            "acc": acc,
        }

        correct += acc
        question_type = gt.get("question_type", "unknown")
        by_type[question_type][0] += acc
        by_type[question_type][1] += 1

    total = len(gt_list)

    lines = [
        f"Completed: {completed}/{total} = {100 * completed / total if total else 0:.2f}%",
        f"Total: {correct}/{total} = {100 * correct / total if total else 0:.2f}%",
    ]

    for question_type, (num_correct, count) in sorted(by_type.items()):
        lines.append(
            f"{question_type}: {num_correct}/{count} = {100 * num_correct / count:.2f}%"
        )

    return results, "\n".join(lines) + "\n"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_predictions(args.output_dir)
    gt_list = json.loads(args.gt_file.read_text(encoding="utf-8"))
    results, csv_text = score_predictions(preds, gt_list)

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Results ===")
    print(csv_text)

    csv_path = args.output_dir / "result.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()