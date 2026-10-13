import argparse
import copy
import json
import os
import re
from pathlib import Path
from time import strftime
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=".data/ReVA_V2/valid_set.json")
    parser.add_argument("--dataset-root", type=str, default=".data")
    parser.add_argument("--dataset-prefix", type=str, default="#dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--video-max-tiles", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timestamp-output", action="store_true")
    return parser.parse_args()


def load_instances(question_file: str) -> list[dict[str, Any]]:
    """Flatten nested ReVA annotations into VILA evaluation instances."""
    # TODO(student): read question_file, flatten instances, and assign stable IDs
    # using qa_id, then global_index, then video/subcategory/question index.
    with open(question_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = []

    for video_id, video_info in data.get("videos", {}).items():
        video_path = video_info["file_path"]

        for category, question_groups in video_info.get("mcq", {}).items():
            for subcategory, qa_list in question_groups.items():
                for question_idx, qa in enumerate(qa_list):
                    if qa.get("qa_id"):
                        qa_id = str(qa["qa_id"])
                    elif qa.get("global_index") is not None:
                        qa_id = f"REVA-G{int(qa['global_index']):06d}"
                    else:
                        video_stem = Path(
                            video_path.replace("\\", "/")
                        ).stem
                        qa_id = (
                            f"REVA-{video_stem}-"
                            f"{subcategory[:3].upper()}-{question_idx:04d}"
                        )

                    instances.append({
                        "qa_id": qa_id,
                        "video_id": video_id,
                        "video_path": video_path,
                        "dataset_name": video_info.get("dataset_name"),
                        "category": category,
                        "subcategory": subcategory,
                        "question": qa["question"],
                        "options": qa["options"],
                        "correct_answer": str(
                            qa["correct_answer"]
                        ).strip().upper(),
                        "reasoning": qa.get("reasoning", ""),
                        "example": qa.get("example", ""),
                    })

    return instances

def resolve_video_path(raw_path: str, dataset_root: str, dataset_prefix: str) -> str:
    """Resolve a ReVA annotation path to an existing local video path."""
    # TODO(student): try raw_path, dataset_root/raw_path, and paths with dataset_prefix removed.
    path = raw_path.replace("\\", "/")
    root = Path(dataset_root)

    candidates = [Path(path), root / path]

    prefix = dataset_prefix.replace("\\", "/").rstrip("/")
    if prefix and path.startswith(prefix + "/"):
        path = path[len(prefix) + 1:]
        candidates.append(root / path)

    if path.startswith("ReVA_V2/"):
        candidates.append(root / path[len("ReVA_V2/"):])

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    raise FileNotFoundError(f"Video not found: {raw_path}")

def build_prompt(question: str, options: dict[str, str]) -> str:
    """Build the VILA multiple-choice prompt."""
    # TODO(student): include question, sorted options, and one-letter instruction.
    options_text = "\n".join(
        f"{label}. {text}"
        for label, text in sorted(options.items())
    )

    return (
        f"{question}\n"
        f"{options_text}\n"
        "Answer with only the option letter from the given choices."
    )

def parse_choice(response: str, options: dict[str, str]) -> str | None:
    """Parse a choice letter from VILA raw response."""
    # TODO(student): robustly parse A/B/C/D from raw output or matching option text.
    text = response.strip()
    valid_labels = {label.upper() for label in options}

    match = re.search(
        r"<answer>\s*([A-H])\s*</answer>",
        text,
        flags=re.IGNORECASE,
    )
    if match and match.group(1).upper() in valid_labels:
        return match.group(1).upper()

    match = re.fullmatch(
        r"[\(\[]?([A-H])[\)\].]?",
        text,
        flags=re.IGNORECASE,
    )
    if match and match.group(1).upper() in valid_labels:
        return match.group(1).upper()

    match = re.search(
        r"\b(?:final\s+answer|answer|option|choice)\s*(?::|is|=)?\s*[\(\[]?([A-H])\b",
        text,
        flags=re.IGNORECASE,
    )
    if match and match.group(1).upper() in valid_labels:
        return match.group(1).upper()

    matches = [
        label.upper()
        for label, option_text in options.items()
        if option_text.strip()
        and re.search(
            r"(?<!\w)" + re.escape(option_text.strip()) + r"(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
    ]

    return matches[0] if len(matches) == 1 else None

def load_existing_predictions(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    predictions = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[record["qa_id"]] = record
    return predictions


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> dict:
    """Compute total, category, and subcategory accuracy for VILA outputs."""
    # TODO(student): compute num_questions, num_answered, accuracy, by_category, by_subcategory.
    def new_stats():
        return {"num_questions": 0, "num_answered": 0, "num_correct": 0}

    overall = new_stats()
    by_category = {}
    by_subcategory = {}

    for record in records:
        category = record.get("category", "unknown")
        subcategory = record.get("subcategory", "unknown")
        answered = bool(record.get("pred_letter"))
        correct = bool(record.get("is_correct"))

        for stats in (
            overall,
            by_category.setdefault(category, new_stats()),
            by_subcategory.setdefault(subcategory, new_stats()),
        ):
            stats["num_questions"] += 1
            stats["num_answered"] += int(answered)
            stats["num_correct"] += int(correct)

    def finalize(stats):
        total = stats["num_questions"]
        return {
            **stats,
            "accuracy": stats["num_correct"] / total if total else 0.0,
        }

    return {
        **finalize(overall),
        "by_category": {
            category: finalize(stats)
            for category, stats in sorted(by_category.items())
        },
        "by_subcategory": {
            subcategory: finalize(stats)
            for subcategory, stats in sorted(by_subcategory.items())
        },
    }

def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import llava
    from llava import Video
    from llava import conversation as conversation_lib

    instances = load_instances(args.question_file)
    if args.max_questions is not None:
        instances = instances[: args.max_questions]

    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir if args.resume or not args.timestamp_output else base_output_dir / strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to: {output_dir}")

    outputs_path = output_dir / "outputs.jsonl"
    metrics_path = output_dir / "metrics.json"
    existing = load_existing_predictions(str(outputs_path)) if args.resume else {}

    if args.conv_mode != "auto":
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    model = llava.load(args.model_path, model_base=args.model_base)
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames
    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    generation_config = copy.deepcopy(model.default_generation_config)
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    outputs = []
    for instance in tqdm(instances):
        if instance["qa_id"] in existing:
            outputs.append(existing[instance["qa_id"]])
            continue

        video_path = resolve_video_path(instance["video_path"], args.dataset_root, args.dataset_prefix)
        prompt = build_prompt(instance["question"], instance["options"])
        response = model.generate_content([Video(video_path), prompt], generation_config=generation_config)
        pred_letter = parse_choice(response, instance["options"])

        outputs.append({
            "qa_id": instance["qa_id"],
            "video_id": instance["video_id"],
            "video_path": video_path,
            "dataset_name": instance.get("dataset_name"),
            "category": instance["category"],
            "subcategory": instance["subcategory"],
            "question": instance["question"],
            "options": instance["options"],
            "prompt": prompt,
            "raw_response": response,
            "pred_letter": pred_letter,
            "correct_answer": instance["correct_answer"],
            "is_correct": pred_letter == instance["correct_answer"],
            "reasoning": instance.get("reasoning", ""),
            "example": instance.get("example", ""),
        })
        save_jsonl(str(outputs_path), outputs)

    save_jsonl(str(outputs_path), outputs)
    save_json(str(metrics_path), summarize(outputs))


if __name__ == "__main__":
    main()