"""Convert ReVA_V2 test_set.json to a flat list for Qwen inference.

Student task: complete every block marked TODO(student).
"""

import json
import os
from collections import Counter
from typing import Any, Iterator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REVA_ROOT = os.environ.get("REVA_ROOT", os.path.join(PROJECT_ROOT, "data", "reva_test"))
REVA_JSON = os.environ.get("REVA_JSON", os.path.join(REVA_ROOT, "test_set.json"))
OUTPUT_JSON = os.environ.get("REVA_PREPARED_JSON", os.path.join(os.path.dirname(__file__), "reva_v2_test_set.json"))


def get_video_duration(video_path: str) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0.0
    except Exception as exc:
        print(f"  Warning: could not read duration for {video_path}: {exc}")
        return 0.0


def to_abs_path(file_path: str) -> str:
    """Convert an annotation video path to an absolute path under REVA_ROOT."""
    # TODO(student): remove optional dataset prefixes and join with REVA_ROOT.
    path = file_path.replace("\\", "/")

    for prefix in ("#dataset/ReVA_V2/", "#dataset/", "ReVA_V2/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break

    return os.path.abspath(os.path.join(REVA_ROOT, path))

def to_rel_stem(file_path: str) -> str:
    """Convert an annotation video path to a relative stem without extension."""
    # TODO(student): normalize the same prefixes as to_abs_path and strip extension.
    path = file_path.replace("\\", "/")

    for prefix in ("#dataset/ReVA_V2/", "#dataset/", "ReVA_V2/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break

    return os.path.splitext(path)[0]

def get_qa_id(qa: dict, video_item: dict, subcategory: str, question_idx: int) -> str:
    """Return a stable QA id for evaluation."""
    # TODO(student): prefer qa_id, then global_index, otherwise synthesize from video stem.
    if qa.get("qa_id"):
        return str(qa["qa_id"])

    if qa.get("global_index") is not None:
        return f"REVA-G{int(qa['global_index']):06d}"

    video_stem = os.path.basename(to_rel_stem(video_item["file_path"]))
    return f"REVA-{video_stem}-{subcategory[:3].upper()}-{question_idx:04d}"

def iter_flat_items(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield flat Qwen-evaluation items from nested ReVA test annotations."""
    # TODO(student): iterate videos, skip missing/unreadable videos, flatten MCQs.
    for video_info in data.get("videos", {}).values():
        file_path = video_info["file_path"]
        video_path = to_abs_path(file_path)

        if not os.path.isfile(video_path):
            continue

        duration = get_video_duration(video_path)
        if duration <= 0:
            continue

        video_id = to_rel_stem(file_path)

        for question_groups in video_info.get("mcq", {}).values():
            for question_type, qa_list in question_groups.items():
                for question_idx, qa in enumerate(qa_list):
                    options_text = "\n".join(
                        f"{label}. {text}"
                        for label, text in sorted(qa["options"].items())
                    )

                    yield {
                        "id": get_qa_id(
                            qa, video_info, question_type, question_idx
                        ),
                        "video_id": video_id,
                        "duration": duration,
                        "question": f"{qa['question']}\n{options_text}",
                        "answer": str(qa["correct_answer"]).strip().upper(),
                        "question_type": question_type,
                    }

def main():
    with open(REVA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    subcats = Counter()
    for item in iter_flat_items(data):
        items.append(item)
        subcats[item.get("question_type", "unknown")] += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(items)} QA items to {OUTPUT_JSON}")
    print("\nSubcategory distribution:")
    for subcat, count in sorted(subcats.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {subcat}: {count}")


if __name__ == "__main__":
    main()