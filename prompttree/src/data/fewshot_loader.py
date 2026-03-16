from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from data.scipostlayouttree import SciPostLayoutTreeDataset
from data.schema import record_to_example, Example
from prompt.builder import FewshotLabels, FewshotExample


def _build_name_index(dataset: SciPostLayoutTreeDataset) -> Dict[str, int]:
    """
    SciPostLayoutTreeDataset から image_name -> index の対応を作る。

    rec["file_name"] の basename を image_name とみなす。
    """
    name_to_idx: Dict[str, int] = {}
    for idx in range(len(dataset)):
        rec = dataset[idx]
        image_name = Path(rec["file_name"]).name
        name_to_idx[image_name] = idx
    return name_to_idx


def load_fewshot_examples(
    jsonl_path: str | Path,
    dataset: SciPostLayoutTreeDataset,
) -> List[FewshotExample]:
    """
    few-shot JSONL と SciPostLayoutTreeDataset から FewshotExample のリストを作る。

    JSONL 1 行のフォーマット（make_fewshot_jsonl.py と対応）:
    {
        "id": "11066.png",
        "bboxes": [...],       # 今は使わない（Example 側から取れる）
        "reading_order": [...],
        "tree": [
            {"bbox_number": int, "parent": int},
            ...
        ]
    }
    """
    jsonl_path = Path(jsonl_path)
    name_to_idx = _build_name_index(dataset)

    fewshot_list: List[FewshotExample] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {jsonl_path}:{line_no}: {e}"
                ) from e

            image_name: str = obj["id"]
            reading_order = obj["reading_order"]
            tree = obj["tree"]  # List[Dict[str, int]]

            # Dataset から rec を引いて Example を作る
            try:
                idx = name_to_idx[image_name]
            except KeyError as e:
                raise KeyError(
                    f"{jsonl_path}:{line_no}: image_name={image_name} "
                    f"not found in dataset (split={dataset.split})"
                ) from e

            rec: Dict[str, Any] = dataset[idx]
            ex: Example = record_to_example(rec)

            labels = FewshotLabels(
                reading_order=reading_order,
                tree=tree,
            )

            fewshot_list.append(FewshotExample(example=ex, labels=labels))

    return fewshot_list
