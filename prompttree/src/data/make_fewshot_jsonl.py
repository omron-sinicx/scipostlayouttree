import json
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

from src.data.scipostlayouttree import SciPostLayoutTreeDataset
from src.data.schema import record_to_example, Example


def rec_to_fewshot_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    SciPostLayoutTreeDataset の 1 レコードから、
    few-shot 用の 1 行 JSON に変換する。

    - bboxes: モデル入力に使う矩形情報
    - reading_order: priority ベースの正解読み順（bbox_number 列）
    - tree: 親子関係の正解（bbox_number 同士で表現）
    """

    ex: Example = record_to_example(rec)

    anns = rec["annotations"]
    id_to_idx = {ann["id"]: ann["idx"] for ann in anns}

    # id ↔ number
    id_to_number = {b.id: b.number for b in ex.bboxes}
    idx_to_number = {
        id_to_idx[b_id]: num for b_id, num in id_to_number.items()
    }

    # 読み順（idx = priority順）
    anns_by_idx = sorted(anns, key=lambda a: a["idx"])
    reading_order: List[int] = [
        idx_to_number[a["idx"]] for a in anns_by_idx
    ]

    # 親子関係
    tree: List[Dict[str, int]] = []
    for ann in anns:
        child_idx = ann["idx"]
        child_num = idx_to_number[child_idx]

        parent_idx = ann["parents"]
        if parent_idx == -1:
            parent_num = -1
        else:
            parent_num = idx_to_number[parent_idx]

        tree.append(
            {
                "bbox_number": child_num,
                "parent": parent_num,
            }
        )

    # 入力側の矩形情報
    bboxes_payload = [
        {
            "number": b.number,
            "category": b.category,
            "x": b.x,
            "y": b.y,
            "w": b.w,
            "h": b.h,
            "x_norm": b.x_norm,
            "y_norm": b.y_norm,
            "w_norm": b.w_norm,
            "h_norm": b.h_norm,
        }
        for b in ex.bboxes
    ]

    row = {
        "id": ex.image_name,
        "bboxes": bboxes_payload,
        "reading_order": reading_order,
        "tree": tree,
    }
    return row


def dump_split(split: str, out_path: Path) -> None:
    ds = SciPostLayoutTreeDataset(split=split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in tqdm(range(len(ds)), desc=f"dump {split}"):
            rec = ds[i]
            row = rec_to_fewshot_row(rec)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    out_dir = Path("./fewshot_jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        out_path = out_dir / f"scipost_{split}_fewshot.jsonl"
        dump_split(split, out_path)
        print(f"[{split}] written to {out_path}")


if __name__ == "__main__":
    main()
