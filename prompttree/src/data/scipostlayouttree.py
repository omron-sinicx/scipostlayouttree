import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATA_ROOT = Path("/scipostlayout/poster")

# 画像ディレクトリ
IMG_ROOT = DATA_ROOT / "png"

# アノテーション JSON のパス
SPLIT_TO_JSON: Dict[str, Path] = {
    "train": IMG_ROOT / "train_tree_ocr.json",
    "dev":   IMG_ROOT / "dev_tree_ocr.json",
    "test":  IMG_ROOT / "test_tree_ocr.json",
}

# カテゴリ関連
ROOT_CAT_ID    = 9
UNKNOWN_CAT_ID = 8
KEEP_CAT_IDS   = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 クラス


def _load_coco_json_with_rel(
    json_file: Path,
    img_root: Path,
    dataset_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    元の load_coco_json_with_rel を参考にした実装

    - COCO 形式の JSON（images / annotations）を読み込み
    - 画像ごとに annotation をまとめたレコードを作成
    - ROOT / UNKNOWN カテゴリは除外
    - category_id を 0‒7 に詰め直し
    - priority 昇順にソート
    - idx / next / parents / depth を再計算
    - OCR テキストは無視（text フィールドは触らない or 削除する）
    """
    if not json_file.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with json_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    # image_id -> record のベース
    id_to_rec: Dict[int, Dict[str, Any]] = {}
    for img in images:
        image_id = img["id"]
        file_name = img["file_name"]
        rec = {
            "file_name": str(img_root / file_name),
            "height": img.get("height", None),
            "width": img.get("width", None),
            "image_id": image_id,
            "annotations": [],
        }
        id_to_rec[image_id] = rec

    # annotation を image ごとに集約
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in id_to_rec:
            # 不整合があれば警告してスキップ
            logger.warning(
                f"annotation(image_id={img_id}) has no corresponding image in {dataset_name}"
            )
            continue
        id_to_rec[img_id]["annotations"].append(ann)

    records: List[Dict[str, Any]] = list(id_to_rec.values())

    # 各画像ごとに annotation を整形
    for rec in records:
        anns = rec["annotations"]

        kept: List[Dict[str, Any]] = []

        # --- 有効な annotation を集め、カテゴリID を詰め直す ---
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id in (ROOT_CAT_ID, UNKNOWN_CAT_ID):
                continue

            try:
                new_cat_id = KEEP_CAT_IDS.index(cat_id)
            except ValueError:
                raise ValueError(f"Unexpected category_id: {cat_id}")

            ann = dict(ann)  # 破壊的変更を避けるならコピー
            ann["category_id"] = new_cat_id
            kept.append(ann)

        if not kept:
            # 画像に有効なアノテーションが一つもない場合は例外を投げる
            raise ValueError(
                f"No valid annotations found for image_id={rec.get('image_id')}"
            )

        # --- priority 昇順で kept を並べ替え ---
        kept.sort(key=lambda a: a["priority"])

        # idx / next を付与
        id2idx: Dict[int, int] = {}
        for i, ann in enumerate(kept):
            ann["idx"] = i
            id2idx[ann["id"]] = i
            ann["next"] = i + 1 if i + 1 < len(kept) else -1

        # parents を再設定（priority順に従った idx を参照）
        for ann in kept:
            parents = ann.get("parents", [])
            if isinstance(parents, list) and parents:
                pid = parents[0]
            else:
                pid = -1
            ann["parents"] = id2idx.get(pid, -1)

        # --- depth を計算 ---
        depths = [-1] * len(kept)

        def compute_depth(i: int) -> int:
            if depths[i] >= 0:
                return depths[i]
            p = kept[i]["parents"]
            if p == -1:
                depths[i] = 2
            else:
                depths[i] = compute_depth(p) + 1
            return depths[i]

        for i in range(len(kept)):
            compute_depth(i)

        for i, ann in enumerate(kept):
            ann["depth"] = depths[i]

        rec["annotations"] = kept

    return records


class SciPostLayoutTreeDataset(Dataset):
    """
    SciPostLayoutTree 用 Dataset クラス

    - __init__(split) だけを受け取る
    - 各要素は dict:
        {
            "file_name": str,       # 画像へのフルパス
            "height": int,
            "width": int,
            "image_id": int,
            "annotations": [
                {
                    "id": int,
                    "image_id": int,
                    "category_id": int,  # 0〜7 に詰め直した ID
                    "category_name": str,
                    "priority": int,
                    "parents": int,      # 親ノードの idx（なければ -1）
                    "children": [...],   # 元 JSON による（必要ならそのまま残す）
                    "idx": int,          # priority 順に 0..N-1
                    "next": int,         # 次の idx（なければ -1）
                    "depth": int,        # 2 起点の深さ
                    "text": str,         # OCR テキスト
                },
                ...
            ],
        }
    """

    def __init__(self, split: str) -> None:
        if split not in SPLIT_TO_JSON:
            raise ValueError(
                f"Unknown split: {split}. Available: {list(SPLIT_TO_JSON.keys())}"
            )

        json_file = SPLIT_TO_JSON[split]
        img_dir = IMG_ROOT / split
        if not json_file.is_file():
            raise FileNotFoundError(f"Annotation file not found: {json_file}")
        if not img_dir.is_dir():
            logger.warning(f"Image root directory not found: {img_dir}")
        self.split = split
        self.dataset_name = f"scipostlayouttree_{split}"

        self._records = _load_coco_json_with_rel(
            json_file=json_file,
            img_root=img_dir,
            dataset_name=self.dataset_name,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._records[idx]
