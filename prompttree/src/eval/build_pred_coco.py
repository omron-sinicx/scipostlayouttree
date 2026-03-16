from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from data.schema import Example


def build_pred_coco_from_dataset(
    examples: List[Example],
    predictions: Dict[str, Dict[str, Any]],
    gt_json: Path,
    out_json: Path,
) -> None:
    """
    GT COCO を基準に、predictions に含まれるサンプルのみを対象として
    構造（parents / priority）のみを予測結果で上書きした pred COCO を生成する。

    方針:
    - images / annotations / categories は GT からコピー
    - annotations の bbox / area / segmentation 等は一切変更しない
    - children / parents / priority 等の GT 由来の構造情報は事前に除去
    - 構造は bbox_number -> annotation.id の対応を通して再構築
    """

    # --- GT COCO を丸ごと読み込む ---
    with gt_json.open("r", encoding="utf-8") as f:
        gt_coco = json.load(f)

    gt_images: List[Dict[str, Any]] = gt_coco.get("images", [])
    gt_annotations: List[Dict[str, Any]] = gt_coco.get("annotations", [])
    gt_categories: List[Dict[str, Any]] = gt_coco.get("categories", [])

    # --- pred に含まれる file_name 集合 ---
    pred_files: Set[str] = set(predictions.keys())
    if not pred_files:
        raise ValueError("predictions is empty")

    # --- GT 側の file_name -> image dict ---
    fname_to_img: Dict[str, Dict[str, Any]] = {img["file_name"]: img for img in gt_images}

    missing = sorted(pred_files - set(fname_to_img.keys()))
    if missing:
        raise ValueError(f"Predictions contain file_name not in GT: {missing}")

    selected_images: List[Dict[str, Any]] = [fname_to_img[fname] for fname in sorted(pred_files)]
    selected_image_ids: Set[int] = {img["id"] for img in selected_images}

    # --- image_id -> GT annotations ---
    imgid_to_anns: Dict[int, List[Dict[str, Any]]] = {}
    for ann in gt_annotations:
        img_id = ann["image_id"]
        if img_id in selected_image_ids:
            imgid_to_anns.setdefault(img_id, []).append(ann)

    # --- image_id -> Root annotation id ---
    imgid_to_rootid: Dict[int, int] = {}
    for img in selected_images:
        img_id = img["id"]
        anns = imgid_to_anns.get(img_id, [])
        root_ids = [a["id"] for a in anns if a.get("category_name") == "Root"]
        if len(root_ids) != 1:
            raise ValueError(
                f"GT Root node issue for file_name={img['file_name']}: {root_ids}"
            )
        imgid_to_rootid[img_id] = root_ids[0]

    # --- Example から bbox_number -> annotation.id の対応 ---
    fname_to_num_to_id: Dict[str, Dict[int, int]] = {}
    for ex in examples:
        fname = Path(ex.images.poster).name

        nums = [b.number for b in ex.bboxes]
        ids = [b.id for b in ex.bboxes]

        if len(nums) != len(set(nums)):
            raise ValueError(f"Example has duplicate bbox_number for file_name={fname}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Example has duplicate annotation.id for file_name={fname}")

        n = len(nums)
        if set(nums) != set(range(1, n + 1)):
            raise ValueError(
                f"Example bbox_number not 1..N for file_name={fname}: "
                f"n={n}, nums_sample={sorted(nums)[:10]}"
            )

        fname_to_num_to_id[fname] = {b.number: b.id for b in ex.bboxes}

    missing_ex = sorted(pred_files - set(fname_to_num_to_id.keys()))
    if missing_ex:
        raise ValueError(f"No Example found for predicted file_name(s): {missing_ex}")

    # --- GT annotations を deep copy（対象 image_id のみ） ---
    pred_annotations: List[Dict[str, Any]] = []
    for img in selected_images:
        img_id = img["id"]
        pred_annotations.extend(deepcopy(imgid_to_anns.get(img_id, [])))

    # --- GT 由来の構造情報をクリア ---
    for ann in pred_annotations:
        ann.pop("children", None)
        ann.pop("parents", None)
        ann.pop("priority", None)
        ann.pop("next", None)
        ann.pop("depth", None)

    # --- (image_id, ann_id) -> annotation ---
    key_to_ann: Dict[Tuple[int, int], Dict[str, Any]] = {
        (ann["image_id"], ann["id"]): ann for ann in pred_annotations
    }

    # --- 構造（parents / priority）を予測で上書き ---
    for fname, pred in predictions.items():
        img = fname_to_img[fname]
        image_id = img["id"]
        root_id = imgid_to_rootid[image_id]
        num_to_id = fname_to_num_to_id[fname]

        reading_order: List[int] = pred["reading_order"]
        tree_edges: List[Dict[str, int]] = pred["tree"]

        parent_num_map: Dict[int, int] = {e["bbox_number"]: e["parent"] for e in tree_edges}
        priority_map: Dict[int, int] = {bn: r for r, bn in enumerate(reading_order)}

        N = len(num_to_id)
        expected_nodes = set(range(1, N + 1))

        if set(num_to_id.keys()) != expected_nodes:
            raise ValueError(f"{fname}: Example bbox_number keys mismatch (expected 1..{N})")

        if set(reading_order) != expected_nodes:
            raise ValueError(f"{fname}: reading_order nodes mismatch with Example (N={N})")

        if set(parent_num_map.keys()) != expected_nodes:
            raise ValueError(f"{fname}: tree nodes mismatch with Example (N={N})")

        # Root
        root_key = (image_id, root_id)
        if root_key not in key_to_ann:
            raise ValueError(f"{fname}: GT Root annotation missing for root_id={root_id}")
        key_to_ann[root_key]["parents"] = []
        key_to_ann[root_key]["priority"] = -1

        # 各 bbox
        for bbox_num, ann_id in num_to_id.items():
            ann_key = (image_id, ann_id)
            if ann_key not in key_to_ann:
                raise ValueError(f"{fname}: GT annotation missing for ann_id={ann_id}")

            parent_num = parent_num_map.get(bbox_num, -1)
            if parent_num == -1:
                parent_id = root_id
            else:
                if parent_num not in num_to_id:
                    raise ValueError(f"{fname}: parent bbox_number={parent_num} not found")
                parent_id = num_to_id[parent_num]

            key_to_ann[ann_key]["parents"] = [parent_id]
            key_to_ann[ann_key]["priority"] = priority_map[bbox_num]

    # --- pred COCO 出力 ---
    pred_coco = {
        "images": deepcopy(selected_images),
        "annotations": pred_annotations,
        "categories": deepcopy(gt_categories),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(pred_coco, f, ensure_ascii=False, indent=2)
