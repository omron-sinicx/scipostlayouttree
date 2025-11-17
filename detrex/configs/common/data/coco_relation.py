import os

from collections import defaultdict

from transformers import AutoTokenizer

from detectron2.data.datasets.coco import load_coco_json as _load
from detectron2.data import DatasetCatalog, MetadataCatalog
import logging
logger = logging.getLogger(__name__)

ROOT_CAT_ID    = 9
UNKNOWN_CAT_ID = 8
KEEP_CAT_IDS   = [0, 1, 2, 3, 4, 5, 6, 7]       # 8 クラス
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------------
def load_coco_json_with_rel(json_file, img_root, dataset_name, tokenizer="scibert"):
    ds = _load(
        json_file, img_root, dataset_name,
        extra_annotation_keys=["id", "category_name", "priority", "parents", "children", "text"]
    )

    if tokenizer == "scibert":
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer}")

    for rec in ds:
        # --- kept に有効な annotation を集め、カテゴリID を詰め直す ---
        kept = []
        new_idx = 0
        for ann in rec["annotations"]:
            if ann["category_id"] in (ROOT_CAT_ID, UNKNOWN_CAT_ID):
                continue
            # カテゴリID を 0‒7 に詰め直す
            ann["category_id"] = KEEP_CAT_IDS.index(ann["category_id"])
            kept.append(ann)
            new_idx += 1

        # --- priority 昇順で kept を並べ替える ---
        kept.sort(key=lambda ann: ann["priority"])

        # idx / next を付与
        id2idx = {}
        for i, ann in enumerate(kept):
            ann["idx"] = i
            id2idx[ann["id"]] = i
            ann["next"] = i + 1 if i + 1 < len(kept) else -1

        # parents を再設定（priority順に従ったidxを参照）
        for ann in kept:
            pid = ann["parents"][0]
            ann["parents"] = id2idx.get(pid, -1)

        # --- depth を計算 ---
        depths = [-1] * len(kept)

        def compute_depth(i):
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

        max_length = 512
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        FIGURE_TOKEN = "[FIGURE]"
        FAILURE_TOKEN = "[OCR FAILURE]"
        for ann in kept:
            if "text" in ann:
                text = ann["text"]
            else:
                if ann["category_name"] == "Figure":
                    text = FIGURE_TOKEN
                else:
                    text = FAILURE_TOKEN
            tokens = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length, padding="max_length")
            ann["text_input_ids"] = tokens["input_ids"]
            ann["text_attention_mask"] = tokens["attention_mask"]

        assert len(kept) > 0
        rec["annotations"] = kept

    return ds

# ------------------------------------------------------------------
def register_scipost(name, json_path, img_root, tokenizer="scibert"):
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(
            name, lambda p=json_path, r=img_root, n=name, t=tokenizer: load_coco_json_with_rel(p, r, n, t)
        )
    else:
        logger.warning(f"Dataset '{name}' already registered; skip.")
