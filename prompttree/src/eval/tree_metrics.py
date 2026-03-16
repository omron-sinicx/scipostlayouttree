import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from apted import APTED, Config
import editdistance
import numpy as np


# =========================================================
# Tree Node 定義
# =========================================================

class TreeNode:
    """
    最低限 APTED と REDS に必要な情報だけを持つ TreeNode.
    """
    def __init__(self, node_id: int, label: str, category_id: int):
        self.id = node_id          # annotation の id (int)
        self.label = label         # 文字列ラベル（REDS 用。ここでは f"id_{id}"）
        self.category = category_id  # カテゴリID（TED の rename コストに使用）
        self.children: List["TreeNode"] = []


# =========================================================
# APTED 用コスト関数
# =========================================================

class TreeCostConfig(Config):
    """
    - rename: カテゴリIDが同じなら 0, 違えば 1
    - insert/delete: いずれもコスト 1
    """
    def rename(self, a: TreeNode, b: TreeNode) -> int:
        return 0 if a.label == b.label else 1

    def insert(self, node: TreeNode) -> int:
        return 1

    def delete(self, node: TreeNode) -> int:
        return 1

    def children(self, node: TreeNode):
        return node.children


# =========================================================
# COCO(拡張) JSON -> 画像ごとの木構造
#  ※ file_name をキーにする
# =========================================================

def load_coco_json(json_path: Path) -> Dict[str, dict]:
    """
    COCO(拡張)フォーマットの JSON を読み込み、
    file_name -> { "image_id", "file_name", "annotations": [...] } の dict を返す。
    """
    with json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns   = coco.get("annotations", [])

    # 一旦 image_id -> rec を作る
    imgid_to_rec: Dict[int, dict] = {}
    for img in images:
        img_id = img["id"]
        imgid_to_rec[img_id] = {
            "image_id": img_id,
            "file_name": img["file_name"],
            "height": img.get("height"),
            "width": img.get("width"),
            "annotations": [],
        }

    # 画像ごとに annotation を集約
    for ann in anns:
        img_id = ann["image_id"]
        if img_id not in imgid_to_rec:
            # 一致しない image_id は無視
            continue
        imgid_to_rec[img_id]["annotations"].append(ann)

    # file_name をキーにした dict に変換
    fname_to_rec: Dict[str, dict] = {
        rec["file_name"]: rec for rec in imgid_to_rec.values()
    }
    return fname_to_rec


def build_tree_from_record(rec: dict) -> TreeNode:
    """
    1 画像分のレコードから TreeNode の木を構築して root を返す。

    前提:
        - rec["annotations"] は、各 annotation が少なくとも:
            - id: int
            - category_id: int
            - category_name: str
            - parents: List[int] (空 or [-1] で親なし)
            - priority: int       (兄弟順の決定に使用。無ければ 0 とみなす)
        を持つ COCO拡張形式。

        - Root ノードは category_name == "Root" のノードで固定。
    """
    anns = rec["annotations"]
    if not anns:
        raise ValueError(f"No annotations found for file_name={rec['file_name']}")

    # id -> ann
    id_to_ann: Dict[int, dict] = {a["id"]: a for a in anns}

    # TreeNode のインスタンスを作成
    nodes: Dict[int, TreeNode] = {}
    for a in anns:
        nid = a["id"]
        cat_id = a.get("category_id", -1)
        label = f"id_{nid}"
        nodes[nid] = TreeNode(nid, label, cat_id)

    # 親子関係のマップを構築
    parent_map: Dict[int, int] = {}
    children_map: Dict[int, List[int]] = defaultdict(list)
    for a in anns:
        nid = a["id"]
        parents = a.get("parents", [])
        if isinstance(parents, list) and len(parents) > 0:
            pid = parents[0]
        else:
            pid = -1

        if pid == -1:
            parent_map[nid] = -1  # 親なし
        else:
            parent_map[nid] = pid
            children_map[pid].append(nid)

    # 兄弟順を priority でソート
    for pid, ch_list in children_map.items():
        ch_list.sort(key=lambda cid: id_to_ann[cid].get("priority", 0))

    # TreeNode 同士をリンク
    for pid, ch_list in children_map.items():
        pnode = nodes[pid]
        pnode.children = [nodes[cid] for cid in ch_list]

    # Root ノードを特定：
    #   category_name == "Root" のノードが必ず Root になる前提
    root_candidates = [a["id"] for a in anns if a.get("category_name") == "Root"]
    if not root_candidates:
        raise ValueError(f"No Root node (category_name=='Root') for file_name={rec['file_name']}")

    assert len(root_candidates) == 1, f"Multiple Root nodes for file_name={rec['file_name']}"

    root_id = root_candidates[0]
    root = nodes[root_id]

    return root


# =========================================================
# TED / STEDS / REDS の計算
# =========================================================

def count_nodes(root: TreeNode) -> int:
    cnt = 0
    def dfs(node: TreeNode):
        nonlocal cnt
        cnt += 1
        for ch in node.children:
            dfs(ch)
    dfs(root)
    return cnt


def flatten_tree(root: TreeNode) -> List[str]:
    """
    REDS 用に木を列に潰す。
    ここではノードの id を文字列化して並べる（読み順評価に特化）。
    カテゴリの順序を見たい場合は node.category でも良い。
    """
    out: List[str] = []
    def dfs(node: TreeNode):
        out.append(str(node.id))
        for ch in node.children:
            dfs(ch)
    dfs(root)
    return out


def compute_tree_metrics(gt_root: TreeNode, pred_root: TreeNode) -> Tuple[float, float, float]:
    """
    1 画像について TED / STEDS / REDS を計算して返す。
    """
    # TED
    ted = float(APTED(gt_root, pred_root, TreeCostConfig()).compute_edit_distance())

    # ノード数
    n_gt = count_nodes(gt_root)
    n_pred = count_nodes(pred_root)

    # STEDS: 1 - TED / max(n_gt, n_pred)
    steds = 1.0 - ted / max(n_gt, n_pred)

    # REDS
    flat_gt = flatten_tree(gt_root)
    flat_pred = flatten_tree(pred_root)
    red = editdistance.eval(flat_gt, flat_pred)
    reds = 1.0 - red / max(len(flat_gt), len(flat_pred))

    return ted, steds, reds


# =========================================================
# カテゴリID対応：GT(test_tree.json) を正として統一
# =========================================================

def build_catname_to_id(gt_recs: Dict[str, dict]) -> Dict[str, int]:
    """
    正解側（test_tree.json）の annotations から
    category_name -> category_id の対応を作る。
    """
    mapping: Dict[str, int] = {}
    for rec in gt_recs.values():
        for ann in rec["annotations"]:
            name = ann.get("category_name")
            cid  = ann.get("category_id")
            if name is None or cid is None:
                continue
            if name in mapping and mapping[name] != cid:
                raise ValueError(
                    f"Inconsistent category_id for name={name}: {mapping[name]} vs {cid}"
                )
            mapping[name] = cid
    return mapping


# =========================================================
# 全 JSON に対する評価（file_name で対応付け）
# =========================================================

def evaluate_trees(
    gt_json: Path,
    pred_json: Path,
) -> Dict[str, float]:
    """
    gt_json:        test_tree.json（正解）
    pred_json:      test_tree_pred.json（予測）

    画像の対応付けは file_name で行う。

    戻り値:
        {
          "num_images": int,
          "mean_ted": float,
          "mean_steds": float,
          "mean_reds": float,
        }
    """
    gt_recs   = load_coco_json(gt_json)    # file_name -> rec
    pred_recs = load_coco_json(pred_json)  # file_name -> rec

    # 1. 正解側から category_name -> category_id 対応を構築
    catname_to_id = build_catname_to_id(gt_recs)

    # 2. pred 側の category_id を正解側の定義に上書き
    for rec in pred_recs.values():
        for ann in rec["annotations"]:
            name = ann.get("category_name")
            if name == "Author_Info":
                name = "Author Info"
            assert name in catname_to_id, f"Unknown category_name in pred: {name}"
            ann["category_id"] = catname_to_id[name]

    # 3. file_name で対応付け
    common_files = sorted(set(gt_recs.keys()) & set(pred_recs.keys()))
    if not common_files:
        raise ValueError("No common file_name between GT and Pred.")

    ted_list   = []
    steds_list = []
    reds_list  = []

    for fname in common_files:
        gt_root   = build_tree_from_record(gt_recs[fname])
        pred_root = build_tree_from_record(pred_recs[fname])

        ted, steds, reds = compute_tree_metrics(gt_root, pred_root)

        ted_list.append(ted)
        steds_list.append(steds)
        reds_list.append(reds)

    results = {
        "num_images": len(common_files),
        "ted_list":   np.array(ted_list),
        "steds_list": np.array(steds_list),
        "reds_list":  np.array(reds_list),
        "mean_ted":   float(np.mean(ted_list)),
        "mean_steds": float(np.mean(steds_list)),
        "mean_reds":  float(np.mean(reds_list)),
    }
    return results
