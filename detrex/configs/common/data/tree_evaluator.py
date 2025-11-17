import os
import json
import itertools
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F

from detectron2.utils import comm
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import pairwise_iou
from scipy.optimize import linear_sum_assignment
from apted import APTED, Config
import editdistance
import heapq

# Absolute import to avoid import conflicts caused by LazyConfig's dynamic loading
from configs.common.data.tree_node import TreeNode


def print_tree(node, indent=0):
    prefix = "  " * indent
    print(f"{prefix}- Label: {node.label}, Category: {node.category}")
    for child in node.children:
        print_tree(child, indent + 1)


# ----------------------------------------------------------------------
# APTED cost
# ----------------------------------------------------------------------
class _CostConfig(Config):
    def rename(self, a, b):
        return 0 if a.label == b.label else 1

    def insert(self, node):
        return 1

    def delete(self, node):
        return 1

    def children(self, node):
        return node.children


# ----------------------------------------------------------------------
# GT : parent & priority → tree
# ----------------------------------------------------------------------
def build_gt_tree(gt):
    parents  = gt.gt_parents.cpu().numpy() + 1
    N = len(gt)
    labels = [f"gt_{i}" for i in range(N)]

    nodes = [TreeNode(i, labels[i], gt.gt_boxes.tensor[i].numpy(), gt.gt_classes[i].item()) for i in range(N)]
    root = TreeNode("root", "root", np.array([10, 10, 10, 10]), -1)

    children_map = defaultdict(list)
    for i, p in enumerate(parents):
        children_map[p].append(i+1)

    def attach(parent_node):
        parent_id = 0 if parent_node.id == "root" else parent_node.id + 1
        parent_node.children = [nodes[i-1] for i in children_map[parent_id]]
        for ch in parent_node.children:
            attach(ch)

    attach(root)
    return root


# ----------------------------------------------------------------------
# Beam search decoding
# ----------------------------------------------------------------------
def decode_pred_tree(parent_logits, next_logits, root_next_logits, bboxes, categories, beam_width=1):
    N = parent_logits.shape[0]

    beam_width = min(N, beam_width)

    logits = F.log_softmax(torch.tensor(next_logits), dim=-1)
    root_logits = F.log_softmax(torch.tensor(root_next_logits), dim=-1)

    topv, topi = root_logits.topk(beam_width)
    beams = []

    for i in range(beam_width):
        next_node = topi[i].item() + 1  # 1-based offset
        score = topv[i].item()
        used = [False] * (N + 1)
        used[0] = True
        used[next_node] = True
        beams.append((score, [0, next_node], used))

    for _ in range(N - 1):
        new_beams = []
        for score, path, used in beams:
            last_node = path[-1]
            next_logits = logits[last_node - 1]  # index shift (1-based)
            topv, topi = next_logits.topk(N + 1)

            candidates = 0
            for j in range(N + 1):
                candidate = topi[j].item()
                if not used[candidate]:
                    new_used = used[:]
                    new_used[candidate] = True
                    new_path = path + [candidate]
                    new_score = score + topv[j].item()
                    new_beams.append((new_score, new_path, new_used))
                    candidates += 1
                    if candidates >= beam_width:
                        break

        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:beam_width]

    best_score, best_path, best_used = beams[0]
    assert sum(best_used) == N + 1
    assert len(best_path) == N + 1

    reading_order = best_path

    log_parent_logits = F.log_softmax(torch.tensor(parent_logits), dim=1)  # shape: (N, N+1)

    init_beam = {
        "score": 0.0,
        "step": 1,
        "children_map": defaultdict(list),
        "path": {reading_order[0]: None},
    }
    beams = [init_beam]

    for step in range(1, N + 1):
        current_node = reading_order[step]
        new_beams = []

        for beam in beams:
            allow_node_list = [reading_order[0]]
            cnode = reading_order[0]
            while beam["children_map"].get(cnode):
                cnode = beam["children_map"][cnode][-1]
                allow_node_list.append(cnode)

            # allow_node_list = reading_order[:step]

            log_probs = log_parent_logits[current_node - 1]  # shape: (N+1,)

            for parent_node in allow_node_list:
                parent_logp = log_probs[parent_node].item()

                new_path = beam["path"].copy()
                new_path[current_node] = parent_node

                new_cmap = defaultdict(list, {k: v[:] for k, v in beam["children_map"].items()})
                new_cmap[parent_node].append(current_node)

                new_beams.append({
                    "score": beam["score"] + parent_logp,
                    "step": step + 1,
                    "children_map": new_cmap,
                    "path": new_path,
                })

        new_beams.sort(key=lambda b: -b["score"])
        beams = new_beams[:beam_width]

    best_beam = beams[0]
    parent_map = best_beam["path"]
    children_map = best_beam["children_map"]
    best_score = best_beam["score"]

    labels = [f"pred_{i}" for i in range(N)]
    nodes  = [TreeNode(i, labels[i], bboxes[i], categories[i]) for i in range(N)]
    root   = TreeNode("root", "root", np.array([10, 10, 10, 10]), -1)
    # new_reading_order = []

    def attach(parent_id, parent_node):
        # new_reading_order.append(parent_id)
        for cid in children_map.get(parent_id, []):
            child_node = nodes[cid-1]
            parent_node.children.append(child_node)
            attach(cid, child_node)

    attach(0, root)
    # reading_order = new_reading_order

    return reading_order, children_map, root


def check_tree(root_node, N):
    if root_node.label != "root":
        return False

    seen = [False] * N
    stack = list(root_node.children)

    while stack:
        node = stack.pop()
        label = node.label
        num = int(label.split("_")[1])

        if num < 0 or num >= N or seen[num] == True:
            return False
        seen[num] = True

        stack.extend(node.children)

    return len(seen) == N


# ----------------------------------------------------------------------
# TreeEvaluator
# ----------------------------------------------------------------------
class TreeEvaluator(DatasetEvaluator):
    def __init__(self, output_dir=None, iou_threshold=0.75, beam_width=1):
        self._distributed = comm.get_world_size() > 1
        self.output_dir = Path(output_dir) if output_dir else None
        self.iou_threshold = iou_threshold
        self.beam_width = beam_width
        self.reset()

    # ---------- D2 API ----------
    def reset(self):
        self._records = []

    @torch.no_grad()
    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            gt   = inp["instances"].to("cpu")
            pred = out["instances"].to("cpu")

            ted, nteds, steds, reds, sim, gt_tree, pred_tree = self._tree_distance_single(gt, pred)

            self._records.append(
                dict(
                    file_name = os.path.basename(inp["file_name"]),
                    image_id  = inp["image_id"],
                    ted       = ted,
                    nteds      = nteds,
                    steds     = steds,
                    reds      = reds,
                    sim       = sim,
                    gt        = gt,
                    pred      = pred,
                    gt_tree   = gt_tree,
                    pred_tree = pred_tree,
                )
            )

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            records = comm.gather(self._records, dst=0)
            records = list(itertools.chain(*records))

            if not comm.is_main_process():
                return {}
        else:
            records = self._records

        ted = [r["ted"] for r in records]
        nteds = [r["nteds"] for r in records]
        steds = [r["steds"] for r in records]
        reds = [r["reds"] for r in records]
        sim = [r["sim"] for r in records]

        results = {
            "tree/num_samples": len(ted),
            "tree/ted": float(np.mean(ted)),
            "tree/nteds": float(np.mean(nteds)),
            "tree/steds": float(np.mean(steds)),
            "tree/reds": float(np.mean(reds)),
            "tree/sim":  float(np.mean(sim)),
        }

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(records, os.path.join(self.output_dir, "tree_predictions.pt"))
            with open(os.path.join(self.output_dir, "tree_metrics.json"), "w") as f:
                json.dump(results, f, indent=2)

        return results

    # ---------- 1. GT↔Pred ----------
    @staticmethod
    def _hungarian_match(gt_boxes, gt_cls, pred_boxes, pred_cls, iou_thr):
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return {}, {}

        iou_mat = pairwise_iou(gt_boxes, pred_boxes)          # (G, P)
        cls_eq  = (gt_cls.view(-1, 1) == pred_cls.view(1, -1))
        valid   = cls_eq & (iou_mat >= iou_thr)

        cost = np.where(valid.numpy(), 1-iou_mat.numpy(), 1e6)
        row, col = linear_sum_assignment(cost)

        gt2pred, pred2gt = {}, {}
        for g, p in zip(row, col):
            if cost[g, p] < 1e6:
                gt2pred[int(g)] = int(p)
                pred2gt[int(p)] = int(g)
        return gt2pred, pred2gt

    @staticmethod
    def _compute_reds(gt_tree, pred_tree, n_gt, n_pred):
        def flatten_tree(root):
            result = []
            def dfs(node):
                result.append(node.label)
                for child in node.children:
                    dfs(child)
            dfs(root)
            return result

        flat_gt = flatten_tree(gt_tree)
        flat_pred = flatten_tree(pred_tree)

        red = editdistance.eval(flat_gt, flat_pred)
        # reds = 1 - red / (n_gt+n_pred)
        # Original REDS definition
        reds = 1 - red / max(n_gt, n_pred)

        return reds

    # ---------- 2. TED ----------
    def _tree_distance_single(self, gt, pred):
        gt_tree = build_gt_tree(gt)
        n_gt = len(gt)

        pos_indices = pred.scores > 0.5
        n_pred = pos_indices.sum().item()
        if n_pred > 0:
            h_gt, w_gt = gt.image_size
            h_pred, w_pred = pred.image_size
            scale_y = h_gt / h_pred
            scale_x = w_gt / w_pred
            
            pred_boxes = pred.pred_boxes[pos_indices].clone()
            pred_boxes.tensor *= torch.tensor([scale_x, scale_y, scale_x, scale_y])
            pred_classes = pred.pred_classes[pos_indices].clone()

            gt2pred, pred2gt = self._hungarian_match(
                gt.gt_boxes, gt.gt_classes,
                pred_boxes, pred_classes,
                self.iou_threshold,
            )

            parent_logits    = pred.pred_parent_logits[pos_indices].numpy()
            next_logits      = pred.pred_next_logits[pos_indices].numpy()
            root_next_logits = pred.pred_root_next_logits[pos_indices].numpy()

            #######################################################
            # Debug
            #######################################################
            # parent_logits *= 0.0
            # next_logits *= 0.0
            # root_next_logits *= 0.0
            # gt_parents = gt.gt_parents + 1
            # gt_next = gt.gt_next + 1
            # for g in range(len(gt_parents)):
            #     parent_logits[g][gt_parents[g]] = 1.0
            #     next_logits[g][gt_next[g]] = 1.0
            #     root_next_logits[0] = 1.0

            #######################################################
            # Debug
            #######################################################
            # # ハンガリアンマッチングした BBox 同士の予測が完璧な場合の upper bound
            # parent_logits[:] = 0.0
            # next_logits[:] = 0.0
            # root_next_logits[:] = 0.0

            # # +1 シフトされた GT parent/next
            # gt_parents = (gt.gt_parents + 1).tolist()
            # gt_nexts = (gt.gt_next + 1).tolist()

            # # --- parent_logits の上書き ---
            # for pred_idx in range(n_pred):
            #     if pred_idx in pred2gt:
            #         gt_idx = pred2gt[pred_idx]

            #         # 親ノード（GT idx）をたどる
            #         parent_gt_idx = gt_parents[gt_idx] - 1  # -1 で元の index
            #         while parent_gt_idx != -1 and parent_gt_idx not in gt2pred:
            #             parent_gt_idx = gt_parents[parent_gt_idx] - 1
            #         if parent_gt_idx == -1:
            #             parent_pred_idx = 0  # root
            #         else:
            #             parent_pred_idx = gt2pred[parent_gt_idx] + 1  # +1 shift for root

            #         parent_logits[pred_idx][parent_pred_idx] = 1.0

            #     else:
            #         # 過剰検出：親は root
            #         parent_logits[pred_idx][0] = 1.0

            # # --- next_logits の上書き ---
            # # GT の next 構造に従って、マッチ済み pred index を順に並べる
            # reading_order = []
            # for g in range(n_gt):
            #     if g in gt2pred:
            #         pred_idx = gt2pred[g]
            #         reading_order.append(pred_idx)

            # # 順序に基づき next をつなぐ
            # if reading_order:
            #     root_next_logits[reading_order[0]] = 1.0
            #     for i in range(len(reading_order) - 1):
            #         a = reading_order[i]
            #         b = reading_order[i + 1]
            #         next_logits[a][b + 1] = 1.0  # +1 for root offset
            #######################################################
            # Debug
            #######################################################

            reading_order, children_map, pred_tree = decode_pred_tree(
                parent_logits    = parent_logits,
                next_logits      = next_logits,
                root_next_logits = root_next_logits,
                bboxes           = pred_boxes.tensor.numpy(),
                categories       = pred_classes.numpy(),
                beam_width       = self.beam_width,
            )

            assert check_tree(gt_tree, n_gt)
            assert check_tree(pred_tree, n_pred)

            def relabel(tree):
                if tree.label.startswith("pred_"):
                    label = tree.label
                    idx = int(label.split("_")[1])
                    if idx in pred2gt:
                        tree.label = f"gt_{pred2gt[idx]}"
                for ch in tree.children:
                    relabel(ch)
            relabel(pred_tree)
        else:
            pred_tree = TreeNode("root", "root", np.array([10, 10, 10, 10]), -1)

        # --- TED, STEDS, REDS ---
        ted = APTED(gt_tree, pred_tree, _CostConfig()).compute_edit_distance()
        ted = float(ted)
        n_gt += 1 # root
        n_pred += 1  # root
        nteds = 1 - ted / (n_gt+n_pred)
        steds = 1 - ted / max(n_gt, n_pred)
        reds = self._compute_reds(gt_tree, pred_tree, n_gt, n_pred)
        sim = 1.0 - (2*ted) / (n_gt+n_pred+ted)

        return ted, nteds, steds, reds, sim, gt_tree, pred_tree
