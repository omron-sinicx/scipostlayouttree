import os
import json
import itertools
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment

from detectron2.utils import comm
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import pairwise_iou, Instances
from detectron2.utils.comm import gather, is_main_process


class RelationEvaluator(DatasetEvaluator):
    def __init__(self, output_dir=None, iou_threshold=0.75):
        self._distributed = comm.get_world_size() > 1
        self.output_dir = Path(output_dir) if output_dir else None
        self.iou_threshold = iou_threshold
        self.reset()

    # ----------------------------
    def reset(self):
        self._records = []

    @staticmethod
    def _hungarian_match(gt_boxes, pred_boxes, iou_thr):
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return {}, {}

        iou_mat = pairwise_iou(gt_boxes, pred_boxes)          # (G, P)
        valid   = iou_mat >= iou_thr

        cost = np.where(valid.numpy(), 1-iou_mat.numpy(), 1e6)
        row, col = linear_sum_assignment(cost)

        gt2pred, pred2gt = {}, {}
        for g, p in zip(row, col):
            if cost[g, p] < 1e6:
                gt2pred[int(g)] = int(p)
                pred2gt[int(p)] = int(g)
        return gt2pred, pred2gt

    # ----------------------------
    @torch.no_grad()
    def process(self, inputs, outputs):
        num_samples = len(inputs)
        total = 0
        parent_ok = 0
        next_ok = 0
        match_ok = 0

        for inp, out in zip(inputs, outputs):
            gt   = inp["instances"].to("cpu")
            pred = out["instances"].to("cpu")
            pos_indices = pred.scores > 0.5

            G = len(gt)
            if G == 0:
                continue
            total += G

            N = pos_indices.sum().item()
            if N == 0:
                continue

            parent_logits    = pred.pred_parent_logits[pos_indices].numpy()
            next_logits      = pred.pred_next_logits[pos_indices].numpy()
            root_next_logits = pred.pred_root_next_logits[pos_indices].numpy()

            # デバッグ用
            # parent_logits *= 0.0
            # next_logits *= 0.0
            # root_next_logits *= 0.0
            # gt_parents = gt.gt_parents + 1
            # gt_next = gt.gt_next + 1
            # for g in range(len(gt_parents)):
            #     parent_logits[g][gt_parents[g]] = 1.0
            #     next_logits[g][gt_next[g]] = 1.0
            #     root_next_logits[0] = 1.0

            h_gt, w_gt = gt.image_size
            h_pred, w_pred = pred.image_size
            scale_y = h_gt / h_pred
            scale_x = w_gt / w_pred
            
            pred_boxes = pred.pred_boxes[pos_indices].clone()
            pred_boxes.tensor *= torch.tensor([scale_x, scale_y, scale_x, scale_y])

            gt2pred, pred2gt = self._hungarian_match(
                gt_boxes=gt.gt_boxes,
                pred_boxes=pred_boxes,
                iou_thr=self.iou_threshold
            )

            match_ok += len(gt2pred)

            # col idx == 0 は root
            root_col = 0
            temp = pred2gt
            pred2gt = {root_col: -1}
            pred2gt.update({k+1: v for k, v in temp.items()})

            # IoU しきいを超えた proposal 行
            valid_cols = list(pred2gt.keys())   
            col2gt = list(pred2gt.values())

            if 0 in gt2pred:
                matched_p = gt2pred[0]  # proposal index
                # +1 for root 列オフセット
                if (matched_p + 1) in valid_cols:  
                    root_next = int(root_next_logits.argmax())
                    if root_next == matched_p:
                        next_ok += 1

            # parent / next をそれぞれ GT 1 個ずつ判定
            for g in range(G):
                p = int(gt2pred.get(g, -1))
                if p == -1:
                    continue

                # parent
                logits_par  = parent_logits[p][valid_cols]
                idx_par     = int(logits_par.argmax())
                pred_par_gt = col2gt[idx_par]

                # next
                logits_nxt  = next_logits[p][valid_cols]
                idx_nxt     = int(logits_nxt.argmax())
                pred_nxt_gt = col2gt[idx_nxt]

                if pred_par_gt == gt.gt_parents[g]:
                    parent_ok += 1

                # priority 一番最後の BBox には next は存在しない
                if g >= G-1:
                    continue

                if pred_nxt_gt == gt.gt_next[g]:
                    next_ok += 1

        self._records.append({
            "parent_ok": parent_ok,
            "next_ok"  : next_ok,
            "match_ok" : match_ok,
            "total"    : total,
            "num_samples": num_samples,
        })

    # --------------------------------------------------
    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            records = comm.gather(self._records, dst=0)
            records = list(itertools.chain(*records))

            if not comm.is_main_process():
                return {}
        else:
            records = self._records

        num_samples = [r["num_samples"] for r in records]
        total = [r["total"] for r in records]
        parent_ok = [r["parent_ok"] for r in records]
        next_ok = [r["next_ok"] for r in records]
        match_ok = [r["match_ok"] for r in records]

        num_samples = sum(num_samples)
        total = sum(total)
        parent_ok = sum(parent_ok)
        next_ok = sum(next_ok)
        match_ok = sum(match_ok)

        results = {
            "relation/num_samples": num_samples,
            "relation/eval_total": total,
            "relation/parent_acc": parent_ok / max(1, total),
            "relation/next_acc"  : next_ok   / max(1, total),
            "relation/match_rate": match_ok  / max(1, total),
        }

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "relation_metrics.json"), "w") as f:
                json.dump(results, f, indent=2)

        return results
