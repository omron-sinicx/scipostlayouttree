from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.schema import Example, BBox
from prompt.builder import FewshotExample


def _bbox_to_xyxy(b: BBox) -> Tuple[float, float, float, float]:
    x1 = float(b.x)
    y1 = float(b.y)
    x2 = x1 + float(b.w)
    y2 = y1 + float(b.h)
    return x1, y1, x2, y2


def _iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0

    return inter / union


def _iou_matrix(
    target_boxes: List[BBox],
    cand_boxes: List[BBox],
) -> np.ndarray:
    """
    IoU matrix of shape (n_target, n_cand)

    Category mismatch is forbidden:
      if target_boxes[i].category != cand_boxes[j].category, IoU = 0.
    """
    t_xyxy = [_bbox_to_xyxy(b) for b in target_boxes]
    c_xyxy = [_bbox_to_xyxy(b) for b in cand_boxes]

    n, m = len(t_xyxy), len(c_xyxy)
    mat = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            if target_boxes[i].category != cand_boxes[j].category:
                # カテゴリ不一致はマッチ禁止（IoU=0）
                continue
            mat[i, j] = _iou_xyxy(t_xyxy[i], c_xyxy[j])
    return mat


def _hungarian_max_iou_sum(
    target_boxes: List[BBox],
    cand_boxes: List[BBox],
) -> float:
    """
    Compute the maximum possible sum of IoUs using Hungarian matching
    (optimal one-to-one assignment).
    """
    if not target_boxes or not cand_boxes:
        return 0.0

    iou = _iou_matrix(target_boxes, cand_boxes)
    cost = -iou  # maximize IoU sum

    row_ind, col_ind = linear_sum_assignment(cost)
    return float(iou[row_ind, col_ind].sum())


def compute_hungarian_iou_score(
    target: Example,
    cand: Example,
) -> float:
    """
    Hungarian IoU score:

        score = sum(matched IoUs) / max(n_target, n_cand)

    - One-to-one matching is solved optimally by Hungarian algorithm.
    - Unmatched boxes are implicitly treated as IoU = 0.
    """
    nt = len(target.bboxes)
    nc = len(cand.bboxes)
    denom = max(nt, nc)
    if denom == 0:
        return 0.0

    matched_sum = _hungarian_max_iou_sum(target.bboxes, cand.bboxes)
    return matched_sum / float(denom)


@dataclass(frozen=True)
class RetrievalResult:
    fewshot: FewshotExample
    score: float  # Hungarian IoU score


def retrieve_topk_by_hungarian_iou(
    target: Example,
    pool: List[FewshotExample],
    k: int,
    exclude_same_image: bool = True,
) -> List[RetrievalResult]:
    if k <= 0:
        return []

    results: List[RetrievalResult] = []
    for fs in pool:
        if exclude_same_image and fs.example.image_name == target.image_name:
            continue

        score = compute_hungarian_iou_score(target, fs.example)
        results.append(RetrievalResult(fewshot=fs, score=score))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]
