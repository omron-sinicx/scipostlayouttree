from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
from matplotlib import colors as mcolors


# 既存のカラーマップ（必要なら外から渡せるようにしてもOK）
COLOR_MAP = {
    "Root": "#AD8D43",
    "Title": "#3df343",
    "Author Info": "#b41257",
    "Section": "#4087e7",
    "Text": "#215000",
    "List": "#be16f9",
    "Figure": "#f5602c",
    "Table": "#8959de",
    "Caption": "#8ff427",
    "Unknown": "#0965f3",
}


@dataclass(frozen=True)
class AnnView:
    """可視化に必要な最小情報だけ持つ view"""
    id: int
    image_name: str
    category_name: str
    bbox: Tuple[float, float, float, float]  # x,y,w,h
    bbox_number: int  # (y,x) sort で付与


def load_anns_by_image(gt_json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    GT JSON（COCO拡張）から image_name -> annotations(list) を返す。
    前提: 各 ann に image_name, bbox, category_name がある。
    """
    with gt_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    anns_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        image_name = ann.get("image_name")
        if image_name is None:
            continue
        anns_by_image.setdefault(image_name, []).append(ann)
    return anns_by_image


def build_numbered_views(ann_list: List[Dict[str, Any]]) -> List[AnnView]:
    """
    record_to_example と同じ規則で bbox_number を振る：
    bbox を (y, x) でソートして 1..N
    """
    sorted_anns = sorted(ann_list, key=lambda a: (a["bbox"][1], a["bbox"][0]))
    views: List[AnnView] = []
    for number, ann in enumerate(sorted_anns, start=1):
        x, y, w, h = ann["bbox"]
        views.append(
            AnnView(
                id=int(ann["id"]),
                image_name=str(ann["image_name"]),
                category_name=str(ann.get("category_name", "Unknown")),
                bbox=(float(x), float(y), float(w), float(h)),
                bbox_number=number,
            )
        )
    return views


def _draw_arrow_fixed_top_to_top(
    canvas: np.ndarray,
    start_box_xywh: Tuple[float, float, float, float],
    end_box_xywh: Tuple[float, float, float, float],
    offset: Tuple[int, int] = (0, 0),
    color: Tuple[int, int, int] = (28, 28, 28),
    thickness: int = 6,
    tip_size: int = 45,
) -> None:
    ox, oy = offset
    sx, sy, sw, sh = start_box_xywh
    ex, ey, ew, eh = end_box_xywh

    start = np.array([int(sx + sw / 2 + ox), int(sy + oy)], dtype=np.int32)
    end = np.array([int(ex + ew / 2 + ox), int(ey + oy)], dtype=np.int32)

    vec = end - start
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return
    vec = vec / norm
    left = np.array([-vec[1], vec[0]])
    right = np.array([vec[1], -vec[0]])

    base = end - vec * tip_size
    point1 = (base + left * tip_size * 0.6).astype(int)
    point2 = (base + right * tip_size * 0.6).astype(int)
    triangle = np.array([end, point1, point2], dtype=np.int32)

    cv2.line(canvas, tuple(start), tuple(end), color, thickness, cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, triangle, color)


def render_one_prediction(
    *,
    poster_image_path: Path,
    ann_list: List[Dict[str, Any]],
    reading_order: List[int],
    tree: List[Dict[str, int]],
    out_path: Path,
    color_map: Dict[str, str] = COLOR_MAP,
    margin: int = 100,
) -> Path:
    """
    1枚の画像について：
      - GT bbox（ann_list）を描画
      - 予測 reading_order から priority を作って表示
      - 予測 tree（bbox_number 空間）から矢印を描画
    """
    image = cv2.imread(str(poster_image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {poster_image_path}")

    h, w = image.shape[:2]
    canvas = np.ones((h + 2 * margin, w + 2 * margin, 3), dtype=np.uint8) * 220
    canvas[margin: margin + h, margin: margin + w] = image.copy()

    views = build_numbered_views(ann_list)
    num_to_view = {v.bbox_number: v for v in views}

    # bbox_number -> priority (0-based). 無い場合 -1
    rank_map = {bn: i for i, bn in enumerate(reading_order)}

    # parent -> children (bbox_number 空間)
    children: Dict[int, List[int]] = {}
    root_children: List[int] = []

    for rel in tree:
        child = int(rel["bbox_number"])
        parent = int(rel["parent"])
        if parent <= 0:
            root_children.append(child)
            continue
        children.setdefault(parent, []).append(child)

    # --- bbox描画 ---
    for v in views:
        x, y, bw, bh = v.bbox
        cat = v.category_name
        priority = rank_map.get(v.bbox_number, -1)

        color_hex = color_map.get(cat, "#000000")
        rgb = tuple(int(c * 255) for c in mcolors.to_rgb(color_hex))

        pt1 = (int(x + margin), int(y + margin))
        pt2 = (int(x + bw + margin), int(y + bh + margin))

        overlay = canvas.copy()
        cv2.rectangle(overlay, pt1, pt2, rgb, -1)
        canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
        cv2.rectangle(canvas, pt1, pt2, rgb, thickness=6)

        label = f"{cat}: {priority}" if priority >= 0 else cat
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 4
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        lx, ly = int(x + margin), int(y + margin)
        label_bg_tl = (lx, max(0, ly - th - 16))
        label_bg_br = (lx + tw + 16, ly)

        cv2.rectangle(canvas, label_bg_tl, label_bg_br, rgb, -1)
        text_color = (255, 255, 255) if cat in ["Author Info", "Text"] else (28, 28, 28)
        cv2.putText(
            canvas,
            label,
            (lx + 8, ly - 8),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    # --- 矢印描画 ---
    for p, child_list in children.items():
        pv = num_to_view.get(p)
        if pv is None:
            continue
        for c in child_list:
            cv = num_to_view.get(c)
            if cv is None:
                continue
            _draw_arrow_fixed_top_to_top(
                canvas,
                pv.bbox,
                cv.bbox,
                offset=(margin, margin),
                color=(28, 28, 28),
                thickness=6,
                tip_size=45,
            )

    # --- Root -> root_children 矢印描画 ---
    root_view = next((v for v in views if v.category_name == "Root"), None)
    if root_view is not None:
        for c in root_children:
            cv = num_to_view.get(c)
            if cv is None:
                continue
            _draw_arrow_fixed_top_to_top(
                canvas,
                root_view.bbox,
                cv.bbox,
                offset=(margin, margin),
                color=(28, 28, 28),
                thickness=6,
                tip_size=45,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path
