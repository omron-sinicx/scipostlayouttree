import os
import json
import cv2
import numpy as np
from collections import defaultdict
from matplotlib import colors as mcolors
import tqdm
from pathlib import Path


color_map = {
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


def draw_annotations(target, json_path, image_dir, color_map, output_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"]

    # image_name -> anns
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_name"]].append(ann)

    # id -> ann
    id_to_ann = {ann["id"]: ann for ann in annotations}

    def build_children_from_parents(ann_list):
        """
        parents ベースの COCO から children マップを再構築する
        """
        children = defaultdict(list)
        for ann in ann_list:
            for pid in ann.get("parents", []):
                if pid != -1:
                    children[pid].append(ann["id"])
        return children

    def draw_arrow_fixed_top_to_top(
        canvas,
        start_box,
        end_box,
        offset=(0, 0),
        color=(0, 0, 0),
        thickness=6,
        tip_size=45,
    ):
        ox, oy = offset
        start = np.array(
            [int(start_box[0] + start_box[2] / 2 + ox), int(start_box[1] + oy)]
        )
        end = np.array(
            [int(end_box[0] + end_box[2] / 2 + ox), int(end_box[1] + oy)]
        )

        vec = end - start
        norm = np.linalg.norm(vec)
        if norm == 0:
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

    pbar = tqdm.tqdm(anns_by_image.items())
    for image_name, ann_list in pbar:
        img_path = os.path.join(image_dir, image_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image not found: {img_path}")
            continue

        h, w = image.shape[:2]
        margin = 100
        canvas = np.ones((h + 2 * margin, w + 2 * margin, 3), dtype=np.uint8) * 220
        canvas[margin : margin + h, margin : margin + w] = image.copy()

        # parents -> children を再構築
        children_map = build_children_from_parents(ann_list)

        # --- bbox 描画 ---
        for ann in ann_list:
            x, y, bw, bh = ann["bbox"]
            cat = ann["category_name"]
            priority = ann.get("priority", -1)

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
            font_scale = 4.0
            thickness = 6
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            tw, th = text_size
            lx, ly = int(x + margin), int(y + margin)
            label_bg_tl = (lx, ly - th - 20)
            label_bg_br = (lx + tw + 20, ly)

            cv2.rectangle(canvas, label_bg_tl, label_bg_br, rgb, -1)
            text_color = (255, 255, 255) if cat in ["Author Info", "Text"] else (28, 28, 28)
            cv2.putText(
                canvas,
                label,
                (lx + 10, ly - 10),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        # --- 親子関係（矢印）描画 ---
        for parent_id, child_ids in children_map.items():
            parent = id_to_ann.get(parent_id)
            if parent is None:
                continue

            px, py, pw, ph = parent["bbox"]
            parent_box = [px, py, pw, ph]

            for cid in child_ids:
                child = id_to_ann.get(cid)
                if child is None:
                    continue
                if child["image_name"] != image_name:
                    continue

                cx, cy, cw, ch = child["bbox"]
                child_box = [cx, cy, cw, ch]

                draw_arrow_fixed_top_to_top(
                    canvas,
                    parent_box,
                    child_box,
                    offset=(margin, margin),
                    color=(28, 28, 28),
                    thickness=6,
                    tip_size=45,
                )

        os.makedirs(output_dir, exist_ok=True)
        out_path = output_dir / image_name
        cv2.imwrite(str(out_path), canvas)
        print(f"[{target}] saved: {out_path}")


def main():
    base_dir = Path("/scipostlayout/poster/png")
    pred_dir = Path("./pred")
    targets = ["test_tree_openai.json"]
    output_base_dir = Path("./vis_pred")

    for target in targets:
        print(f"Processing target: {target}")
        json_path = pred_dir / target
        image_dir = base_dir / "test"
        output_dir = output_base_dir / target.replace(".json", "")
        draw_annotations(target, json_path, image_dir, color_map, output_dir)


if __name__ == "__main__":
    main()
