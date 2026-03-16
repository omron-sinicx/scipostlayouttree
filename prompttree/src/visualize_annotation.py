# prompttree/src/visualize_bboxes_numbered.py

import os
import json

from torch import mode
import cv2
import numpy as np
from collections import defaultdict
from matplotlib import colors as mcolors
import tqdm
from pathlib import Path


# カテゴリごとの色（必要に応じて調整）
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

TARGET_HEIGHT = 1000


def resize_image_and_bboxes(image, ann_list, target_height=TARGET_HEIGHT):
    """
    画像の高さを target_height に揃えるようにリサイズし、
    bbox も同じ倍率でスケーリングする。
    アスペクト比は維持する。
    """
    h, w = image.shape[:2]
    if h == 0:
        raise ValueError("Image height is zero")

    scale = target_height / float(h)
    new_w = int(round(w * scale))
    new_h = target_height

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    new_anns = []
    for ann in ann_list:
        x, y, bw, bh = ann["bbox"]
        new_ann = ann.copy()
        new_ann["bbox"] = [
            x * scale,
            y * scale,
            bw * scale,
            bh * scale,
        ]
        new_anns.append(new_ann)

    return resized, new_anns


def draw_numbered_bboxes_for_split(split, json_path, image_dir, color_map, output_dir, with_poster):
    """
    - JSON（annotations / image_name / bbox / category_name を想定）を読み込む
    - 各画像ごとに bbox を (上→下, 左→右) でソート
    - 1,2,3,... と番号を振って bbox 上に描画
    - priority / 親子関係 は一切使わない（leakage 回避）
    - 画像は高さを統一してから bbox も同倍率でリサイズ
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"]

    # 画像名ごとにアノテーションをまとめる
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_name"]].append(ann)

    pbar = tqdm.tqdm(anns_by_image.items(), desc=f"[{split}]")

    for image_name, ann_list in pbar:
        img_path = os.path.join(image_dir, image_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image not found: {img_path}")
            continue

        # 画像・bbox をリサイズ（高さを TARGET_HEIGHT に揃える）
        image_resized, ann_list_resized = resize_image_and_bboxes(
            image, ann_list, target_height=TARGET_HEIGHT
        )
        h, w = image_resized.shape[:2]

        # 必要なら少し余白をつける（ラベルがはみ出しにくくなる）
        margin = 0
        canvas = np.ones(
            (h + 2 * margin, w + 2 * margin, 3), dtype=np.uint8
        )

        if with_poster:
            # ポスターあり：元画像を貼る
            canvas[:] = (220, 220, 220)
            canvas[margin : margin + h, margin : margin + w] = image_resized.copy()
        else:
            # ポスターなし：真っ白キャンバスにそのまま描画
            canvas[:] = (255, 255, 255)

        filtered = [
            ann for ann in ann_list_resized 
            if ann.get("category_name") not in ["Root", "Unknown"]
        ]

        # 上から下、左から右でソート
        # y(上) -> x(左) の順
        sorted_anns = sorted(
            filtered,
            key=lambda a: (a["bbox"][1], a["bbox"][0]),
        )

        # 1,2,3,... の番号を振りながら描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5  # 画像高さを揃えているので固定値でほぼ統一される
        thickness = 3

        for idx, ann in enumerate(sorted_anns, start=1):
            x, y, bw, bh = ann["bbox"]
            cat = ann.get("category_name", "Unknown")

            color_hex = color_map.get(cat, "#000000")
            # mcolors.to_rgb は 0〜1 の RGB
            rgb = tuple(int(c * 255) for c in mcolors.to_rgb(color_hex))

            pt1 = (int(x + margin), int(y + margin))
            pt2 = (int(x + bw + margin), int(y + bh + margin))

            # bbox 塗りつぶし（薄め）＋枠線
            overlay = canvas.copy()
            cv2.rectangle(overlay, pt1, pt2, rgb, -1)
            canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
            cv2.rectangle(canvas, pt1, pt2, rgb, thickness=3)

            # ラベル（番号のみ）
            label = str(idx)

            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            tw, th = text_size

            # bbox の左上付近に番号を描く（背景付き）
            # 中心座標計算
            cx = int(x + bw * 0.5) + margin
            cy = int(y + bh * 0.5) + margin

            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            tw, th = text_size

            # 中心に文字を配置（テキスト基準点が左下なので調整）
            text_x = int(cx - tw / 2)
            text_y = int(cy + th / 2)

            # 背景矩形（少し余裕を持たせる）
            label_bg_tl = (text_x - 10, text_y - th - 10)
            label_bg_br = (text_x + tw + 10, text_y + 10)

            cv2.rectangle(canvas, label_bg_tl, label_bg_br, rgb, -1)
            text_color = (255, 255, 255)
            cv2.putText(canvas, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        save_dir = os.path.join(output_dir, split)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, canvas)
        pbar.set_postfix_str(f"saved: {out_path}")


def main():
    with_poster = False

    base_dir = Path("./../../scipostlayout/poster")
    splits = ["train", "dev", "test"]
    if with_poster:
        output_dir = base_dir / "vis_w_poster"
    else:
        output_dir = base_dir / "vis_wo_poster"

    for split in splits:
        print(f"Processing split: {split}")
        json_path = base_dir / "png"/ f"{split}_tree.json"
        image_dir = base_dir / "png" / split
        draw_numbered_bboxes_for_split(split, json_path, image_dir, color_map, output_dir, with_poster)


if __name__ == "__main__":
    main()
