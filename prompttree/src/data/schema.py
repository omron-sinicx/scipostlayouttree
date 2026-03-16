from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

CategoryName = Literal[
    "Root",
    "Title",
    "Author Info",
    "Section",
    "Text",
    "List",
    "Figure",
    "Table",
    "Caption",
    "Unknown",
]


@dataclass
class BBox:
    """1つのレイアウト矩形（bounding box）"""

    id: int
    number: int  # 上から下、左から右ソート後に振る 1,2,3,...
    category: CategoryName

    # ピクセル座標（元画像そのまま）
    x: float
    y: float
    w: float
    h: float

    # 0〜1 に正規化した座標（LLM 入力用）
    x_norm: float
    y_norm: float
    w_norm: float
    h_norm: float

    text: str | None = None  # OCR テキスト


@dataclass
class ExampleImages:
    """1枚のポスターに対して利用可能な画像バリエーション"""

    poster: Path  # 生ポスター画像（png/train/dev/test/...）
    bboxes_white: Path | None = None  # 白背景に BBox だけ描画した画像
    bboxes_on_poster: Path | None = None  # ポスター上に BBox を重ね描きした画像


@dataclass
class Example:
    """LLM に渡す1サンプル分の情報"""

    image_name: str
    width: int
    height: int
    bboxes: List[BBox]
    images: ExampleImages


def record_to_example(rec: Dict[str, Any]) -> Example:
    """
    SciPostLayoutTreeDataset が返す1レコード(dict)を Example に変換する。

    - bbox を (y, x) でソートして number=1,2,... を振る
    - 座標を画像サイズで正規化して x_norm, y_norm, w_norm, h_norm を計算
    - file_name から各種画像パス (poster, vis_w_poster, vis_wo_poster) を組み立てる
    """

    file_path = Path(rec["file_name"]).resolve()
    width: int = rec["width"]
    height: int = rec["height"]
    image_name: str = file_path.name

    # 例: /scipostlayout/poster/png/dev/11066.png
    split = file_path.parent.name  # "train" / "dev" / "test" を想定

    # poster_root = /scipostlayout/poster
    poster_root = file_path.parent.parent.parent

    poster_path = file_path
    bboxes_white_path = poster_root / "vis_wo_poster" / split / image_name
    bboxes_on_poster_path = poster_root / "vis_w_poster" / split / image_name

    images = ExampleImages(
        poster=poster_path,
        bboxes_white=bboxes_white_path if bboxes_white_path.is_file() else None,
        bboxes_on_poster=bboxes_on_poster_path if bboxes_on_poster_path.is_file() else None,
    )

    anns = rec["annotations"]

    # 上から下、左から右でソート (y, x)
    sorted_anns = sorted(anns, key=lambda a: (a["bbox"][1], a["bbox"][0]))

    bboxes: List[BBox] = []
    for number, ann in enumerate(sorted_anns, start=1):
        x, y, w, h = ann["bbox"]

        x_norm = x / width
        y_norm = y / height
        w_norm = w / width
        h_norm = h / height

        bboxes.append(
            BBox(
                id=ann["id"],
                number=number,
                category=ann["category_name"],
                x=x,
                y=y,
                w=w,
                h=h,
                x_norm=x_norm,
                y_norm=y_norm,
                w_norm=w_norm,
                h_norm=h_norm,
                text=ann.get("text", "[BLANK]"),
            )
        )

    return Example(
        image_name=image_name,
        width=width,
        height=height,
        bboxes=bboxes,
        images=images,
    )
