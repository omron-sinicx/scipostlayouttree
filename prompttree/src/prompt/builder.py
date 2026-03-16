from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple
import base64
from mimetypes import guess_type

from data.schema import Example, BBox


# ----------------------------------------------------------------------
# 画像パターン
# ----------------------------------------------------------------------

class ImagePattern(str, Enum):
    """
    LLM に渡す画像の組み合わせパターン。

    - POSTER_PLUS_WHITE: 生ポスター + 白背景BBox
    - POSTER_OVERLAY:    ポスター上にBBoxを重ねた画像
    """
    POSTER_PLUS_WHITE = "poster_plus_white"
    POSTER_OVERLAY = "poster_overlay"


# ----------------------------------------------------------------------
# few-shot 1 例分の情報
# ----------------------------------------------------------------------

@dataclass
class FewshotLabels:
    reading_order: List[int]           # [bbox_number, ...]
    tree: List[Dict[str, int]]         # {"bbox_number": int, "parent": int}


@dataclass
class FewshotExample:
    """
    few-shot 1 例分。元データと同じ Example（BBox + 画像）に、
    reading_order / tree の正解ラベルを紐づけたもの。
    """
    example: Example
    labels: FewshotLabels


# ----------------------------------------------------------------------
# 共通テキスト部分の構築
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in document structure analysis.
You are given bounding boxes from a scientific poster and one or more images of the poster.
Your task is to infer:
1) the reading order of the bboxes,
2) the parent-child relationships forming a rooted tree.
"""


def _format_bbox(b: BBox) -> str:
    t = (b.text or "").strip()
    if t:
        t = t.replace("\n", " ")
        if len(t) > 120:
            t = t[:120] + "..."
        text_part = f', text="{t}"'
    else:
        text_part = ""

    return (
        f"- bbox_number={b.number}, "
        f"category={b.category}, "
        f"x={b.x_norm:.3f}, y={b.y_norm:.3f}, "
        f"w={b.w_norm:.3f}, h={b.h_norm:.3f}"
        f"{text_part}"
    )

def _format_bbox_from_dict(d: Dict[str, Any]) -> str:
    """
    JSONL 側の bboxes を使う場合のフォーマット。
    （必要ならこちらも利用できます）
    """
    return (
        f"- bbox_number={d['number']}, "
        f"category={d['category']}, "
        f"x={d['x_norm']:.3f}, y={d['y_norm']:.3f}, "
        f"w={d['w_norm']:.3f}, h={d['h_norm']:.3f}"
    )


def _format_labels(labels: FewshotLabels) -> str:
    """
    few-shot の正解 (reading_order, tree) をテキストにする。
    """
    ro_json = json.dumps(labels.reading_order, ensure_ascii=False)
    tree_json = json.dumps(labels.tree, ensure_ascii=False, indent=2)
    return f"reading_order = {ro_json}\n" f"tree = {tree_json}"


def build_core_text_for_example(example: Example) -> str:
    """
    1 例分の「Input bboxes: ...」というテキスト部分を作る（画像は含めない）。
    """
    lines: List[str] = []
    lines.append("Input bboxes:")
    for b in example.bboxes:
        lines.append(_format_bbox(b))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 画像選択ロジック
# ----------------------------------------------------------------------

def select_images_for_pattern(example: Example, pattern: ImagePattern) -> List[Path]:
    """
    Example.images と ImagePattern に基づいて、LLM に渡す画像 Path のリストを返す。
    """
    imgs: List[Path] = []

    if pattern == ImagePattern.POSTER_PLUS_WHITE:
        # 生ポスター + 白背景BBox
        imgs.append(example.images.poster)
        if example.images.bboxes_white is not None:
            imgs.append(example.images.bboxes_white)
    elif pattern == ImagePattern.POSTER_OVERLAY:
        # ポスター上にBBoxを重ねた画像。なければ生ポスターにフォールバック。
        if example.images.bboxes_on_poster is not None:
            imgs.append(example.images.bboxes_on_poster)
        else:
            imgs.append(example.images.poster)
    else:
        raise ValueError(f"Unknown ImagePattern: {pattern}")

    return imgs


# ----------------------------------------------------------------------
# OpenAI 用 builder（few-shot も画像付き）
# ----------------------------------------------------------------------


def _path_to_data_url(path: Path) -> str:
    """
    画像ファイルを読み込み、data URL (data:image/png;base64,...) に変換する。
    """
    # MIME type を推定（拡張子が .png / .jpg などならここで決まる）
    mime, _ = guess_type(str(path))
    if mime is None:
        # 一応フォールバック
        mime = "application/octet-stream"

    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_llm_messages(
    target: Example,
    fewshots: List[FewshotExample] | None,
    pattern: ImagePattern,
) -> List[Dict[str, Any]]:
    """
    OpenAI Chat Completions (multimodal) 用の messages を構築する。

    - system: SYSTEM_PROMPT
    - user: content = [text, image, text, image, ..., text, image, ...] の配列
        * 最初に few-shot 例を並べる（例ごとにテキスト → 画像 → 正解ラベル）
        * 最後に今回解くべき example のテキスト → 画像
    """

    content_parts: List[Dict[str, Any]] = []

    # --- few-shot 例（テキスト + 画像 + ラベル）---
    if fewshots:
        for i, fs in enumerate(fewshots, start=1):
            ex = fs.example
            labels = fs.labels

            # few-shot テキスト（入力＋画像の案内）
            lines: List[str] = []
            lines.append(f"Example {i}:")
            lines.append("")
            lines.append(build_core_text_for_example(ex))
            lines.append("")
            lines.append(f"Images for Example {i}:")
            text_block = "\n".join(lines)

            content_parts.append({"type": "text", "text": text_block})

            # few-shot 画像
            for img_path in select_images_for_pattern(ex, pattern):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": _path_to_data_url(img_path)},
                })

            # few-shot 正解ラベル
            content_parts.append({
                "type": "text",
                "text": "Output:\n" + _format_labels(labels),
            })

    # --- 推論対象 example（テキスト + 画像）---
    lines: List[str] = []
    lines.append("Now solve the following example.")
    lines.append("")
    lines.append(build_core_text_for_example(target))
    lines.append("")
    lines.append(
        "Use both the textual description of the bboxes and the visual layout in the images.\n"
        "Output the result in the same format as the previous examples, using JSON only.\n"
        "Do not include any explanations."
    )
    lines.append("")
    lines.append("Images for the target example:")
    target_text = "\n".join(lines)

    # ターゲット例のテキスト
    content_parts.append({"type": "text", "text": target_text})

    # ターゲット例の画像
    for img_path in select_images_for_pattern(target, pattern):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": _path_to_data_url(img_path)},
        })

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]
    return messages


def build_llm_messages_with_feedback(
    target: Example,
    fewshots: List[FewshotExample] | None,
    pattern: ImagePattern,
    prev_prediction: Dict[str, Any],
    prev_vis_path: Path,
    round_idx: int,
) -> List[Dict[str, Any]]:
    """
    Round2+ 用 builder。

    Round1 と同じ入力に加えて、
    - 直前ラウンドのモデル予測（reading_order / tree の JSON）
    - 直前ラウンドの可視化画像
    を追加コンテキストとして与える。

    出力フォーマットは Round1 と同一（JSON のみ）。
    """

    # --- Round1 と同じ messages をまず作る ---
    messages = build_llm_messages(
        target=target,
        fewshots=fewshots,
        pattern=pattern,
    )

    # user メッセージの content 配列を取得
    content_parts: List[Dict[str, Any]] = messages[1]["content"]

    prev_json = json.dumps(prev_prediction, ensure_ascii=False, indent=2)

    feedback_text = (
        f"Round {round_idx}: Refinement step.\n"
        "You previously produced a prediction for this SAME target. "
        "Now refine it using the visualization.\n\n"
        "Goal: improve correctness while making minimal necessary changes.\n\n"
        "Hard constraints (must satisfy):\n"
        "1) reading_order is a permutation of ALL bbox_number values exactly once.\n"
        "2) tree contains exactly one entry per bbox_number.\n"
        "3) parent is either -1 (Root) or a valid bbox_number.\n"
        "4) The parent-child graph must be a rooted tree: no cycles, and every node reaches Root.\n"
        "5) Output JSON only with keys: reading_order, tree. No extra keys, no explanations.\n\n"
        "Previous prediction JSON:\n"
        f"{prev_json}\n\n"
        "Visualization of the previous prediction is attached next. "
        "Use it to identify incorrect arrows and ordering, then fix them.\n"
    )

    content_parts.append(
        {
            "type": "text",
            "text": feedback_text,
        }
    )

    # --- フィードバック用画像（前回可視化）---
    content_parts.append(
        {
            "type": "image_url",
            "image_url": {"url": _path_to_data_url(prev_vis_path)},
        }
    )

    # --- 最後の指示 ---
    content_parts.append(
        {
            "type": "text",
            "text": (
                "Now output the improved result in the same JSON format as before.\n"
                "JSON only. No explanations."
            ),
        }
    )

    return messages
