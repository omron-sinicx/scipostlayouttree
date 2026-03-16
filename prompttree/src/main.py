from __future__ import annotations

import os
import argparse
from pathlib import Path

from data.scipostlayouttree import SciPostLayoutTreeDataset
from pipeline.run_eval import run_full_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-based document layout tree analyzer (prompttree)."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="使用する LLM のモデル名。",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="結果を書き出すディレクトリ。",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="この実験の名前。output-dir/run-name/ 以下に保存される。",
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=2,
        help="フィードバックループのラウンド数（Round1含む）。",
    )

    parser.add_argument(
        "--k-fewshot",
        type=int,
        default=2,
        help="retrieval で選ぶ few-shot 件数。",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="LLM 出力の parse に失敗した場合の再試行回数。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="parse 失敗時フォールバック（ランダム生成）の再現性用シード。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir) / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    train_dataset = SciPostLayoutTreeDataset(split="train")
    test_dataset = SciPostLayoutTreeDataset(split="test")

    run_full_eval(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=args.model,
        output_root=output_root,
        run_name=args.run_name,
        num_rounds=args.num_rounds,
        k_fewshot=args.k_fewshot,
        max_retries=args.max_retries,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
