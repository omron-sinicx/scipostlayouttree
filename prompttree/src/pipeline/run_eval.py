from pathlib import Path
import json
import random
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime
import tqdm

from data.scipostlayouttree import SciPostLayoutTreeDataset
from data.schema import record_to_example, Example
from data.fewshot_loader import load_fewshot_examples
from data.fewshot_retriever import retrieve_topk_by_hungarian_iou
from data.parse import parse_llm_output, try_parse_llm_output, PredictedStructure
from data.validate import validate_predicted_structure
from data.pred_visualizer import load_anns_by_image, render_one_prediction

from prompt.builder import (
    build_llm_messages,
    build_llm_messages_with_feedback,
    ImagePattern,
)

from llm.openai_client import OpenAIClient
from llm.gemini_client import GeminiClient
from llm.claude_client import ClaudeClient

from eval.build_pred_coco import build_pred_coco_from_dataset
from eval.tree_metrics import evaluate_trees


ROOT_PARENT = -1


def make_random_prediction(n: int, seed: int | None = None) -> Dict[str, Any]:
    """
    root 親は -1 固定で、必ず木になるランダム構造を作る。
    - reading_order: 1..n のランダム permutation
    - tree: 各 bbox_number に親を 1つ割当（親は -1 または他ノード）
      cycle を避けるため、ランダム順序の「先に出たノード」から親を選ぶ。
    """
    rng = random.Random(seed)

    reading_order = list(range(1, n + 1))
    rng.shuffle(reading_order)

    nodes = list(range(1, n + 1))
    rng.shuffle(nodes)

    tree: List[Dict[str, int]] = []
    # 先頭は root child にする（parent=-1）
    first = nodes[0]
    tree.append({"bbox_number": first, "parent": ROOT_PARENT})

    seen = [first]
    for child in nodes[1:]:
        # 親候補: 既に出たノード or Root(-1)
        parent = rng.choice([ROOT_PARENT] + seen)
        tree.append({"bbox_number": child, "parent": parent})
        seen.append(child)

    # tree を bbox_number 昇順に揃えておくと扱いやすい
    tree.sort(key=lambda d: d["bbox_number"])

    return {"reading_order": reading_order, "tree": tree}


def run_llm_with_retry_or_fallback(
    client,
    messages: List[Dict[str, Any]],
    n_bboxes: int,
    max_retries: int,
    seed: int | None = None,
) -> Tuple[PredictedStructure, Dict[str, Any]]:
    """
    LLM -> try_parse を max_retries 回試す。
    成功したら (PredictedStructure, meta) を返す。
    全失敗ならランダム予測を返す（root親=-1固定）。
    """
    attempts = 0
    for _ in range(max_retries):
        attempts += 1
        raw = client.run(messages)

        parsed = try_parse_llm_output(raw)
        if parsed is None:
            last_errors = ["parse_failed"]
            last_stage = "parse"
            continue

        errors = validate_predicted_structure(
            parsed,
            n_bboxes=n_bboxes,
            root_parent=ROOT_PARENT,
        )
        if not errors:
            meta = {
                "parse_success": True,
                "validate_success": True,
                "validate_errors": [],
                "attempts": attempts,
                "is_fallback": False,
            }
            return parsed, meta

        # validate failed -> retry
        last_errors = errors
        last_stage = "validate"

    fallback = make_random_prediction(n_bboxes, seed=seed)
    parsed = PredictedStructure(
        reading_order=fallback["reading_order"],
        tree=fallback["tree"],
    )
    meta = {
        "parse_success": False,          # 「最終採用が parse+validate を満たしたか」という意味に寄せる
        "validate_success": False,
        "validate_errors": last_errors[:10],
        "failed_stage": last_stage,
        "attempts": attempts,
        "is_fallback": True,
    }
    return parsed, meta


def dataset_to_examples(dataset: SciPostLayoutTreeDataset) -> List[Example]:
    return [record_to_example(dataset[i]) for i in range(len(dataset))]


def debug_single_example(train, test) -> None:
    fewshots = load_fewshot_examples(
        Path("fewshot_jsonl/scipost_train_fewshot.jsonl"), train
    )
    fewshots_for_prompt = fewshots[:2]

    target = record_to_example(test[0])

    messages = build_llm_messages(
        target=target,
        fewshots=fewshots_for_prompt,
        pattern=ImagePattern.POSTER_PLUS_WHITE,
    )

    client = OpenAIClient(model="gpt-5-mini-2025-08-07")
    import pdb; pdb.set_trace()
    raw_text = client.run(messages)

    result = parse_llm_output(raw_text)
    print(result)


def run_full_eval(
    train_dataset: SciPostLayoutTreeDataset,
    test_dataset: SciPostLayoutTreeDataset,
    *,
    model: str,
    output_root: Path,
    run_name: str,
    num_rounds: int,
    k_fewshot: int,
    max_retries: int,
    seed: int,
) -> None:
    """
    SciPostLayoutTree を対象に、few-shot retrieval + multi-round 推論を行い、
    pred COCO を生成して TED/STEDS/REDS を評価する。

    運用方針:
    - log.jsonl は image×round の 1行ログを append する
    - 再実行時は log.jsonl を正規化し、最終round行が存在しない画像の途中ログを削除して最初からやり直す
    - 最終round行が存在する画像はスキップする
    - 対象画像集合（例: 先頭300件）に対してのみ、再開・評価・pred COCO 生成を行う
    """
    fewshots = load_fewshot_examples(
        Path("fewshot_jsonl/scipost_train_fewshot.jsonl"), train_dataset
    )

    def _make_client(model_name: str):
        name = model_name.lower()
        if name.startswith("gpt") or "openai" in name:
            return OpenAIClient(model=model_name)
        if name.startswith("gemini") or "google" in name:
            return GeminiClient(model=model_name)
        if name.startswith("claude") or "anthropic" in name:
            return ClaudeClient(model=model_name)
        raise ValueError(f"Unknown model/provider for model='{model_name}'")

    def _compact_log_for_resume(
        log_path: Path,
        run_name: str,
        num_rounds: int,
        allowed_files: Set[str],
    ) -> Set[str]:
        """
        log.jsonl を読み、run_name が一致し、かつ allowed_files に含まれる file_name のみを対象に
        file_name 単位で完了判定を行う。

        完了 = round == num_rounds の行が存在する。
        未完了 file_name の行は全削除し、完了済み file_name の行だけ残す。
        allowed_files 外の行、および他の run_name の行は保持する。

        戻り値は完了済み file_name の集合（スキップ用）。
        """
        if not log_path.is_file():
            return set()

        kept_lines: List[str] = []
        run_lines_by_file: Dict[str, List[str]] = {}
        completed: Set[str] = set()

        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    kept_lines.append(raw)
                    continue

                if rec.get("run_name") != run_name:
                    kept_lines.append(raw)
                    continue

                fname = rec.get("file_name")
                rnd = rec.get("round")
                if not isinstance(fname, str) or not isinstance(rnd, int):
                    continue

                if fname not in allowed_files:
                    kept_lines.append(raw)
                    continue

                run_lines_by_file.setdefault(fname, []).append(raw)
                if rnd == num_rounds:
                    completed.add(fname)

        for fname, lines in run_lines_by_file.items():
            if fname in completed:
                kept_lines.extend(lines)

        tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            for raw in kept_lines:
                f.write(raw + "\n")
        tmp_path.replace(log_path)

        return completed

    def _load_final_predictions_from_log(
        log_path: Path,
        run_name: str,
        num_rounds: int,
        allowed_files: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        log.jsonl から、run_name の最終round（round==num_rounds）の結果を、
        allowed_files に含まれる file_name のみ復元する。
        同一 file_name が複数回完了している場合は、ファイル内で後に出たものを採用する。
        """
        preds: Dict[str, Dict[str, Any]] = {}
        if not log_path.is_file():
            return preds

        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if rec.get("run_name") != run_name:
                    continue
                if rec.get("round") != num_rounds:
                    continue

                fname = rec.get("file_name")
                if not isinstance(fname, str) or not fname:
                    continue
                if fname not in allowed_files:
                    continue

                reading_order = rec.get("reading_order")
                tree = rec.get("tree")
                if not isinstance(reading_order, list) or not isinstance(tree, list):
                    continue

                preds[fname] = {"reading_order": reading_order, "tree": tree}

        return preds

    client = _make_client(model)

    base_dir = Path("/scipostlayout/poster/png")
    gt_json = base_dir / "test_tree_ocr.json"
    anns_by_image = load_anns_by_image(gt_json)

    pred_dir = output_root / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_json = pred_dir / f"{run_name}.json"

    vis_root = output_root / "vis_pred" / run_name
    vis_root.mkdir(parents=True, exist_ok=True)

    log_path = output_root / "log.jsonl"

    all_test_examples: List[Example] = dataset_to_examples(test_dataset)
    ex_by_fname: Dict[str, Example] = {Path(ex.images.poster).name: ex for ex in all_test_examples}

    run_examples: List[Example] = all_test_examples
    allowed_files: Set[str] = {Path(ex.images.poster).name for ex in run_examples}

    completed_files = _compact_log_for_resume(log_path, run_name, num_rounds, allowed_files)
    log_f = log_path.open("a", encoding="utf-8")

    pbar = tqdm.tqdm(run_examples)
    for ex in pbar:
        poster_path = Path(ex.images.poster)
        file_name = poster_path.name

        if file_name in completed_files:
            print(f"Skipping completed: {file_name}")
            continue

        ann_list = anns_by_image[file_name]

        topk = retrieve_topk_by_hungarian_iou(ex, fewshots, k=k_fewshot)
        fewshots_for_prompt = [r.fewshot for r in topk]

        prev_pred: Optional[Dict[str, Any]] = None
        prev_vis_path: Optional[Path] = None
        final_parsed: Optional[PredictedStructure] = None
        final_meta: Optional[Dict[str, Any]] = None

        per_image_seed = (seed ^ (hash(file_name) & 0xFFFFFFFF)) & 0xFFFFFFFF

        for r in range(1, num_rounds + 1):
            if r == 1:
                messages = build_llm_messages(
                    target=ex,
                    fewshots=fewshots_for_prompt,
                    pattern=ImagePattern.POSTER_PLUS_WHITE,
                )
            else:
                if prev_pred is None or prev_vis_path is None:
                    raise RuntimeError(
                        f"Missing previous prediction/visualization for feedback round: file_name={file_name}, round={r}"
                    )
                messages = build_llm_messages_with_feedback(
                    target=ex,
                    fewshots=fewshots_for_prompt,
                    pattern=ImagePattern.POSTER_PLUS_WHITE,
                    prev_prediction=prev_pred,
                    prev_vis_path=prev_vis_path,
                    round_idx=r,
                )

            parsed, meta = run_llm_with_retry_or_fallback(
                client=client,
                messages=messages,
                n_bboxes=len(ex.bboxes),
                max_retries=max_retries,
                seed=per_image_seed + r,
            )
            final_parsed = parsed
            final_meta = meta

            out_path = vis_root / f"{Path(file_name).stem}_round{r}.png"
            render_one_prediction(
                poster_image_path=poster_path,
                ann_list=ann_list,
                reading_order=parsed.reading_order,
                tree=parsed.tree,
                out_path=out_path,
            )

            log_record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "run_name": run_name,
                "model": model,
                "file_name": file_name,
                "round": r,
                "parse_success": meta.get("parse_success"),
                "attempts": meta.get("attempts"),
                "is_fallback": meta.get("is_fallback"),
                "validate_success": meta.get("validate_success"),
                "failed_stage": meta.get("failed_stage", None),
                "validate_errors": meta.get("validate_errors", [])[:5],
                "reading_order": parsed.reading_order,
                "tree": parsed.tree,
                "vis_path": str(out_path.relative_to(output_root)),
            }
            log_f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            log_f.flush()

            prev_pred = {"reading_order": parsed.reading_order, "tree": parsed.tree}
            prev_vis_path = out_path

        if final_parsed is None or final_meta is None:
            raise RuntimeError(f"Unexpected: missing final result for file_name={file_name}")

    log_f.close()

    predictions = _load_final_predictions_from_log(log_path, run_name, num_rounds, allowed_files)
    if not predictions:
        print("No completed predictions; skipping pred COCO build/eval.")
        return

    missing_eval_ex = sorted(set(predictions.keys()) - set(ex_by_fname.keys()))
    if missing_eval_ex:
        raise ValueError(f"No Example found for predicted file_name(s): {missing_eval_ex}")

    eval_examples = [ex_by_fname[fname] for fname in sorted(predictions.keys())]

    build_pred_coco_from_dataset(
        examples=eval_examples,
        predictions=predictions,
        gt_json=gt_json,
        out_json=pred_json,
    )

    results = evaluate_trees(gt_json, pred_json)
    print(f"Results for model: {model}")
    print("  Number of images:", results["num_images"])
    print("  Mean TED:   ", results["mean_ted"])
    print("  Mean STEDS: ", results["mean_steds"])
    print("  Mean REDS:  ", results["mean_reds"])

    eval_out = {
        "run_name": run_name,
        "model": model,
        "num_rounds": num_rounds,
        "k_fewshot": k_fewshot,
        "seed": seed,
        "num_images": results["num_images"],
        "mean_ted": results["mean_ted"],
        "mean_steds": results["mean_steds"],
        "mean_reds": results["mean_reds"],
    }

    eval_path = output_root / run_name / "eval.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_out, f, ensure_ascii=False, indent=2)
