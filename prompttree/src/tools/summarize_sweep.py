import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict


RUN_DIR_RE = re.compile(r"^(?P<model>.+)_r(?P<r>\d+)_k(?P<k>\d+)_s(?P<s>-?\d+)$")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL at {path} line {lineno}: {e}") from e


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    model: str
    num_rounds: int
    k_fewshot: int
    seed: int
    run_dir: Path
    log_path: Path
    eval_path: Path


@dataclass
class Agg:
    n_images: int = 0
    n_fallback: int = 0
    n_validate_ok: int = 0
    n_validate_known: int = 0
    attempts_sum: int = 0

    # eval aggregation (mean over runs, weighted by num_images)
    eval_images_sum: int = 0
    ted_sum: float = 0.0
    steds_sum: float = 0.0
    reds_sum: float = 0.0
    eval_runs: int = 0

    def add_log_final(self, attempts: int, is_fallback: bool, validate_success: Optional[bool]) -> None:
        self.n_images += 1
        if is_fallback:
            self.n_fallback += 1
        if validate_success is True:
            self.n_validate_ok += 1
        if validate_success is not None:
            self.n_validate_known += 1
        self.attempts_sum += attempts

    def add_eval(self, num_images: int, mean_ted: float, mean_steds: float, mean_reds: float) -> None:
        # weighted by num_images
        self.eval_images_sum += int(num_images)
        self.ted_sum += float(mean_ted) * int(num_images)
        self.steds_sum += float(mean_steds) * int(num_images)
        self.reds_sum += float(mean_reds) * int(num_images)
        self.eval_runs += 1

    def fallback_rate(self) -> float:
        return self.n_fallback / self.n_images if self.n_images else 0.0

    def validate_rate(self) -> Optional[float]:
        if self.n_validate_known == 0:
            return None
        return self.n_validate_ok / self.n_validate_known

    def avg_attempts(self) -> float:
        return self.attempts_sum / self.n_images if self.n_images else 0.0

    def has_eval(self) -> bool:
        return self.eval_runs > 0 and self.eval_images_sum > 0

    def mean_ted_weighted(self) -> Optional[float]:
        if not self.has_eval():
            return None
        return self.ted_sum / self.eval_images_sum

    def mean_steds_weighted(self) -> Optional[float]:
        if not self.has_eval():
            return None
        return self.steds_sum / self.eval_images_sum

    def mean_reds_weighted(self) -> Optional[float]:
        if not self.has_eval():
            return None
        return self.reds_sum / self.eval_images_sum


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:6.2f}%"


def fmt_f(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:8.3f}"


def print_table(rows: List[List[str]]) -> None:
    if not rows:
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for i, r in enumerate(rows):
        line = "  ".join(r[j].ljust(widths[j]) for j in range(len(r)))
        print(line)
        if i == 0:
            print("  ".join("-" * widths[j] for j in range(len(r))))


def parse_run_dir_name(dir_name: str) -> Optional[Tuple[str, int, int, int]]:
    m = RUN_DIR_RE.match(dir_name)
    if not m:
        return None
    model = m.group("model")
    r = int(m.group("r"))
    k = int(m.group("k"))
    s = int(m.group("s"))
    return model, r, k, s


def discover_runs(output_root: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    if not output_root.is_dir():
        raise FileNotFoundError(f"output_root not found: {output_root}")

    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_run_dir_name(child.name)
        if parsed is None:
            continue
        model, r, k, s = parsed

        log_path = child / "log.jsonl"
        eval_path = child / "eval.json"
        if not log_path.is_file():
            continue

        runs.append(
            RunInfo(
                run_name=child.name,
                model=model,
                num_rounds=r,
                k_fewshot=k,
                seed=s,
                run_dir=child,
                log_path=log_path,
                eval_path=eval_path,
            )
        )
    return runs


def load_final_rows(run: RunInfo) -> Dict[str, Dict[str, Any]]:
    finals: Dict[str, Dict[str, Any]] = {}
    for rec in iter_jsonl(run.log_path):
        if rec.get("run_name") != run.run_name:
            continue
        if rec.get("round") != run.num_rounds:
            continue
        fname = rec.get("file_name")
        if not isinstance(fname, str) or not fname:
            continue
        finals[fname] = rec
    return finals


def load_eval(run: RunInfo) -> Optional[Dict[str, Any]]:
    if not run.eval_path.is_file():
        return None
    with run.eval_path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True, help="e.g. output/")
    ap.add_argument("--include", type=str, default=None, help="optional regex to include run_name")
    ap.add_argument("--exclude", type=str, default=None, help="optional regex to exclude run_name")
    ap.add_argument("--require-eval", action="store_true", help="only include runs that have eval.json")
    args = ap.parse_args()

    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    runs = discover_runs(args.output_root)
    if include_re:
        runs = [r for r in runs if include_re.search(r.run_name)]
    if exclude_re:
        runs = [r for r in runs if not exclude_re.search(r.run_name)]

    if args.require_eval:
        runs = [r for r in runs if r.eval_path.is_file()]

    if not runs:
        print("No runs found under output_root (expected: {model}_r{R}_k{K}_s{S}/log.jsonl).")
        return

    bucket_agg: Dict[Tuple[str, int, int], Agg] = defaultdict(Agg)
    bucket_runs: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)

    per_run_rows: List[List[str]] = [
        [
            "run_name",
            "model",
            "num_rounds",
            "k_fewshot",
            "seed",
            "images",
            "fallback_rate",
            "validate_rate",
            "avg_attempts",
            "mean_TED",
            "mean_STEDS",
            "mean_REDS",
        ]
    ]

    for run in runs:
        finals = load_final_rows(run)
        a = Agg()
        for _, rec in finals.items():
            attempts = rec.get("attempts", 0)
            try:
                attempts_i = int(attempts)
            except Exception:
                attempts_i = 0

            is_fallback = bool(rec.get("is_fallback", False))
            vs = rec.get("validate_success", None)
            validate_success: Optional[bool] = vs if isinstance(vs, bool) else None

            a.add_log_final(attempts_i, is_fallback, validate_success)

        ev = load_eval(run)
        mean_ted = mean_steds = mean_reds = None
        if isinstance(ev, dict):
            try:
                nimg = int(ev.get("num_images"))
                mean_ted = float(ev.get("mean_ted"))
                mean_steds = float(ev.get("mean_steds"))
                mean_reds = float(ev.get("mean_reds"))
                a.add_eval(nimg, mean_ted, mean_steds, mean_reds)
            except Exception:
                mean_ted = mean_steds = mean_reds = None

        key = (run.model, run.num_rounds, run.k_fewshot)
        bucket_agg[key].n_images += a.n_images
        bucket_agg[key].n_fallback += a.n_fallback
        bucket_agg[key].n_validate_ok += a.n_validate_ok
        bucket_agg[key].n_validate_known += a.n_validate_known
        bucket_agg[key].attempts_sum += a.attempts_sum
        bucket_agg[key].eval_images_sum += a.eval_images_sum
        bucket_agg[key].ted_sum += a.ted_sum
        bucket_agg[key].steds_sum += a.steds_sum
        bucket_agg[key].reds_sum += a.reds_sum
        bucket_agg[key].eval_runs += a.eval_runs

        bucket_runs[key].append(run.run_name)

        per_run_rows.append(
            [
                run.run_name,
                run.model,
                str(run.num_rounds),
                str(run.k_fewshot),
                str(run.seed),
                str(a.n_images),
                fmt_pct(a.fallback_rate()),
                fmt_pct(a.validate_rate()),
                fmt_f(a.avg_attempts()),
                fmt_f(mean_ted),
                fmt_f(mean_steds),
                fmt_f(mean_reds),
            ]
        )

    rows: List[List[str]] = [
        [
            "model",
            "num_rounds",
            "k_fewshot",
            "runs",
            "images",
            "fallback_rate",
            "validate_rate",
            "avg_attempts",
            "mean_TED",
            "mean_STEDS",
            "mean_REDS",
        ]
    ]

    def sort_key(k: Tuple[str, int, int]) -> Tuple[str, int, int]:
        return (k[0], k[1], k[2])

    for key in sorted(bucket_agg.keys(), key=sort_key):
        model, r, k = key
        a = bucket_agg[key]
        rows.append(
            [
                model,
                str(r),
                str(k),
                str(len(bucket_runs[key])),
                str(a.n_images),
                fmt_pct(a.fallback_rate()),
                fmt_pct(a.validate_rate()),
                fmt_f(a.avg_attempts()),
                fmt_f(a.mean_ted_weighted()),
                fmt_f(a.mean_steds_weighted()),
                fmt_f(a.mean_reds_weighted()),
            ]
        )

    print("== Sweep summary (final round only; bucketed by model/num_rounds/k_fewshot) ==")
    print_table(rows)
    print()
    print("== Per run (final round only) ==")
    print_table(per_run_rows)


if __name__ == "__main__":
    main()
