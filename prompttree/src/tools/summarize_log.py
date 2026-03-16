import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Agg:
    n: int = 0
    parse_success: int = 0
    validate_success: int = 0
    is_fallback: int = 0
    attempts_sum: int = 0
    attempts_hist: Counter = None  # type: ignore

    def __post_init__(self) -> None:
        if self.attempts_hist is None:
            self.attempts_hist = Counter()

    def add(self, rec: Dict[str, Any]) -> None:
        self.n += 1
        ps = bool(rec.get("parse_success", False))
        vs = rec.get("validate_success", None)
        fb = bool(rec.get("is_fallback", False))
        att = int(rec.get("attempts", 0) or 0)

        if ps:
            self.parse_success += 1
        if vs is True:
            self.validate_success += 1
        if fb:
            self.is_fallback += 1
        self.attempts_sum += att
        if att > 0:
            self.attempts_hist[att] += 1

    def rates(self) -> Dict[str, float]:
        n = self.n if self.n else 1
        return {
            "parse_success_rate": self.parse_success / n,
            "validate_success_rate": self.validate_success / n,
            "fallback_rate": self.is_fallback / n,
            "avg_attempts": self.attempts_sum / n,
        }


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL at line {lineno}: {e}") from e


def _fmt_rate(x: float) -> str:
    return f"{x*100:6.2f}%"


def _fmt_float(x: float) -> str:
    return f"{x:8.3f}"


def print_table(rows: List[List[str]]) -> None:
    if not rows:
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for ridx, r in enumerate(rows):
        line = "  ".join(r[i].ljust(widths[i]) for i in range(len(r)))
        print(line)
        if ridx == 0:
            print("  ".join("-" * widths[i] for i in range(len(r))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True, help="path to output_root/log.jsonl")
    ap.add_argument("--topk-errors", type=int, default=20, help="top-K validate errors to show")
    ap.add_argument("--show-attempts-hist", action="store_true", help="print attempts histogram")
    args = ap.parse_args()

    log_path: Path = args.log
    if not log_path.is_file():
        raise FileNotFoundError(f"log not found: {log_path}")

    records = list(iter_jsonl(log_path))
    if not records:
        print("No records.")
        return

    # Basic schema sanity (non-fatal)
    required = ["run_name", "model", "file_name", "round", "parse_success", "attempts", "is_fallback"]
    missing_keys = Counter()
    for r in records:
        for k in required:
            if k not in r:
                missing_keys[k] += 1
    if missing_keys:
        print("[WARN] Some required keys are missing in records:")
        for k, c in missing_keys.most_common():
            print(f"  {k}: missing in {c} records")
        print()

    overall = Agg()
    by_model: Dict[str, Agg] = defaultdict(Agg)
    by_model_round: Dict[Tuple[str, int], Agg] = defaultdict(Agg)

    # Per-image summary (group by file_name, take max round as "final")
    by_image_rounds: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = defaultdict(dict)  # (run_name,file)->round->rec

    validate_error_counter = Counter()
    failed_stage_counter = Counter()

    for rec in records:
        overall.add(rec)
        model = str(rec.get("model", "UNKNOWN"))
        rnd = int(rec.get("round", -1))
        by_model[model].add(rec)
        by_model_round[(model, rnd)].add(rec)

        run_name = str(rec.get("run_name", "UNKNOWN_RUN"))
        fname = str(rec.get("file_name", "UNKNOWN_FILE"))
        if rnd >= 0:
            by_image_rounds[(run_name, fname)][rnd] = rec

        # optional fields
        fs = rec.get("failed_stage", None)
        if fs is not None:
            failed_stage_counter[str(fs)] += 1

        verrs = rec.get("validate_errors", None)
        if isinstance(verrs, list):
            for e in verrs:
                if isinstance(e, str) and e:
                    validate_error_counter[e] += 1

    # Overall
    r = overall.rates()
    print("== Overall ==")
    print(f"Records: {overall.n}")
    print(f"Parse success rate:    {_fmt_rate(r['parse_success_rate'])}")
    if any(rec.get("validate_success", None) is not None for rec in records):
        print(f"Validate success rate: {_fmt_rate(r['validate_success_rate'])}")
    print(f"Fallback rate:         {_fmt_rate(r['fallback_rate'])}")
    print(f"Avg attempts:          {_fmt_float(r['avg_attempts'])}")
    print()

    # By model
    print("== By model ==")
    rows = [["model", "n", "parse_ok", "validate_ok", "fallback", "avg_attempts"]]
    for model in sorted(by_model.keys()):
        a = by_model[model]
        rr = a.rates()
        rows.append(
            [
                model,
                str(a.n),
                _fmt_rate(rr["parse_success_rate"]),
                _fmt_rate(rr["validate_success_rate"]) if any(rec.get("validate_success", None) is not None for rec in records) else "   n/a",
                _fmt_rate(rr["fallback_rate"]),
                _fmt_float(rr["avg_attempts"]),
            ]
        )
    print_table(rows)
    print()

    # By model x round
    rounds = sorted({int(rec.get("round", -1)) for rec in records if rec.get("round", -1) != -1})
    if rounds:
        print("== By model x round ==")
        rows = [["model", "round", "n", "parse_ok", "validate_ok", "fallback", "avg_attempts"]]
        for model in sorted(by_model.keys()):
            for rnd in rounds:
                a = by_model_round.get((model, rnd))
                if not a or a.n == 0:
                    continue
                rr = a.rates()
                rows.append(
                    [
                        model,
                        str(rnd),
                        str(a.n),
                        _fmt_rate(rr["parse_success_rate"]),
                        _fmt_rate(rr["validate_success_rate"]) if any(rec.get("validate_success", None) is not None for rec in records) else "   n/a",
                        _fmt_rate(rr["fallback_rate"]),
                        _fmt_float(rr["avg_attempts"]),
                    ]
                )
        print_table(rows)
        print()

    # Per-image "final" stats (max round per image)
    finals: List[Dict[str, Any]] = []
    for (run_name, fname), rmap in by_image_rounds.items():
        if not rmap:
            continue
        max_r = max(rmap.keys())
        finals.append(rmap[max_r])

    if finals:
        final_agg = Agg()
        for rec in finals:
            final_agg.add(rec)
        rr = final_agg.rates()
        print("== Final round per image (max round) ==")
        print(f"Images: {len(finals)}")
        print(f"Parse success rate:    {_fmt_rate(rr['parse_success_rate'])}")
        if any(rec.get("validate_success", None) is not None for rec in finals):
            print(f"Validate success rate: {_fmt_rate(rr['validate_success_rate'])}")
        print(f"Fallback rate:         {_fmt_rate(rr['fallback_rate'])}")
        print(f"Avg attempts:          {_fmt_float(rr['avg_attempts'])}")
        print()

    # Attempts histogram
    if args.show_attempts_hist:
        print("== Attempts histogram (all records) ==")
        for k in sorted(overall.attempts_hist.keys()):
            print(f"attempts={k}: {overall.attempts_hist[k]}")
        print()

    # Failed stage
    if failed_stage_counter:
        print("== failed_stage frequency ==")
        for k, v in failed_stage_counter.most_common():
            print(f"{k}: {v}")
        print()

    # Validate errors
    if validate_error_counter:
        print(f"== Top validate_errors (top {args.topk_errors}) ==")
        for e, c in validate_error_counter.most_common(args.topk_errors):
            print(f"{c:6d}  {e}")
        print()


if __name__ == "__main__":
    main()
