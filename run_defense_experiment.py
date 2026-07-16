#!/usr/bin/env python
"""Defense experiment runner.

Pipeline:  raw CSV
             -> optional fake-record injection   (--strategy)
             -> ARX-LR k-anonymization           (builtin_lr, or --skip-arx)
             -> CRA evaluation                   (cra/ library)
             -> metrics report

Single 2-QI experiment (preferred defense):
    cra/.venv/bin/python run_defense_experiment.py \
        --input dataset/min1/example-raw-dataset.csv \
        --output-dir out/defense_2qi_k5_sparse \
        --qi-cols "Age,Systolic Blood Pressure" \
        --k 5 --strategy sparse_basic_noise \
        --noise-budget 20 --max-noise-per-segment 2 \
        --seed 0 --cra-timeout-seconds 30

Baseline (no noise):
    cra/.venv/bin/python run_defense_experiment.py \
        --input dataset/min1/example-raw-dataset.csv \
        --output-dir out/baseline_2qi_k5 \
        --qi-cols "Age,Systolic Blood Pressure" \
        --k 5 --strategy no_noise --seed 0 --cra-timeout-seconds 30

Validate against real ARX output instead of the builtin anonymizer:
    ... --strategy no_noise --skip-arx \
        --anonymized dataset/min1/anonymized_2qi_k5.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

# Make `defense/` and `cra/` importable regardless of CWD.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from defense.lr_anonymizer import anonymize_qi
from defense.metrics import run_cra_eval
from defense.noise_injector import MARKER, STRATEGIES, inject
from defense.segment_utils import load_hierarchies


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("defense")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _parse_budget(spec: str, n_real: int) -> int:
    """Accept an int ('20') or a percentage of dataset size ('5%')."""
    spec = str(spec).strip()
    if spec.endswith("%"):
        return round(float(spec[:-1]) / 100.0 * n_real)
    return int(spec)


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        header = list(r.fieldnames or [])
    return header, rows


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _anonymize_builtin(
    rows: list[dict], header: list[str], qi_cols: list[str], hiers, k: int
) -> list[dict]:
    records_qi = [[r[q] for q in qi_cols] for r in rows]
    gen = anonymize_qi(records_qi, hiers, k)
    out: list[dict] = []
    for r, g in zip(rows, gen):
        nr = dict(r)
        for q, val in zip(qi_cols, g):
            nr[q] = val
        out.append(nr)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path, help="Raw dataset CSV.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--qi-cols", required=True,
                   help="Comma-separated QI column names, e.g. "
                        "'Age,Systolic Blood Pressure'.")
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--hierarchy-dir", type=Path, default=None,
                   help="Directory of hierarchy_<qi>.csv files. Defaults to "
                        "<input dir>/hierarchies.")
    p.add_argument("--strategy", choices=STRATEGIES, default="no_noise")
    p.add_argument("--noise-budget", default="0",
                   help="Total fake records. Int ('20') or percent of dataset "
                        "size ('5%%'). Default 0.")
    p.add_argument("--max-noise-per-segment", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-eqs", type=int, default=None,
                   help="Evaluate only the first N EQs (3-QI speed control).")
    p.add_argument("--cra-timeout-seconds", type=float, default=None,
                   help="Per-EQ CP-SAT wall-clock limit.")
    p.add_argument("--max-solutions-per-eq", type=int, default=None,
                   help="Cap CP-SAT enumeration per EQ (recommended for m>=4).")
    p.add_argument("--backend", choices=["builtin_lr", "external"],
                   default="builtin_lr")
    p.add_argument("--arx-cmd", default=None,
                   help="External anonymizer command template with {input} "
                        "{output} {k} placeholders (backend=external).")
    p.add_argument("--skip-arx", action="store_true",
                   help="Anonymized output already exists; skip injection + "
                        "anonymization and evaluate it directly.")
    p.add_argument("--anonymized", type=Path, default=None,
                   help="Pre-anonymized CSV to use with --skip-arx (e.g. a real "
                        "ARX export). Defaults to "
                        "<output-dir>/anonymized_output.csv.")
    p.add_argument("--reuse-existing", action="store_true",
                   help="Reuse anonymized_output.csv in the output dir if it "
                        "exists instead of recomputing.")
    args = p.parse_args()

    qi_cols = [c.strip() for c in args.qi_cols.split(",") if c.strip()]
    hier_dir = args.hierarchy_dir or (args.input.parent / "hierarchies")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(out_dir / "run.log")

    t_total0 = perf_counter()
    log.info("=== defense run: strategy=%s k=%s qi=%s seed=%s ===",
             args.strategy, args.k, qi_cols, args.seed)

    hiers = load_hierarchies(qi_cols, hier_dir)
    anon_path = out_dir / "anonymized_output.csv"

    injection_seconds = 0.0
    arx_seconds = 0.0
    n_real = 0
    n_fake = 0
    target_segments: list = []

    if args.skip_arx:
        src = args.anonymized or anon_path
        if not Path(src).exists():
            log.error("skip-arx: anonymized file not found: %s", src)
            raise SystemExit(2)
        if Path(src) != anon_path:
            shutil.copy(src, anon_path)
        log.info("skip-arx: evaluating existing anonymized file %s", src)
        # n_real / n_fake best-effort from sidecar if present.
        fr = out_dir / "fake_records.csv"
        if fr.exists():
            _, frrows = _read_csv(fr)
            n_fake = len(frrows)
        _, arows = _read_csv(anon_path)
        n_real = len(arows) - n_fake
    else:
        header, raw_rows = _read_csv(args.input)
        n_real = len(raw_rows)
        shutil.copy(args.input, out_dir / "raw_input.csv")
        budget = _parse_budget(args.noise_budget, n_real)
        log.info("loaded %d raw records; noise_budget resolved to %d", n_real, budget)

        # --- Stage 1: injection ---
        t0 = perf_counter()
        result = inject(
            rows=raw_rows, header=header, qi_cols=qi_cols, hiers=hiers,
            strategy=args.strategy, k=args.k, noise_budget=budget,
            max_noise_per_segment=args.max_noise_per_segment, seed=args.seed,
        )
        injection_seconds = perf_counter() - t0
        n_fake = result.n_fake
        target_segments = result.target_segments
        log.info("injection: strategy=%s added %d fake records into %d segments "
                 "(%.3fs)", args.strategy, n_fake, len(target_segments),
                 injection_seconds)

        _write_csv(out_dir / "noised_input.csv", result.header, result.rows)
        _write_csv(out_dir / "fake_records.csv", result.header, result.fake_rows)

        # --- Stage 2: anonymization ---
        reused = args.reuse_existing and anon_path.exists()
        if reused:
            log.info("reuse-existing: keeping %s", anon_path)
        else:
            t0 = perf_counter()
            if args.backend == "external":
                if not args.arx_cmd:
                    raise SystemExit("backend=external requires --arx-cmd")
                cmd = args.arx_cmd.format(
                    input=out_dir / "noised_input.csv", output=anon_path, k=args.k)
                log.info("external anonymizer: %s", cmd)
                subprocess.run(cmd, shell=True, check=True)
            else:
                anon_rows = _anonymize_builtin(
                    result.rows, result.header, qi_cols, hiers, args.k)
                _write_csv(anon_path, result.header, anon_rows)
            arx_seconds = perf_counter() - t0
            log.info("anonymization (%s): %.3fs -> %s",
                     args.backend, arx_seconds, anon_path)

    # --- Stage 3: CRA evaluation ---
    t0 = perf_counter()
    summary, per_eq = run_cra_eval(
        anon_path, hiers, args.k, qi_columns=qi_cols,
        max_solutions=args.max_solutions_per_eq, max_eqs=args.max_eqs,
        cra_timeout_seconds=args.cra_timeout_seconds,
    )
    cra_seconds = perf_counter() - t0
    log.info("CRA: %d EQs evaluated, %d exact, single-out proxy=%.1f%% (%.3fs)",
             summary["eqs_evaluated"], summary["eqs_exact"],
             summary["single_out_risk_proxy"], cra_seconds)

    total_seconds = perf_counter() - t_total0
    fake_pct = round(100.0 * n_fake / n_real, 4) if n_real else 0.0

    run_config = {
        "dataset_name": args.input.name,
        "input_path": str(args.input),
        "qi_cols": qi_cols,
        "k": args.k,
        "strategy": args.strategy,
        "noise_budget_arg": args.noise_budget,
        "max_noise_per_segment": args.max_noise_per_segment,
        "seed": args.seed,
        "backend": ("external_or_skip" if args.skip_arx else args.backend),
        "skip_arx": args.skip_arx,
        "max_eqs": args.max_eqs,
        "cra_timeout_seconds": args.cra_timeout_seconds,
        "max_solutions_per_eq": args.max_solutions_per_eq,
        "hierarchy_dir": str(hier_dir),
        "target_segments": ["|".join(s) for s in target_segments],
    }

    timings = {
        "injection_seconds": round(injection_seconds, 4),
        "arx_seconds": round(arx_seconds, 4),
        "cra_seconds": round(cra_seconds, 4),
        "total_seconds": round(total_seconds, 4),
    }

    metrics_summary = {
        "dataset_name": args.input.name,
        "qi_cols": qi_cols,
        "k": args.k,
        "strategy": args.strategy,
        "seed": args.seed,
        "n_original_records": n_real,
        "n_fake_records": n_fake,
        "fake_record_pct": fake_pct,
        **summary,
        **timings,
    }

    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (out_dir / "cra_results.json").write_text(json.dumps(
        {"summary": summary, "per_eq": per_eq}, indent=2))
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))

    if per_eq:
        with (out_dir / "metrics_per_eq.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_eq[0].keys()))
            w.writeheader()
            w.writerows(per_eq)

    log.info("DONE strategy=%s k=%s fakes=%d (%.2f%%) | exact_eq%%=%.1f "
             "single_out%%=%.1f cra_ratio_med=%s | total=%.2fs",
             args.strategy, args.k, n_fake, fake_pct,
             summary["exact_eq_pct"], summary["single_out_risk_proxy"],
             summary["cra_ratio_median"], total_seconds)
    log.info("outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
