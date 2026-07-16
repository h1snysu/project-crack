#!/usr/bin/env python
"""2-QI defense sweep driver.

Runs run_defense_experiment.py across strategies x k x noise-budget x seed,
then aggregates every run's metrics_summary.json into:
    out/defense_summary.csv
    out/defense_summary.md

Defaults match the project's 2-QI experiment plan:
    strategies = no_noise, random_noise, sparse_basic_noise
    k          = 3, 4, 5
    budgets    = 0, 1%, 2%, 5%   (of dataset size)
    seeds      = 0, 1, 2
    cra timeout= 30 s/EQ

Dedup rules: no_noise is run once per (k, seed) at budget 0 and serves as the
budget=0 point for the noise strategies; noise strategies run only the >0
budgets. Run from project root:

    cra/.venv/bin/python scripts/sweep_2qi_defense.py
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(_ROOT / "cra/.venv/bin/python")
RUNNER = str(_ROOT / "run_defense_experiment.py")

# Columns surfaced in the trimmed Markdown comparison table.
MD_COLS = [
    "strategy", "k", "noise_budget_label", "seed", "n_fake_records",
    "fake_record_pct", "eqs_evaluated", "exact_eq_pct",
    "cra_ratio_mean", "cra_ratio_median", "cra_ratio_max",
    "pct_eqs_ratio_le_1_05", "single_out_risk_proxy",
    "cra_int_assignments_median", "avg_info_loss_record_weighted",
    "eqs_timed_out", "cra_seconds",
]

GLOSSARY = {
    "n_fake_records": "fake records injected before anonymization",
    "fake_record_pct": "fakes as % of original records (utility cost)",
    "eqs_evaluated": "equivalence classes the CRA solved",
    "exact_eq_pct": "% EQs the attacker pins to a single solution (lower=better defense)",
    "cra_ratio_mean": "mean of LR_solutions/CRA_solutions; >=1, higher=stronger attack",
    "cra_ratio_median": "median LR/CRA ratio; closer to 1 = weaker attack",
    "cra_ratio_max": "worst-case single-EQ refinement",
    "pct_eqs_ratio_le_1_05": "% EQs where CRA barely beat the naive baseline (higher=better defense)",
    "single_out_risk_proxy": "% EQs with a CRA solution having some basic z_i=1 (FPSO proxy; lower=better)",
    "cra_int_assignments_median": "median # integer assignments per EQ (higher=more attacker uncertainty)",
    "avg_info_loss_record_weighted": "record-weighted mean info loss (utility cost; lower=better)",
    "eqs_timed_out": "EQs that hit the per-EQ CRA timeout",
    "cra_seconds": "CRA wall time for the run",
}


def run_one(out_dir: Path, input_csv: Path, qi: str, k: int, strategy: str,
            budget: str, seed: int, per_seg: int, timeout: float,
            max_solutions: int | None) -> dict | None:
    cmd = [
        PYTHON, RUNNER, "--input", str(input_csv), "--output-dir", str(out_dir),
        "--qi-cols", qi, "--k", str(k), "--strategy", strategy,
        "--noise-budget", budget, "--max-noise-per-segment", str(per_seg),
        "--seed", str(seed), "--cra-timeout-seconds", str(timeout),
    ]
    if max_solutions is not None:
        cmd += ["--max-solutions-per-eq", str(max_solutions)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    summ = out_dir / "metrics_summary.json"
    if r.returncode != 0 or not summ.exists():
        print(f"  !! FAILED {strategy} k={k} b={budget} s={seed}\n{r.stderr[-500:]}")
        return None
    data = json.loads(summ.read_text())
    data["noise_budget_label"] = budget
    data["run_dir"] = str(out_dir)
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=_ROOT / "dataset/min1/example-raw-dataset.csv")
    p.add_argument("--qi-cols", default="Age,Systolic Blood Pressure")
    p.add_argument("--out-root", type=Path, default=_ROOT / "out/sweep_2qi")
    p.add_argument("--summary-csv", type=Path, default=_ROOT / "out/defense_summary.csv")
    p.add_argument("--summary-md", type=Path, default=_ROOT / "out/defense_summary.md")
    p.add_argument("--strategies", default="no_noise,random_noise,sparse_basic_noise")
    p.add_argument("--k-values", default="3,4,5")
    p.add_argument("--budgets", default="0,1%,2%,5%")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max-noise-per-segment", type=int, default=2)
    p.add_argument("--cra-timeout-seconds", type=float, default=30.0)
    p.add_argument("--max-solutions-per-eq", type=int, default=None)
    p.add_argument("--plots", action="store_true", help="Also emit PNG plots.")
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    ks = [int(x) for x in args.k_values.split(",")]
    budgets = [b.strip() for b in args.budgets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = perf_counter()
    n_done = 0
    for k in ks:
        for seed in seeds:
            for strat in strategies:
                # Dedup: no_noise only at budget 0; noise strategies only >0.
                if strat == "no_noise":
                    strat_budgets = ["0"]
                else:
                    strat_budgets = [b for b in budgets if b not in ("0", "0%")]
                for b in strat_budgets:
                    blabel = b.replace("%", "pct")
                    od = args.out_root / f"{strat}_k{k}_b{blabel}_s{seed}"
                    res = run_one(od, args.input, args.qi_cols, k, strat, b, seed,
                                  args.max_noise_per_segment, args.cra_timeout_seconds,
                                  args.max_solutions_per_eq)
                    n_done += 1
                    if res is not None:
                        rows.append(res)
                        print(f"  ok [{n_done}] {strat:20s} k={k} b={b:3s} s={seed} "
                              f"exact%={res['exact_eq_pct']:.1f} "
                              f"single%={res['single_out_risk_proxy']:.1f} "
                              f"ratio_med={res['cra_ratio_median']} "
                              f"fakes={res['n_fake_records']}")
    elapsed = perf_counter() - t0
    print(f"\nsweep finished: {len(rows)} runs in {elapsed:.1f}s")

    if not rows:
        print("no successful runs; nothing to aggregate.")
        return

    # ---- aggregate CSV (full detail) ----
    all_keys: list[str] = []
    for r in rows:
        for kk in r:
            if kk not in all_keys:
                all_keys.append(kk)
    for r in rows:
        if isinstance(r.get("qi_cols"), list):
            r["qi_cols"] = "|".join(r["qi_cols"])
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.summary_csv}")

    # ---- trimmed Markdown ----
    def cell(r, c):
        v = r.get(c, "")
        return "" if v is None else str(v)

    rows_sorted = sorted(rows, key=lambda r: (r["k"], r["strategy"],
                                              str(r["noise_budget_label"]), r["seed"]))
    lines = ["# 2-QI Defense Sweep Summary", "",
             f"Runs: {len(rows)} | input: `{args.input.name}` | "
             f"QIs: {args.qi_cols} | timeout: {args.cra_timeout_seconds}s/EQ", "",
             "| " + " | ".join(MD_COLS) + " |",
             "|" + "|".join(["---"] * len(MD_COLS)) + "|"]
    for r in rows_sorted:
        lines.append("| " + " | ".join(cell(r, c) for c in MD_COLS) + " |")
    lines += ["", "## Metric glossary", ""]
    for kk, vv in GLOSSARY.items():
        lines.append(f"- **{kk}** — {vv}")
    args.summary_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.summary_md}")

    if args.plots:
        try:
            from defense.plots import make_plots
            sys.path.insert(0, str(_ROOT))
            out = make_plots(rows, args.summary_csv.parent)
            print(f"wrote plots: {out}")
        except Exception as e:  # pragma: no cover
            print(f"plots skipped: {e}")


if __name__ == "__main__":
    main()
