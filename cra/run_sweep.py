"""Run the CRA attack across many anonymised datasets and emit one comparison
table in CSV + Markdown form.

The driver discovers datasets by scanning `--input-dir` for files matching
`anonymized_<m>qi_k<k>.csv` (the ARX-LR output naming convention used in the
project's dataset/min1 layout). For each match, it parses `m` and `k` from
the filename and calls `evaluate_dataset()` from `eval_cra.py`.

Examples (run from the cra/ directory):
    # Sweep every anonymized_*qi_k*.csv under dataset/min1, with a per-EQ cap
    # of 10000 (needed for m=4 datasets):
    python run_sweep.py \\
        --input-dir ../dataset/min1 \\
        --hier-dir ../dataset/min1/hierarchies \\
        --out-csv sweep.csv --out-md sweep.md \\
        --max-solutions 10000

    # Just the 2-QI and 3-QI files (skip 4-QI):
    python run_sweep.py \\
        --files ../dataset/min1/anonymized_2qi_k5.csv \\
                ../dataset/min1/anonymized_3qi_k5.csv \\
        --hier-dir ../dataset/min1/hierarchies \\
        --out-csv quick.csv --out-md quick.md
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from eval_cra import EvalResult, evaluate_dataset


# anonymized_2qi_k5.csv  ->  m=2, k=5
FILENAME_RE = re.compile(
    r"^anonymized_(?P<m>\d+)qi_k(?P<k>\d+)\.csv$", re.IGNORECASE
)


def discover_files(input_dir: Path) -> list[tuple[Path, int]]:
    """Return [(path, k)] for every anonymized_*qi_k*.csv in `input_dir`."""
    out: list[tuple[Path, int]] = []
    for p in sorted(input_dir.iterdir()):
        m = FILENAME_RE.match(p.name)
        if m is not None:
            out.append((p, int(m.group("k"))))
    return out


# CSV column order. Keep this stable; downstream tooling may parse it.
CSV_COLUMNS = [
    "dataset_file", "m", "k", "n",
    "grid_basics", "eqs_pre_merge", "eqs_merged", "outlier_count",
    "eqs_infeasible", "eqs_exact", "eqs_highrisk", "eqs_uncertain",
    "eqs_capped", "max_solutions_cap",
    "exact_eq_pct", "highrisk_eq_pct", "uncertain_eq_pct",
    "records_in_eqs", "records_exact", "records_highrisk", "records_uncertain",
    "exact_record_pct", "highrisk_record_pct", "uncertain_record_pct",
    "cra_ratio_mean", "cra_ratio_median", "cra_ratio_min", "cra_ratio_max",
    "eqs_trivially_unique", "eqs_refined_to_unique",
    "outlier_free_basics", "outlier_closed_form",
    "outlier_enum_skipped", "outlier_solutions_enumerated",
    "wall_time_alg3", "wall_time_alg4", "wall_time_total",
]


def result_to_row(r: EvalResult) -> dict:
    """Flatten an EvalResult into a CSV-friendly dict (computed % included)."""
    d = asdict(r)
    # Drop the QI-name list -- it isn't a single CSV cell.
    d.pop("qi_names", None)
    # Computed properties.
    d["exact_eq_pct"] = round(r.exact_eq_pct, 2)
    d["highrisk_eq_pct"] = round(r.highrisk_eq_pct, 2)
    d["uncertain_eq_pct"] = round(r.uncertain_eq_pct, 2)
    d["exact_record_pct"] = round(r.exact_record_pct, 2)
    d["highrisk_record_pct"] = round(r.highrisk_record_pct, 2)
    d["uncertain_record_pct"] = round(r.uncertain_record_pct, 2)
    d["wall_time_total"] = round(r.wall_time_total, 2)
    # Round floats for compactness.
    for f in ("cra_ratio_mean", "cra_ratio_median",
              "cra_ratio_min", "cra_ratio_max",
              "solvable_mean", "solvable_median",
              "wall_time_alg3", "wall_time_alg4"):
        v = d.get(f)
        if isinstance(v, float):
            d[f] = round(v, 4)
    return d


# --- Markdown rendering -----------------------------------------------------

# Markdown column: (header, key, format_spec).
# format_spec values: "int" / "pct1" (one-decimal percent) / "ratio" (4 dp).
MD_COLUMNS: list[tuple[str, str, str]] = [
    ("File",                 "dataset_file",    "str"),
    ("m",                    "m",               "int"),
    ("k",                    "k",               "int"),
    ("n",                    "n",               "int"),
    ("EQs",                  "eqs_merged",      "int"),
    ("Outliers",             "outlier_count",   "int"),
    ("Exact EQ%",            "exact_eq_pct",    "pct1"),
    ("Exact Records%",       "exact_record_pct","pct1"),
    ("High-Risk EQ%",        "highrisk_eq_pct", "pct1"),
    ("Uncertain EQ%",        "uncertain_eq_pct","pct1"),
    ("CRA Ratio (mean)",     "cra_ratio_mean",  "ratio"),
    ("Capped EQs",           "eqs_capped",      "int"),
    ("Wall (s)",             "wall_time_total", "float2"),
]


def _format_cell(value, spec: str) -> str:
    if value is None:
        return ""
    if spec == "int":
        return str(int(value))
    if spec == "pct1":
        return f"{float(value):.1f}"
    if spec == "ratio":
        return f"{float(value):.4f}"
    if spec == "float2":
        return f"{float(value):.2f}"
    return str(value)


def render_markdown(rows: list[dict], cap: int | None) -> str:
    headers = [h for h, _, _ in MD_COLUMNS]
    lines: list[str] = []
    lines.append("# CRA Attack Sweep")
    lines.append("")
    if cap is not None:
        lines.append(
            f"_Per-EQ enumeration cap: **{cap:,}**. EQs marked 'Capped' have "
            f"reached the cap; their reported solution counts are lower bounds._"
        )
        lines.append("")
    lines.append("## Comparison table")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [_format_cell(row.get(key), spec) for _, key, spec in MD_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Metric glossary")
    lines.append("")
    lines.append(
        "- **EQs** -- number of equivalence classes after snap-merge "
        "(grid-footprint collapse)."
    )
    lines.append(
        "- **|O|** -- number of fully-suppressed records (outliers)."
    )
    lines.append(
        "- **Exact EQ%** -- fraction of EQs whose CRA LP yields exactly one "
        "integer solution. Each such EQ is a 100%-confidence breach."
    )
    lines.append(
        "- **Exact Records%** -- fraction of records living in those exact-"
        "breach EQs (records weighted by |EQ|)."
    )
    lines.append(
        "- **High-Risk EQ%** -- EQs with 2-5 solutions (>= 20% blind-guess "
        "probability for the attacker)."
    )
    lines.append(
        "- **Uncertain EQ%** -- EQs with > 5 solutions. Includes any EQ that "
        "hit the per-EQ cap (column **Capped EQs**)."
    )
    lines.append(
        "- **CRA Ratio (mean)** -- mean of `(enumerated solutions) / "
        "(stars-and-bars baseline)` across solvable EQs. Closer to 0 means "
        "halves + sparse constraints cut more of the candidate space."
    )
    lines.append(
        "- **Wall (s)** -- total Alg 3 + Alg 4 time. The outlier LP uses a "
        "closed-form bypass when its solution count exceeds the threshold."
    )
    return "\n".join(lines) + "\n"


# --- CLI --------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("dataset source (give one of --input-dir / --files)")
    src.add_argument(
        "--input-dir", type=Path, default=None, metavar="DIR",
        help="Scan DIR for files matching anonymized_<m>qi_k<k>.csv and "
             "evaluate each.",
    )
    src.add_argument(
        "--files", nargs="+", type=Path, default=None, metavar="CSV",
        help="Explicit list of anonymised CSVs to evaluate. m and k are "
             "parsed from each filename.",
    )

    req = p.add_argument_group("required")
    req.add_argument(
        "--hier-dir", required=True, type=Path, metavar="DIR",
        help="Directory containing the generalisation-hierarchy CSVs.",
    )

    out = p.add_argument_group("outputs")
    out.add_argument(
        "--out-csv", type=Path, default=Path("sweep_results.csv"), metavar="CSV",
        help="Where to write the full CSV table. Default: sweep_results.csv",
    )
    out.add_argument(
        "--out-md", type=Path, default=Path("sweep_results.md"), metavar="MD",
        help="Where to write the Markdown summary. Default: sweep_results.md",
    )

    opt = p.add_argument_group("optional")
    opt.add_argument(
        "--max-solutions", type=int, default=10_000, metavar="N",
        help="Per-EQ enumeration cap forwarded to eval_cra.evaluate_dataset. "
             "Default 10000 (keeps 4-QI runs tractable; set to 0 to disable).",
    )
    opt.add_argument(
        "--outlier-enum-threshold", type=int, default=200_000, metavar="N",
        help="Closed-form bypass threshold for the outlier LP. Default 200000.",
    )
    opt.add_argument(
        "--max-outlier-solutions", type=int, default=None, metavar="N",
        help="Cap CP-SAT enumeration for the outlier LP. Default: unlimited.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.input_dir is None and args.files is None:
        sys.exit("Specify either --input-dir or --files (see --help).")

    if args.files is not None:
        datasets: list[tuple[Path, int]] = []
        for p in args.files:
            m = FILENAME_RE.match(p.name)
            if m is None:
                sys.exit(
                    f"Filename {p.name!r} does not match "
                    f"anonymized_<m>qi_k<k>.csv; rename it or use --input-dir."
                )
            datasets.append((p, int(m.group("k"))))
    else:
        if not args.input_dir.exists():
            sys.exit(f"--input-dir {args.input_dir} does not exist.")
        datasets = discover_files(args.input_dir)
        if not datasets:
            sys.exit(
                f"No files matching anonymized_<m>qi_k<k>.csv in "
                f"{args.input_dir}."
            )

    cap = args.max_solutions if args.max_solutions > 0 else None

    print(f"Evaluating {len(datasets)} dataset(s) with per-EQ cap={cap}.")
    rows: list[dict] = []
    total_t0 = perf_counter()
    for path, k in datasets:
        print(f"  --> {path.name}  (k={k}) ...", flush=True)
        t0 = perf_counter()
        result = evaluate_dataset(
            dataset_path=path,
            hier_dir=args.hier_dir,
            k=k,
            max_solutions=cap,
            max_outlier_solutions=args.max_outlier_solutions,
            outlier_enum_threshold=args.outlier_enum_threshold,
        )
        dt = perf_counter() - t0
        print(
            f"      done in {dt:.1f}s | "
            f"EQs={result.eqs_merged} |O|={result.outlier_count} | "
            f"Exact EQ%={result.exact_eq_pct:.1f} | "
            f"CRA ratio mean={result.cra_ratio_mean if result.cra_ratio_mean is not None else 'n/a'}",
            flush=True,
        )
        rows.append(result_to_row(result))
    print(f"Total sweep wall time: {perf_counter() - total_t0:.1f}s")

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {args.out_csv}")

    # Write Markdown
    md = render_markdown(rows, cap=cap)
    args.out_md.write_text(md)
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
