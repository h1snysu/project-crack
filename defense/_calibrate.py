"""Calibration: run builtin_lr (no noise) on the raw 2-QI data and compare its
CRA metrics against the real ARX export, to check the builtin anonymizer is a
faithful-enough stand-in. Run from project root:

    cra/.venv/bin/python -m defense._calibrate
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from defense.segment_utils import load_hierarchies
from defense.lr_anonymizer import anonymize_qi

# cra/ is on sys.path via defense/__init__.py
from eval_cra import evaluate_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset/min1/example-raw-dataset.csv"
HIER = ROOT / "dataset/min1/hierarchies"
REAL_ARX = ROOT / "dataset/min1/anonymized_2qi_k5.csv"
QI = ["Age", "Systolic Blood Pressure"]
K = 5


def _fmt(r):
    return (
        f"n={r.n} eqs={r.eqs_merged} |O|={r.outlier_count} "
        f"exactEQ%={r.exact_eq_pct:.1f} exactRec%={r.exact_record_pct:.1f} "
        f"mean_sol={r.solvable_mean:.2f} med_sol={r.solvable_median} "
        f"max_sol={r.solvable_max}"
    )


def main() -> None:
    with RAW.open(newline="") as f:
        rows = list(csv.DictReader(f))
    hiers = load_hierarchies(QI, HIER)
    records_qi = [[row[q] for q in QI] for row in rows]
    gen = anonymize_qi(records_qi, hiers, K)

    out = ROOT / "out" / "_calib_builtin_2qi_k5.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PATIENT"] + QI)
        for i, g in enumerate(gen):
            w.writerow([rows[i].get("PATIENT", f"r{i}")] + list(g))

    print("== builtin_lr (no noise) ==")
    rb = evaluate_dataset(out, HIER, K, qi_columns=QI)
    print("  ", _fmt(rb))

    print("== real ARX export ==")
    ra = evaluate_dataset(REAL_ARX, HIER, K, qi_columns=QI)
    print("  ", _fmt(ra))

    # Quick structural sanity.
    n_supp = sum(1 for g in gen if g[0] == "*")
    print(f"\nbuiltin suppressed rows: {n_supp}")
    print(f"builtin output written: {out}")


if __name__ == "__main__":
    main()
