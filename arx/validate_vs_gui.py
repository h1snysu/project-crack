"""Validation: run the ARX-LR driver (and builtin_lr) on the existing raw data +
hand-made hierarchies for 2/3/4 QI at several k, and compare the multiset of
generalized QI tuples against the hand-made GUI exports. Identical multiset =>
same anonymization.

Run from project root:
    cra/.venv/bin/python arx/validate_vs_gui.py
"""
from __future__ import annotations

import csv
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset/min1/example-raw-dataset.csv"
HIER = ROOT / "dataset/min1/hierarchies"
RUN_ARX = ROOT / "arx/run_arx.sh"

import sys
sys.path.insert(0, str(ROOT))
from defense.segment_utils import load_hierarchies
from defense.lr_anonymizer import anonymize_qi

QISETS = {
    2: ["Age", "Systolic Blood Pressure"],
    3: ["Age", "Systolic Blood Pressure", "Heart rate"],
    4: ["Age", "Body Weight", "Systolic Blood Pressure", "Heart rate"],
}
HIERMAP = {
    "Age": "hierarchy_age.csv",
    "Body Weight": "hierarchy_body-weight.csv",
    "Systolic Blood Pressure": "hierarchy_systolic-blood-pressure.csv",
    "Heart rate": "hierarchy_heart-rate.csv",
}
KS = [3, 5, 8]


def read_qi_multiset(path: Path, qis: list[str]) -> Counter:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return Counter(tuple(r[q] for q in qis) for r in rows)


def run_driver(qis: list[str], k: int, out: Path) -> None:
    qi_pipe = "|".join(qis)
    hier_pipe = "|".join(HIERMAP[q] for q in qis)
    subprocess.run(
        ["bash", str(RUN_ARX), str(RAW), str(out), str(k), qi_pipe, hier_pipe, str(HIER)],
        check=True, capture_output=True, text=True,
    )


def builtin_multiset(qis: list[str], k: int) -> Counter:
    with RAW.open(newline="") as f:
        rows = list(csv.DictReader(f))
    hiers = load_hierarchies(qis, HIER)
    rec = [[r[q] for q in qis] for r in rows]
    gen = anonymize_qi(rec, hiers, k)
    return Counter(gen)


def diff(a: Counter, b: Counter) -> int:
    return sum((a - b).values()) + sum((b - a).values())


def main() -> None:
    outdir = ROOT / "out/validate"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{'set':>4} {'k':>2} | {'GUI EQs':>7} | {'ARX EQs':>7} {'ARXΔrec':>7} {'ARX==':>6} | "
          f"{'blt EQs':>7} {'bltΔrec':>7} {'blt==':>6}")
    print("-" * 78)
    for m, qis in QISETS.items():
        for k in KS:
            gui = ROOT / f"dataset/min1/anonymized_{m}qi_k{k}.csv"
            if not gui.exists():
                continue
            gui_ms = read_qi_multiset(gui, qis)
            # driver
            out = outdir / f"arx_{m}qi_k{k}.csv"
            run_driver(qis, k, out)
            arx_ms = read_qi_multiset(out, qis)
            # builtin
            blt_ms = builtin_multiset(qis, k)
            print(f"{m:>4} {k:>2} | {len(gui_ms):>7} | "
                  f"{len(arx_ms):>7} {diff(arx_ms, gui_ms):>7} "
                  f"{'YES' if arx_ms==gui_ms else 'no':>6} | "
                  f"{len(blt_ms):>7} {diff(blt_ms, gui_ms):>7} "
                  f"{'YES' if blt_ms==gui_ms else 'no':>6}")


if __name__ == "__main__":
    main()
