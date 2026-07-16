"""Direct-EQ injection experiment (anonymized -> inject -> CRA).

This is the *other* defense flow, distinct from run_defense_experiment.py:

    run_defense_experiment.py : raw -> inject -> RE-ANONYMIZE -> CRA
    eq_injector.py (this)     : anonymized -> inject into one published EQ -> CRA

Here we do NOT touch the anonymizer. We take an already-anonymized D_gen (a real
ARX/GUI export), append one (or n) fake row(s) whose QI cells equal an existing
EQ's published tuple -- so the record joins that EQ and only inflates its |EQ| by
n -- then re-run the CRA. The ARX partition, generalization levels, and overlap
structure are all preserved exactly.

Key facts this experiment exposes:
  * A basic EQ (one free basic) has a unique CRA solution z = |EQ|; adding rows
    keeps it unique -> NO effect. Only COMPOUND EQs (>1 free basic) react.
  * Adding to a compound EQ raises its LP total `sum z_b = |EQ|`, so the number
    of consistent integer assignments (cra_solutions) generally grows -> more
    attacker uncertainty for that EQ.
  * k-anonymity is preserved (an already->=k EQ only gets larger).

Usage (from project root, venv python):
    # 1) list EQs and mark which ones a +1 injection can actually affect
    cra/.venv/bin/python -m defense.eq_injector \
        --dataset dataset/min1/anonymized_2qi_k5.csv \
        --hier-dir dataset/min1/hierarchies \
        --qi-cols "Age,Systolic Blood Pressure" --k 5 --list

    # 2) inject n rows into one EQ (by index from the listing) and show before/after
    cra/.venv/bin/python -m defense.eq_injector ... --eq-index 80 --n 1
    # or let it pick the first compound EQ automatically:
    cra/.venv/bin/python -m defense.eq_injector ... --auto-compound --n 1
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Sequence

from dataset import Dataset
from equivalence import (
    extract_equivalence_classes,
    merge_by_snapped_segment,
    sort_equivalence_classes,
)
from segment import Segment

from .segment_utils import load_hierarchies
from .metrics import run_cra_eval
from .noise_injector import MARKER


def build_eq_table(
    dataset_path: Path, hiers, k: int, qi_cols: Sequence[str]
) -> list:
    """Return the sorted + snap-merged EQ list (same order run_cra_eval uses)."""
    ds = Dataset.from_csv(dataset_path, k=k, qi_columns=list(qi_cols))
    extraction = extract_equivalence_classes(ds, list(hiers))
    sorted_eqs = merge_by_snapped_segment(sort_equivalence_classes(extraction.eqs))
    return sorted_eqs


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        header = list(r.fieldnames or [])
    return header, rows


def inject_into_eq(
    dataset_path: Path,
    hiers,
    k: int,
    qi_cols: Sequence[str],
    eq_index: int,
    n_insert: int,
    out_path: Path,
    seed: int = 0,
) -> dict:
    """Append n_insert fake rows into the EQ at `eq_index` of the anonymized
    file and write the augmented CSV to out_path. Returns metadata about the
    target EQ (its published tuple and |EQ|)."""
    sorted_eqs = build_eq_table(dataset_path, hiers, k, qi_cols)
    if not (0 <= eq_index < len(sorted_eqs)):
        raise IndexError(f"eq_index {eq_index} out of range 0..{len(sorted_eqs)-1}")
    target = sorted_eqs[eq_index]
    published = target.qi_values  # exact CSV cell strings for each QI (stripped)

    header, rows = _read_rows(dataset_path)
    rng = random.Random(seed)
    out_header = list(header) + ([MARKER] if MARKER not in header else [])

    # Existing rows keep marker 0.
    out_rows = [{**r, MARKER: "0"} for r in rows]

    # Build the fake rows: QI cells = the EQ's published tuple; non-QI sampled
    # from a real row of that EQ when possible (purely cosmetic; CRA ignores them).
    template_pool = [
        r for r in rows
        if all(r.get(q, "").strip().strip('"') == published[j]
               for j, q in enumerate(qi_cols))
    ]
    for t in range(n_insert):
        base = dict(rng.choice(template_pool)) if template_pool else {c: "" for c in header}
        fake = {**base}
        for j, q in enumerate(qi_cols):
            fake[q] = published[j]
        for c in header:
            lc = c.strip().lower()
            if lc in {"patient", "id", "name"} or lc.endswith("_id") or "uuid" in lc:
                fake[c] = f"FAKE-EQ{eq_index}-{t:04d}"
        fake[MARKER] = "1"
        out_rows.append(fake)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_header, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    return {
        "eq_index": eq_index,
        "published_tuple": list(published),
        "qi_layers": "/".join(str(l) for l in target.qi_layers),
        "is_basic": Segment(target.qi_nodes).is_basic,
        "record_count_before": target.record_count,
        "record_count_after": target.record_count + n_insert,
        "n_inserted": n_insert,
        "template_rows_found": len(template_pool),
        "out_path": str(out_path),
    }


def _print_eq_listing(per_eq: list[dict], only_compound: bool) -> None:
    print(f"\n{'idx':>4} {'|EQ|':>5} {'layers':>7} {'kind':>8} "
          f"{'free':>4} {'LR_sol':>8} {'CRA_sol':>8} {'ratio':>7} {'zi=1':>5}  qi_values")
    print("-" * 100)
    shown = 0
    for e in per_eq:
        actionable = (not e["is_basic"]) and e["cra_solutions"] != 1
        if only_compound and e["is_basic"]:
            continue
        mark = " *" if actionable else "  "
        print(f"{e['eq_index']:>4} {e['record_count']:>5} {e['qi_layers']:>7} "
              f"{'basic' if e['is_basic'] else 'compound':>8} {e['free_basics']:>4} "
              f"{e['lr_solutions']:>8} {e['cra_solutions']:>8} "
              f"{str(e['cra_ratio']):>7} {str(e['has_zi_eq_1']):>5}{mark} {e['qi_values']}")
        shown += 1
    print(f"\n({shown} EQs listed; rows marked * are compound with CRA_sol!=1 -- "
          f"a +n injection can actually change these.)")


def _delta(a, b):
    if a is None or b is None:
        return f"{a} -> {b}"
    d = b - a
    return f"{a} -> {b}  (Δ{'+' if d >= 0 else ''}{round(d, 4)})"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--hier-dir", required=True, type=Path)
    p.add_argument("--qi-cols", required=True)
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--list", action="store_true",
                   help="List EQs (mark those a +n injection can affect) and exit.")
    p.add_argument("--only-compound", action="store_true",
                   help="With --list, show only compound EQs.")
    p.add_argument("--eq-index", type=int, default=None,
                   help="Target EQ index (from --list).")
    p.add_argument("--auto-compound", action="store_true",
                   help="Pick the first compound EQ with CRA_sol!=1 automatically.")
    p.add_argument("--n", type=int, default=1, help="Rows to insert. Default 1.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output augmented CSV. Default <dataset>_plus.csv.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cra-timeout-seconds", type=float, default=30.0)
    p.add_argument("--max-solutions-per-eq", type=int, default=None)
    args = p.parse_args()

    qi_cols = [c.strip() for c in args.qi_cols.split(",") if c.strip()]
    hiers = load_hierarchies(qi_cols, args.hier_dir)

    # Baseline CRA (always needed for listing and for before/after).
    before_summary, before_per_eq = run_cra_eval(
        args.dataset, hiers, args.k, qi_columns=qi_cols,
        max_solutions=args.max_solutions_per_eq,
        cra_timeout_seconds=args.cra_timeout_seconds,
    )

    if args.list or (args.eq_index is None and not args.auto_compound):
        _print_eq_listing(before_per_eq, args.only_compound)
        if args.list:
            return

    # Resolve target index.
    eq_index = args.eq_index
    if args.auto_compound and eq_index is None:
        cand = [e for e in before_per_eq if not e["is_basic"] and e["cra_solutions"] != 1]
        if not cand:
            print("No compound EQ with CRA_sol!=1 found; nothing to inject into.")
            return
        eq_index = cand[0]["eq_index"]
        print(f"auto-compound: selected EQ #{eq_index}")

    out_path = args.out or args.dataset.with_name(args.dataset.stem + "_plus.csv")
    meta = inject_into_eq(
        args.dataset, hiers, args.k, qi_cols, eq_index, args.n, out_path, args.seed)

    after_summary, after_per_eq = run_cra_eval(
        out_path, hiers, args.k, qi_columns=qi_cols,
        max_solutions=args.max_solutions_per_eq,
        cra_timeout_seconds=args.cra_timeout_seconds,
    )

    tb = before_per_eq[eq_index]
    ta = after_per_eq[eq_index]

    print("\n=== TARGET EQ (before -> after) ===")
    print(f"  eq_index        : {eq_index}")
    print(f"  published tuple : {meta['published_tuple']}  layers={meta['qi_layers']} "
          f"({'basic' if meta['is_basic'] else 'compound'})")
    print(f"  |EQ|            : {_delta(tb['record_count'], ta['record_count'])}")
    print(f"  free basics     : {tb['free_basics']}")
    print(f"  LR_solutions    : {_delta(tb['lr_solutions'], ta['lr_solutions'])}")
    print(f"  CRA_solutions   : {_delta(tb['cra_solutions'], ta['cra_solutions'])}")
    print(f"  CRA_ratio LR/CRA: {_delta(tb['cra_ratio'], ta['cra_ratio'])}")
    print(f"  has z_i=1       : {tb['has_zi_eq_1']} -> {ta['has_zi_eq_1']}")

    print("\n=== GLOBAL (before -> after) ===")
    for key, label in [
        ("eqs_exact", "exact EQs"),
        ("exact_eq_pct", "exact EQ %"),
        ("single_out_risk_proxy", "single-out risk proxy %"),
        ("cra_ratio_mean", "CRA_ratio mean"),
        ("cra_ratio_median", "CRA_ratio median"),
        ("cra_int_assignments_mean", "CRA int-assignments mean"),
    ]:
        print(f"  {label:26s}: {_delta(before_summary.get(key), after_summary.get(key))}")

    # Persist a compact record of the experiment.
    exp = {
        "dataset": str(args.dataset), "eq_index": eq_index, "n_inserted": args.n,
        "target_before": tb, "target_after": ta,
        "global_before": before_summary, "global_after": after_summary,
        "meta": meta,
    }
    rec = out_path.with_name(out_path.stem + "_experiment.json")
    rec.write_text(json.dumps(exp, indent=2))
    print(f"\naugmented CSV : {out_path}")
    print(f"experiment log: {rec}")


if __name__ == "__main__":
    main()
