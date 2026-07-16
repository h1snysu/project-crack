"""Defense flow (post-anonymization): insert fake record(s) directly into an
existing equivalence class of an ALREADY-anonymized dataset, then re-run the CRA
-- NO re-anonymization. This matches the refined defense model:

    have raw + anonymized + hierarchy + k
      -> add one record into one EQ of the anonymized data
      -> run CRA, observe how the attack changes.

A fake record simply repeats a chosen EQ's generalized QI interval values (so it
falls inside that EQ, raising |EQ| by 1); PATIENT is a traceable FAKE id and the
marker column __fake__=1. The CRA reads only the QI columns, so the marker/id
are ignored by the attack but let us check single-out afterwards.

Usage (from project root):
    # list the EQs so you can pick one
    cra/.venv/bin/python -m defense.inject_into_eq --list \
        --input dataset/min1/anonymized_2qi_k5.csv \
        --qi-cols "Age,Systolic Blood Pressure" --k 5

    # inject 1 fake into EQ #12 and compare CRA before/after
    cra/.venv/bin/python -m defense.inject_into_eq \
        --input dataset/min1/anonymized_2qi_k5.csv \
        --output out/inj_2qi_k5_eq12.csv \
        --qi-cols "Age,Systolic Blood Pressure" --k 5 \
        --eq-index 12 --n-fake 1
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .segment_utils import load_hierarchies
from .metrics import run_cra_eval

MARKER = "__fake__"


def read_rows(path: Path):
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r), list(r.fieldnames or [])


def list_eqs(rows, qi_cols):
    """Distinct EQs (QI tuples) with counts, sorted by count descending."""
    c = Counter(tuple(r[q] for q in qi_cols) for r in rows)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


def make_fake_row(header, qi_cols, eq_values, idx):
    row = {}
    for col in header:
        row[col] = ""
    for q, v in zip(qi_cols, eq_values):
        row[q] = v
    for col in header:
        cl = col.strip().lower()
        if cl in ("patient", "id", "name"):
            row[col] = f"FAKE-{idx:06d}"
        elif col not in qi_cols and cl not in ("patient", "id", "name"):
            if row[col] == "":
                row[col] = "FAKE"
    row[MARKER] = "1"
    return row


def _fmt_eq(e):
    return (f"|EQ|={e['record_count']:>4}  lr={e['lr_solutions']:>8}  "
            f"cra={e['cra_solutions']:>8}  cra_ratio={e['cra_ratio']}  "
            f"z_i=1?={e['has_zi_eq_1']}  basic={e['is_basic']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="ALREADY-anonymized CSV.")
    p.add_argument("--qi-cols", required=True)
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--hier-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the injected CSV.")
    p.add_argument("--list", action="store_true",
                   help="Just list the EQs (index, QI values, count) and exit.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--eq-index", type=int, default=None,
                   help="Index (from --list) of the EQ to inject into.")
    g.add_argument("--eq-values", default=None,
                   help="Pipe-separated QI interval values of the target EQ, "
                        "e.g. '[80, 85[|[120, 130['.")
    p.add_argument("--n-fake", type=int, default=1,
                   help="How many fake records to add into the EQ (default 1).")
    p.add_argument("--cra-timeout-seconds", type=float, default=None)
    p.add_argument("--max-solutions-per-eq", type=int, default=None)
    args = p.parse_args()

    qi = [c.strip() for c in args.qi_cols.split(",") if c.strip()]
    hier_dir = args.hier_dir or (args.input.parent / "hierarchies")
    hiers = load_hierarchies(qi, hier_dir)
    rows, header = read_rows(args.input)
    eqs = list_eqs(rows, qi)

    if args.list or (args.eq_index is None and args.eq_values is None):
        print(f"{len(eqs)} EQs in {args.input.name} (index: QI values | count):")
        for i, (vals, cnt) in enumerate(eqs):
            star = "  <- OUTLIER (*)" if vals[0] == "*" else ""
            print(f"  [{i:>3}] {' | '.join(vals):<40} count={cnt}{star}")
        if args.list:
            return
        print("\nPass --eq-index N or --eq-values to inject. Nothing written.")
        return

    if args.eq_values is not None:
        target = tuple(v.strip() for v in args.eq_values.split("|"))
    else:
        target = eqs[args.eq_index][0]
    target_count = dict(eqs).get(target)
    if target_count is None:
        raise SystemExit(f"EQ {target} not found in the dataset.")

    if MARKER not in header:
        header = header + [MARKER]
    for r in rows:
        r.setdefault(MARKER, "0")

    fakes = [make_fake_row(header, qi, target, i) for i in range(args.n_fake)]
    new_rows = rows + fakes

    out = args.output or args.input.with_name(args.input.stem + "_injected.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(new_rows)

    # CRA before / after
    print(f"Target EQ: {' | '.join(target)}  (count {target_count} -> "
          f"{target_count + args.n_fake} after +{args.n_fake} fake)\n")
    before, per_before = run_cra_eval(
        args.input, hiers, args.k, qi_columns=qi,
        cra_timeout_seconds=args.cra_timeout_seconds,
        max_solutions=args.max_solutions_per_eq)
    after, per_after = run_cra_eval(
        out, hiers, args.k, qi_columns=qi,
        cra_timeout_seconds=args.cra_timeout_seconds,
        max_solutions=args.max_solutions_per_eq)

    def find_eq(per):
        for e in per:
            if e["qi_values"] == "|".join(target):
                return e
        return None
    eb, ea = find_eq(per_before), find_eq(per_after)

    print("=== Target EQ (CRA per-EQ) ===")
    print(f"  before: {_fmt_eq(eb) if eb else '(EQ not present / outlier)'}")
    print(f"  after : {_fmt_eq(ea) if ea else '(EQ not present / outlier)'}")

    def line(tag, s):
        return (f"  {tag}: EQs={s['eqs_evaluated']} exact%={s['exact_eq_pct']} "
                f"single_out%={s['single_out_risk_proxy']} "
                f"cra_ratio_mean={s['cra_ratio_mean']} "
                f"cra_ratio_median={s['cra_ratio_median']} "
                f"int_assign_median={s['cra_int_assignments_median']}")
    print("\n=== Whole-dataset CRA summary ===")
    print(line("before", before))
    print(line("after ", after))
    print(f"\nwrote injected dataset -> {out}")


if __name__ == "__main__":
    main()
