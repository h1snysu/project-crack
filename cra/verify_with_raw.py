"""Cross-verify CRA output against the raw (pre-anonymisation) dataset.

We can't row-match raw <-> anonymized rows (ARX reorders), so we work at the
aggregate basic-segment level:

  1. For each raw record, snap its (Age, Systolic Blood Pressure) to its
     layer-1 basic and accumulate `gt_count[basic]`.
  2. Walk merged EQs in sort order. For each EQ E, the "natural" attribution
     is `gt_count[b]` for every b in E's free basics (basics in E's scope
     not owned by any prior EQ). All raw records at b that aren't yet
     attributed are assigned to E.
  3. Check that this natural attribution sums to |E| and appears in E's
     enumerated CRA solutions.
  4. The 3 outliers' true basics must be inside the outlier LP's free set,
     and the natural outlier attribution must be in the enumerated outlier
     solutions.

Run from cra/:
    python verify_with_raw.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from cra import build_eq_spec, build_outlier_spec, run_cra
from dataset import Dataset
from grid import Grid
from hierarchy import Hierarchy
from segment import Basic
from solver import SegmentLP

RAW_FILE = Path("../dataset/min1/example-raw-dataset.csv")
ANON_FILE = Path("../dataset/min1/anonymized_2qi_k5.csv")
HIER_DIR = Path("../dataset/min1/hierarchies")
K = 5


def header(s: str) -> None:
    print(f"\n=== {s} ===")


def load_raw_age_sbp(path: Path) -> list[tuple[str, str]]:
    """Read the raw CSV; return the (Age, Systolic Blood Pressure) columns."""
    out: list[tuple[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((row["Age"].strip(), row["Systolic Blood Pressure"].strip()))
    return out


def main() -> None:
    age = Hierarchy.from_csv(HIER_DIR / "hierarchy_age.csv", "Age", 0)
    sbp = Hierarchy.from_csv(
        HIER_DIR / "hierarchy_systolic-blood-pressure.csv",
        "Systolic Blood Pressure", 1,
    )
    hiers = [age, sbp]
    grid = Grid(hiers)
    ds = Dataset.from_csv(ANON_FILE, k=K)

    raw_pairs = load_raw_age_sbp(RAW_FILE)
    print(
        f"Loaded {len(raw_pairs)} raw records "
        f"(anonymised file has {ds.n} records)"
    )

    # --- 1. Ground-truth basic distribution from raw data --------------
    gt_count: Counter[Basic] = Counter()
    for a, s in raw_pairs:
        age_node = age.resolve(a)                # layer 0 (raw)
        sbp_node = sbp.resolve(s)                # layer 0 (raw)
        age_basic = age_node.parent.value        # snap up to layer 1
        sbp_basic = sbp_node.parent.value
        gt_count[(age_basic, sbp_basic)] += 1

    print(f"  ground-truth basics covered: {len(gt_count)}  (grid has {grid.total_basics})")
    print(f"  sum(gt_count) = {sum(gt_count.values())}  (raw n = {len(raw_pairs)})")
    assert sum(gt_count.values()) == len(raw_pairs)

    # --- 2. Run CRA over the anonymised dataset ------------------------
    header("Running CRA")
    result = run_cra(ds, hiers)
    print(
        f"  pre_merge_eqs={result.pre_merge_eq_count}  "
        f"merged_eqs={len(result.sorted_eqs)}  "
        f"outlier_count={result.outlier_count}"
    )
    print(
        f"  total |EQ| sum + outlier_count = "
        f"{sum(e.record_count for e in result.sorted_eqs) + result.outlier_count}"
    )

    # --- 3. Natural attribution + per-EQ membership ---------------------
    header("Natural attribution check (per EQ)")
    attributed: dict[Basic, int] = {}
    matched = 0
    sum_mismatch = []
    membership_miss = []
    no_free = []
    for i, (eq, seg, sols) in enumerate(
        zip(result.sorted_eqs, result.eq_segments, result.eq_solutions)
    ):
        scope_basics = set(grid.basics_in(seg))
        # Build the prior-claimed set (= union of prior EQ scope basics).
        excluded: set[Basic] = set()
        for b in result.eq_basics[:i]:
            excluded |= b
        free = scope_basics - excluded
        if not free:
            no_free.append(i)
            continue
        # Each EQ owns the remaining gt records in its free basics.
        natural_z = {b: gt_count.get(b, 0) - attributed.get(b, 0) for b in free}
        # Drop zeros to match solver's omit-zero convention.
        natural_nz = {b: v for b, v in natural_z.items() if v > 0}
        s = sum(natural_z.values())
        if s != eq.record_count:
            sum_mismatch.append((i, eq.record_count, s))
            # still try to update attribution to keep going
        # Update attribution
        for b, v in natural_z.items():
            attributed[b] = attributed.get(b, 0) + max(v, 0)
        # Compare natural attribution against enumerated solutions.
        # A solution is a dict[Basic, int] with zeros omitted.
        if natural_nz in sols:
            matched += 1
        else:
            membership_miss.append((i, eq.qi_values, eq.record_count, natural_nz, len(sols)))

    print(f"  EQs whose natural attribution IS in CRA solutions: {matched} / {len(result.sorted_eqs)}")
    print(f"  EQs with no free basics (anomalies): {len(no_free)} (indices {no_free})")
    print(f"  EQs with sum mismatch (natural sum != |EQ|): {len(sum_mismatch)}")
    for i, exp, got in sum_mismatch:
        eq = result.sorted_eqs[i]
        print(f"    EQ#{i} {eq.qi_values}  expected sum={exp}  got={got}")
    print(f"  EQs where natural attribution NOT in solutions: {len(membership_miss)}")
    for i, vals, n, nz, nsols in membership_miss[:5]:
        print(f"    EQ#{i} {vals}  |EQ|={n}  natural={nz}  (|sols|={nsols})")

    # --- 4. Outlier verification ---------------------------------------
    header("Outlier LP cross-check")
    active_values = {s.qi_values for s in result.eq_segments}
    spec_o = build_outlier_spec(
        grid=grid,
        outlier_count=result.outlier_count,
        eq_basics=result.eq_basics,
        active_segment_values=active_values,
        k=ds.k,
    )
    free_outlier = set(grid.basics_in(spec_o.scope)) - spec_o.excluded_basics
    # Whatever raw records weren't attributed to any EQ are the outliers.
    leftover: dict[Basic, int] = {}
    for b, gt in gt_count.items():
        att = attributed.get(b, 0)
        if gt > att:
            leftover[b] = gt - att
    print(f"  sum(gt_count - attributed) = {sum(leftover.values())}")
    print(f"  outlier_count from D_gen   = {result.outlier_count}")
    print(f"  free outlier basics        = {len(free_outlier)}")
    print(f"  unattributed basic detail  = {dict(leftover)}")
    # The leftover distribution should be exactly one of the enumerated solutions.
    leftover_nz = {b: v for b, v in leftover.items() if v > 0}
    in_solutions = leftover_nz in result.outlier_solutions
    print(f"  leftover ∈ outlier_solutions: {in_solutions}")
    # All leftover basics should be in the free outlier set (i.e. not excluded).
    leftover_in_free = all(b in free_outlier for b in leftover_nz)
    print(f"  all leftover basics ∈ free outlier set: {leftover_in_free}")


if __name__ == "__main__":
    main()
