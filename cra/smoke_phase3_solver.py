"""Phase 3 step-2 smoke test: SegmentLP / CP-SAT enumeration.

Uses the 2-QI dataset to obtain a real `Segment` with 2 basics, then runs the
solver under five constraint configurations and checks that the number of
integer solutions matches a closed-form prediction.

Stars-and-bars:
    # non-negative integer solutions to z_0 + ... + z_{n-1} = N
    = C(N + n - 1, n - 1)

Run from cra/:
    python smoke_phase3_solver.py
"""
from __future__ import annotations

from math import comb
from pathlib import Path

from dataset import Dataset
from equivalence import extract_equivalence_classes, sort_equivalence_classes
from grid import Grid
from hierarchy import Hierarchy
from segment import Segment
from solver import LPSpec, SegmentLP

HIER_DIR = Path("../dataset/min1/hierarchies")
DATASET = Path("../dataset/min1/anonymized_2qi_k5.csv")
K = 5


def header(s: str) -> None:
    print(f"\n=== {s} ===")


def check(name: str, got, expected) -> None:
    ok = "PASS" if got == expected else "FAIL"
    print(f"  [{ok}] {name}: got={got!r} expected={expected!r}")
    if got != expected:
        raise SystemExit(1)


def main() -> None:
    ds = Dataset.from_csv(DATASET, k=K)
    age = Hierarchy.from_csv(HIER_DIR / "hierarchy_age.csv", "Age", 0)
    sbp = Hierarchy.from_csv(
        HIER_DIR / "hierarchy_systolic-blood-pressure.csv",
        "Systolic Blood Pressure", 1,
    )
    grid = Grid([age, sbp])

    # Pick the first EQ in sort order whose snapped scope has exactly 2 basics
    # (a small compound EQ -- well-suited for closed-form checking).
    extraction = extract_equivalence_classes(ds, [age, sbp])
    eq = next(
        e for e in sort_equivalence_classes(extraction.eqs)
        if Segment(e.qi_nodes).size() == 2
    )
    scope = Segment(eq.qi_nodes)
    basics = list(scope.iter_basic_leaf_tuples())

    header("Setup")
    print(f"  EQ {eq.qi_values}, |EQ| = {eq.record_count}, scope = {scope}")
    print(f"  basics = {basics}")
    check("|basics|", len(basics), 2)
    n = eq.record_count

    # ---- Test 1: minimal LP (only the total-sum constraint) -------------
    header(f"T1: sum(z) == {n}, no halves, no sparse, no exclusion")
    spec = LPSpec(grid=grid, scope=scope, total=n, k=K)
    sols = SegmentLP.solve(spec)
    expected = comb(n + len(basics) - 1, len(basics) - 1)  # = n + 1 for 2 basics
    check("|solutions|", len(sols), expected)
    # Sanity: each solution sums to n.
    for s in sols:
        if sum(s.values()) != n:
            print(f"  FAIL: solution {s} does not sum to {n}")
            raise SystemExit(1)
    print(f"  first solution: {sols[0]}")
    print(f"  last  solution: {sols[-1]}")

    # ---- Test 2: halves >= 1 (each basic is itself a half-segment) ------
    halves = list(scope.iter_half_segments())
    header(f"T2: + halves >= 1 ({len(halves)} halves; each is one basic)")
    spec = LPSpec(grid=grid, scope=scope, total=n, k=K, halves=halves)
    sols = SegmentLP.solve(spec)
    # Both z_0, z_1 >= 1 with sum n -> n - 1 solutions.
    expected = n - 1
    check("|solutions|", len(sols), expected)
    for s in sols:
        # No basic should be missing (= zero).
        check_basics = set(s)
        if check_basics != set(basics):
            print(f"  FAIL: half >= 1 violated in {s}")
            raise SystemExit(1)

    # ---- Test 3: sparse <= k-1 on ONE basic -----------------------------
    # Treat the first half-segment (= first basic alone) as a sparse sub-segment.
    sparse_one = [halves[0]]
    header(f"T3: + sparse {sparse_one[0]} <= {K - 1}")
    spec = LPSpec(
        grid=grid, scope=scope, total=n, k=K, sparse=sparse_one
    )
    sols = SegmentLP.solve(spec)
    # z_0 in [0, K-1] (= 0..4), z_1 = n - z_0.
    # Both must be >= 0 -> z_0 in [max(0, n - n), min(K-1, n)] = [0, min(K-1, n)]
    expected = min(K, n + 1)  # z_0 takes K values 0..K-1
    check("|solutions|", len(sols), expected)
    for s in sols:
        # Look up z_0 (corresponding to the basic in halves[0]) -- it must be <= K-1.
        b0 = next(iter(set(halves[0].iter_basic_leaf_tuples())))
        if s.get(b0, 0) > K - 1:
            print(f"  FAIL: sparse cap violated in {s}")
            raise SystemExit(1)

    # ---- Test 4: exclude one basic (overlap) ----------------------------
    b0, b1 = basics[0], basics[1]
    header(f"T4: exclude {b0} -> only z_1 free")
    spec = LPSpec(
        grid=grid, scope=scope, total=n, k=K,
        excluded_basics={b0},
    )
    sols = SegmentLP.solve(spec)
    check("|solutions|", len(sols), 1)
    check("only solution", sols[0], {b1: n})

    # ---- Test 5: total = 0 (no outliers / no records to distribute) -----
    header("T5: total=0 with two free basics")
    spec = LPSpec(grid=grid, scope=scope, total=0, k=K)
    sols = SegmentLP.solve(spec)
    check("|solutions|", len(sols), 1)
    check("only solution (all zeros)", sols[0], {})

    # ---- Test 6: infeasible (halves >= 1 but excluded covers a half) ----
    header("T6: half whose only basic is excluded -> infeasible -> 0 solutions")
    spec = LPSpec(
        grid=grid, scope=scope, total=n, k=K,
        excluded_basics={b0},
        halves=[halves[0]],  # halves[0] contains only b0
    )
    sols = SegmentLP.solve(spec)
    check("|solutions|", len(sols), 0)

    # ---- Test 7: max_solutions cap --------------------------------------
    header("T7: max_solutions=3 caps enumeration")
    spec = LPSpec(grid=grid, scope=scope, total=n, k=K, max_solutions=3)
    sols = SegmentLP.solve(spec)
    check("|solutions|", len(sols), 3)

    print("\nAll Phase 3 step-2 solver checks passed.")


if __name__ == "__main__":
    main()
