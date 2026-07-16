"""Phase 3 step-3 smoke test: full Algorithm-3 orchestration on 2-QI / k=5.

Verifies:
  1. Every EQ's solutions sum to |EQ|.
  2. Every solution satisfies overlap (excluded basics are 0).
  3. Every solution satisfies halves >= 1 and sparse <= k-1.
  4. CP-SAT enumeration matches brute force on a representative EQ.

Run from cra/:
    python smoke_phase3_cra.py
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

from cra import build_eq_spec, build_outlier_spec, run_cra
from dataset import Dataset
from grid import Grid
from hierarchy import Hierarchy
from segment import Basic
from solver import LPSpec, SegmentLP, Solution
from math import comb

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


def brute_force(spec: LPSpec) -> list[Solution]:
    """Enumerate all integer non-negative assignments satisfying the spec.

    Only feasible for small free-variable counts (< ~12).
    """
    free = sorted(set(spec.grid.basics_in(spec.scope)) - spec.excluded_basics)
    if not free:
        return [{}] if spec.total == 0 else []
    n = len(free)
    if n > 12:
        raise ValueError(f"brute force only supports n <= 12, got {n}")
    out: list[Solution] = []
    for assignment in _ordered_compositions(spec.total, n):
        # Halves: sum over each half's free basics >= 1.
        ok = True
        for half in spec.halves:
            half_basics = set(spec.grid.basics_in(half)) & set(free)
            s = sum(assignment[free.index(b)] for b in half_basics)
            if s < 1:
                ok = False
                break
        if not ok:
            continue
        # Sparse: sum <= k-1 over each sub's free basics.
        bound = spec.k - 1
        for sub in spec.sparse:
            sub_basics = set(spec.grid.basics_in(sub)) & set(free)
            s = sum(assignment[free.index(b)] for b in sub_basics)
            if s > bound:
                ok = False
                break
        if not ok:
            continue
        out.append({b: v for b, v in zip(free, assignment) if v > 0})
    return out


def _ordered_compositions(total: int, n: int):
    """Yield every length-n tuple of non-negative ints summing to `total`."""
    if n == 1:
        yield (total,)
        return
    for v in range(total + 1):
        for rest in _ordered_compositions(total - v, n - 1):
            yield (v,) + rest


def main() -> None:
    ds = Dataset.from_csv(DATASET, k=K)
    age = Hierarchy.from_csv(HIER_DIR / "hierarchy_age.csv", "Age", 0)
    sbp = Hierarchy.from_csv(
        HIER_DIR / "hierarchy_systolic-blood-pressure.csv",
        "Systolic Blood Pressure", 1,
    )
    hiers = [age, sbp]

    header("Running run_cra(...) end-to-end (Algorithm 3 only)")
    result = run_cra(ds, hiers, max_solutions_per_eq=None)
    check("|pre_merge_eq_count|", result.pre_merge_eq_count, 88)
    check("|sorted_eqs| (after snap-merge)", len(result.sorted_eqs), 88)
    check("outlier_count", result.outlier_count, 1)
    # Conservation: every record in D_gen is in either a merged EQ or outliers.
    check(
        "record conservation",
        sum(eq.record_count for eq in result.sorted_eqs) + result.outlier_count,
        ds.n,
    )

    # ---- 1. Per-EQ correctness checks (cheap, run on all 167) ----------
    grid = Grid(hiers)
    active_values = {s.qi_values for s in result.eq_segments}
    feasibility_failures = 0
    total_solutions = 0
    max_solutions_in_one_eq = 0
    eqs_with_no_solution = 0
    for i, (eq, seg, sols) in enumerate(
        zip(result.sorted_eqs, result.eq_segments, result.eq_solutions)
    ):
        spec = build_eq_spec(
            grid=grid,
            eq=eq,
            eq_segment=seg,
            prior_eq_basics=result.eq_basics[:i],
            active_segment_values=active_values,
            k=ds.k,
        )
        if not sols:
            eqs_with_no_solution += 1
            continue
        total_solutions += len(sols)
        max_solutions_in_one_eq = max(max_solutions_in_one_eq, len(sols))
        for sol in sols:
            # (a) sum equals |EQ|.
            if sum(sol.values()) != eq.record_count:
                feasibility_failures += 1
                print(f"  FAIL sum: EQ#{i} {eq.qi_values}  sol={sol}")
                break
            # (b) excluded basics are zero (by construction, no var exists).
            for b in spec.excluded_basics:
                if sol.get(b, 0) != 0:
                    feasibility_failures += 1
                    print(f"  FAIL excl: EQ#{i} b={b} val={sol[b]}")
                    break
            # (c) halves >= 1.
            for half in spec.halves:
                hb = set(grid.basics_in(half))
                if sum(sol.get(b, 0) for b in hb) < 1:
                    feasibility_failures += 1
                    print(f"  FAIL half: EQ#{i} half={half}")
                    break
            # (d) sparse <= k-1.
            for sub in spec.sparse:
                sb = set(grid.basics_in(sub))
                if sum(sol.get(b, 0) for b in sb) > ds.k - 1:
                    feasibility_failures += 1
                    print(f"  FAIL sparse: EQ#{i} sub={sub}")
                    break

    check("per-EQ constraint failures", feasibility_failures, 0)
    print(
        f"  total solutions across all EQs: {total_solutions}, "
        f"max in single EQ: {max_solutions_in_one_eq}, "
        f"EQs w/ no solutions: {eqs_with_no_solution}"
    )
    # min1 (no layer-0 EQs) has clean Global-Recoding-style coverage, so
    # every EQ should yield at least one solution.
    check("EQs w/ no solutions (clean Global Recoding)", eqs_with_no_solution, 0)

    # ---- 2. Brute-force cross-check on a 2-basic compound EQ -----------
    target_idx = next(
        (
            i for i, (eq, seg) in enumerate(
                zip(result.sorted_eqs, result.eq_segments)
            )
            if seg.size() == 2
        ),
        None,
    )
    if target_idx is None:
        raise SystemExit("No 2-basic EQ in this dataset; can't run brute-force")
    eq = result.sorted_eqs[target_idx]
    seg = result.eq_segments[target_idx]
    header(f"Brute-force vs CP-SAT on EQ#{target_idx} {eq.qi_values}")
    spec = build_eq_spec(
        grid=grid,
        eq=eq,
        eq_segment=seg,
        prior_eq_basics=result.eq_basics[:target_idx],
        active_segment_values=active_values,
        k=ds.k,
    )
    print(
        f"  EQ#{target_idx}  |EQ|={eq.record_count}  "
        f"basics={spec.scope.size()}  "
        f"excluded={len(spec.excluded_basics)}  "
        f"halves={len(spec.halves)}  "
        f"sparse={len(spec.sparse)}"
    )
    bf = brute_force(spec)
    cp = result.eq_solutions[target_idx]
    check("|brute_force|", len(bf), len(cp))
    check("solutions match (as sets)", set(map(tuple, (sorted(s.items()) for s in bf))),
          set(map(tuple, (sorted(s.items()) for s in cp))))

    # ---- 3. Brute-force on a small compound EQ with > 2 basics ----------
    multi_idx = next(
        (
            i for i, (eq, seg) in enumerate(
                zip(result.sorted_eqs, result.eq_segments)
            )
            if 3 <= seg.size() <= 4 and i != target_idx
        ),
        None,
    )
    if multi_idx is not None:
        header(f"Brute-force vs CP-SAT on a second small EQ (#{multi_idx})")
        eq = result.sorted_eqs[multi_idx]
        seg = result.eq_segments[multi_idx]
        spec = build_eq_spec(
            grid=grid,
            eq=eq,
            eq_segment=seg,
            prior_eq_basics=result.eq_basics[:multi_idx],
            active_segment_values=active_values,
            k=ds.k,
        )
        print(
            f"  EQ {eq.qi_values}  |EQ|={eq.record_count}  "
            f"basics={spec.scope.size()}"
        )
        bf = brute_force(spec)
        cp = result.eq_solutions[multi_idx]
        check("|brute_force|", len(bf), len(cp))

    # ---- 4. Mass exhaustive cross-check on every EQ with <= 8 basics ---
    header("Brute-force vs CP-SAT on every EQ with <= 8 basics")
    matched = mismatched = skipped = 0
    for i, (eq, seg) in enumerate(zip(result.sorted_eqs, result.eq_segments)):
        if seg.size() > 8:
            skipped += 1
            continue
        spec = build_eq_spec(
            grid=grid,
            eq=eq,
            eq_segment=seg,
            prior_eq_basics=result.eq_basics[:i],
            active_segment_values=active_values,
            k=ds.k,
        )
        bf = brute_force(spec)
        cp = result.eq_solutions[i]
        if len(bf) == len(cp):
            matched += 1
        else:
            mismatched += 1
            print(f"  MISMATCH EQ#{i} {eq.qi_values}: bf={len(bf)} cp={len(cp)}")
    print(
        f"  matched={matched}  mismatched={mismatched}  skipped(>8 basics)={skipped}"
    )
    check("|mismatched|", mismatched, 0)

    # ---- 5. Algorithm 4 (outliers) -------------------------------------
    header("Algorithm 4: outlier LP")
    grid = Grid(hiers)
    active_values = {s.qi_values for s in result.eq_segments}
    spec_o = build_outlier_spec(
        grid=grid,
        outlier_count=result.outlier_count,
        eq_basics=result.eq_basics,
        active_segment_values=active_values,
        k=ds.k,
    )
    free_outlier = set(grid.basics_in(spec_o.scope)) - spec_o.excluded_basics
    print(
        f"  total={spec_o.total}  k={spec_o.k}  "
        f"|grid|={grid.total_basics}  "
        f"excluded={len(spec_o.excluded_basics)}  "
        f"free={len(free_outlier)}  halves={len(spec_o.halves)}  "
        f"sparse={len(spec_o.sparse)}"
    )
    # Closed-form: outlier_count < k -> halves disabled, sparse cap (k-1)
    # non-binding when total <= k-1. Solutions = C(total + n_free - 1, n_free - 1).
    closed_form = comb(spec_o.total + len(free_outlier) - 1, len(free_outlier) - 1)
    check("|outlier_solutions| == closed-form", len(result.outlier_solutions), closed_form)
    check("halves disabled when outlier_count < k", len(spec_o.halves), 0)
    # Every solution sums to outlier_count and never exceeds k-1 per basic.
    for s in result.outlier_solutions:
        if sum(s.values()) != result.outlier_count:
            raise SystemExit(f"FAIL: outlier sol {s} sums to != {result.outlier_count}")
        if any(v > ds.k - 1 for v in s.values()):
            raise SystemExit(f"FAIL: outlier sol {s} violates sparse cap")
        if set(s) & spec_o.excluded_basics:
            raise SystemExit(f"FAIL: outlier sol {s} touches excluded basic")

    # ---- 6. Synthetic outlier_count >= k -> halves kick in --------------
    header("Algorithm 4 halves kick in when outlier_count >= k")
    spec_h = build_outlier_spec(
        grid=grid,
        outlier_count=ds.k,   # exactly k
        eq_basics=result.eq_basics,
        active_segment_values=active_values,
        k=ds.k,
    )
    print(
        f"  total={spec_h.total}  halves={len(spec_h.halves)}  "
        f"sparse={len(spec_h.sparse)}"
    )
    # Every grid-half not matching an active EQ should be a constraint.
    grid_halves_total = sum(1 for _ in grid.grid_segment.iter_half_segments())
    exempted = sum(
        1 for h in grid.grid_segment.iter_half_segments()
        if h.qi_values in active_values
    )
    check(
        "grid-halves applied",
        len(spec_h.halves),
        grid_halves_total - exempted,
    )
    sols_h = SegmentLP.solve(spec_h)
    print(f"  solutions with halves: {len(sols_h)}")
    # Every solution must put at least 1 record in each applied half.
    for s in sols_h:
        for half in spec_h.halves:
            hb = set(grid.basics_in(half)) - spec_h.excluded_basics
            if sum(s.get(b, 0) for b in hb) < 1:
                raise SystemExit(f"FAIL: outlier halves violated by {s}")
    check("halves feasibility", True, len(sols_h) > 0)

    print("\nAll Phase 3 step-3 + Algorithm 4 checks passed.")


if __name__ == "__main__":
    main()
