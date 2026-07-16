"""Run the Combinatorial Refinement Attack (CRA) on one anonymised CSV and
report attack-precision metrics.

The script ingests:
  * an ARX-LR-anonymised dataset (D_gen) as CSV,
  * the generalisation hierarchies for its quasi-identifiers,
  * the k-anonymity parameter k,

and prints a structured report covering:
  * Per-EQ attack precision (Exact Breach / High-Risk / Uncertain)
  * Record-level breach percentages (weighted by |EQ|)
  * CRA Ratio (enumerated solutions / stars-and-bars baseline)
  * Algorithm 4 outlier metrics (with closed-form bypass for huge LPs)
  * Per-phase wall time

Examples (run from the cra/ directory):
    # 2 QIs, k=5: fast (< 1 sec).
    python eval_cra.py \\
        --dataset ../dataset/min1/anonymized_2qi_k5.csv \\
        --hier-dir ../dataset/min1/hierarchies \\
        -k 5

    # 4 QIs, k=5: per-EQ enumeration can explode -- always cap.
    python eval_cra.py \\
        --dataset ../dataset/min1/anonymized_4qi_k5.csv \\
        --hier-dir ../dataset/min1/hierarchies \\
        -k 5 \\
        --max-solutions 10000 --verbose

This module is also importable: `from eval_cra import evaluate_dataset`. The
`run_sweep.py` driver uses that API to evaluate many datasets at once.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Callable

from cra import (
    build_eq_spec,
    build_outlier_spec,
    run_algorithm4,
)
from dataset import Dataset
from equivalence import (
    extract_equivalence_classes,
    merge_by_snapped_segment,
    sort_equivalence_classes,
)
from grid import Grid
from hierarchy import Hierarchy
from segment import Segment
from solver import SegmentLP


# Canonical QI column name -> hierarchy CSV filename (case-insensitive lookup).
# Extend this map to add new QIs.
QI_TO_HIER: dict[str, str] = {
    "age": "hierarchy_age.csv",
    "sex": "hierarchy_sex.csv",
    "race": "hierarchy_race.csv",
    "education": "hierarchy_education.csv",
}


def resolve_hierarchies(
    qi_names: list[str], hier_dir: Path
) -> list[Hierarchy]:
    """Match each QI name to a hierarchy CSV file in `hier_dir`."""
    lookup = {k.lower(): v for k, v in QI_TO_HIER.items()}
    hiers: list[Hierarchy] = []
    for i, qi in enumerate(qi_names):
        fname = lookup.get(qi.lower())
        if fname is None:
            raise SystemExit(
                f"No hierarchy mapping registered for QI {qi!r}. "
                f"Add it to QI_TO_HIER in eval_cra.py."
            )
        hier_path = hier_dir / fname
        if not hier_path.exists():
            raise SystemExit(
                f"Hierarchy file not found: {hier_path}. "
                f"Check --hier-dir."
            )
        hiers.append(Hierarchy.from_csv(hier_path, qi, i))
    return hiers


# =====================================================================
# Result dataclass + evaluation function (importable API).
# =====================================================================


@dataclass
class EvalResult:
    """Complete metrics from one CRA evaluation. All counts are absolute;
    helper properties expose the canonical percentages."""

    # Input/identity
    dataset_file: str
    m: int
    k: int
    n: int
    qi_names: list[str]
    grid_basics: int

    # EQ counts
    eqs_pre_merge: int
    eqs_merged: int
    outlier_count: int

    # Per-EQ bucket counts (out of eqs_merged)
    eqs_infeasible: int
    eqs_exact: int
    eqs_highrisk: int
    eqs_uncertain: int

    # Per-EQ solution-count stats on solvable EQs
    solvable_mean: float | None
    solvable_median: float | None
    solvable_q25: int | None
    solvable_q75: int | None
    solvable_max: int | None

    # Cap diagnostics
    max_solutions_cap: int | None
    eqs_capped: int

    # Record-level (weighted by |EQ|)
    records_in_eqs: int
    records_infeasible: int
    records_exact: int
    records_highrisk: int
    records_uncertain: int

    # CRA Ratio = enumerated / stars-and-bars baseline (per EQ, then aggregated)
    cra_ratio_mean: float | None
    cra_ratio_median: float | None
    cra_ratio_min: float | None
    cra_ratio_max: float | None
    eqs_trivially_unique: int
    eqs_refined_to_unique: int

    # Outlier LP (Algorithm 4)
    outlier_free_basics: int
    outlier_closed_form: int | None
    outlier_enum_skipped: bool
    outlier_solutions_enumerated: int

    # Timing
    wall_time_alg3: float
    wall_time_alg4: float

    @property
    def wall_time_total(self) -> float:
        return self.wall_time_alg3 + self.wall_time_alg4

    # Convenience % helpers ------------------------------------------------
    def _pct(self, num: int, denom: int) -> float:
        return 100.0 * num / denom if denom else 0.0

    @property
    def exact_eq_pct(self) -> float:
        return self._pct(self.eqs_exact, self.eqs_merged)

    @property
    def highrisk_eq_pct(self) -> float:
        return self._pct(self.eqs_highrisk, self.eqs_merged)

    @property
    def uncertain_eq_pct(self) -> float:
        return self._pct(self.eqs_uncertain, self.eqs_merged)

    @property
    def exact_record_pct(self) -> float:
        return self._pct(self.records_exact, self.records_in_eqs)

    @property
    def highrisk_record_pct(self) -> float:
        return self._pct(self.records_highrisk, self.records_in_eqs)

    @property
    def uncertain_record_pct(self) -> float:
        return self._pct(self.records_uncertain, self.records_in_eqs)


def evaluate_dataset(
    dataset_path: Path,
    hier_dir: Path,
    k: int,
    *,
    qi_columns: list[str] | None = None,
    max_solutions: int | None = None,
    max_outlier_solutions: int | None = None,
    outlier_enum_threshold: int = 200_000,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> EvalResult:
    """Run the full CRA pipeline on one dataset and return all metrics.

    Parameters
    ----------
    dataset_path : Path
        Anonymised CSV. QI columns are auto-detected from interval shape.
    hier_dir : Path
        Directory containing one `hierarchy_<qi>.csv` per QI.
    k : int
        k-anonymity parameter (used by halves/sparse constraints).
    qi_columns : list[str] | None
        Override the auto-detected QI column list.
    max_solutions : int | None
        Per-EQ enumeration cap. None = enumerate completely; recommended for
        m >= 4 (e.g. 10000).
    max_outlier_solutions : int | None
        Cap for the outlier LP. Ignored when the closed-form bypass triggers.
    outlier_enum_threshold : int
        If the outlier LP's closed-form solution count exceeds this AND the LP
        is in the closed-form regime (no halves, total <= k-1), we skip
        enumeration and report only the count.
    progress_callback : Callable | None
        Called once per EQ as `cb(i, total, elapsed_seconds)`.
    """
    ds = Dataset.from_csv(dataset_path, k=k, qi_columns=qi_columns)
    hiers = resolve_hierarchies(ds.qi_names, hier_dir)
    grid = Grid(hiers)

    # Phase 2: extract + sort + snap-merge.
    extraction = extract_equivalence_classes(ds, hiers)
    pre_merge = len(extraction.eqs)
    sorted_eqs = merge_by_snapped_segment(
        sort_equivalence_classes(extraction.eqs)
    )

    # Phase 3-8: per-EQ LPs (Algorithm 3).
    eq_segments = [Segment(eq.qi_nodes) for eq in sorted_eqs]
    eq_basics = [set(grid.basics_in(s)) for s in eq_segments]
    active_values = {s.qi_values for s in eq_segments}

    eq_solutions: list = []
    eq_specs_excluded: list[set] = []   # cached for the CRA-ratio computation
    t0 = perf_counter()
    for i, (eq, seg) in enumerate(zip(sorted_eqs, eq_segments)):
        if progress_callback is not None:
            progress_callback(i, len(sorted_eqs), perf_counter() - t0)
        spec = build_eq_spec(
            grid=grid, eq=eq, eq_segment=seg,
            prior_eq_basics=eq_basics[:i],
            active_segment_values=active_values, k=ds.k,
            max_solutions=max_solutions,
        )
        eq_specs_excluded.append(spec.excluded_basics)
        eq_solutions.append(SegmentLP.solve(spec))
    t_alg3 = perf_counter() - t0

    # Algorithm 4: outlier LP, with closed-form bypass.
    outlier_count = extraction.outlier_count
    outlier_solutions: list = []
    outlier_closed_form: int | None = None
    outlier_free = 0
    outlier_skipped = False
    t_alg4 = 0.0

    if outlier_count > 0:
        outlier_spec = build_outlier_spec(
            grid=grid, outlier_count=outlier_count,
            eq_basics=eq_basics, active_segment_values=active_values, k=ds.k,
        )
        outlier_free_set = (
            set(grid.basics_in(outlier_spec.scope))
            - outlier_spec.excluded_basics
        )
        outlier_free = len(outlier_free_set)
        # Closed-form is exact iff halves disabled AND sparse cap non-binding.
        closed_form_applies = (
            not outlier_spec.halves and outlier_count <= ds.k - 1
        )
        if closed_form_applies and outlier_free > 0:
            outlier_closed_form = comb(
                outlier_count + outlier_free - 1, outlier_free - 1
            )

        if (
            outlier_closed_form is not None
            and outlier_closed_form > outlier_enum_threshold
        ):
            outlier_skipped = True
        else:
            t0 = perf_counter()
            outlier_solutions = run_algorithm4(
                grid=grid,
                outlier_count=outlier_count,
                eq_basics=eq_basics,
                active_segment_values=active_values,
                k=ds.k,
                max_solutions=max_outlier_solutions,
            )
            t_alg4 = perf_counter() - t0

    # ---- Per-EQ + record-level aggregations ----------------------------
    sol_counts = [len(s) for s in eq_solutions]
    n_total = len(sol_counts)
    n_infeasible = sum(1 for c in sol_counts if c == 0)
    n_exact = sum(1 for c in sol_counts if c == 1)
    n_highrisk = sum(1 for c in sol_counts if 2 <= c <= 5)
    n_uncertain = sum(1 for c in sol_counts if c > 5)
    solvable = [c for c in sol_counts if c > 0]
    n_capped = (
        sum(1 for c in sol_counts if max_solutions is not None and c == max_solutions)
        if max_solutions is not None else 0
    )

    rec_infeasible = sum(
        eq.record_count for eq, c in zip(sorted_eqs, sol_counts) if c == 0
    )
    rec_exact = sum(
        eq.record_count for eq, c in zip(sorted_eqs, sol_counts) if c == 1
    )
    rec_highrisk = sum(
        eq.record_count
        for eq, c in zip(sorted_eqs, sol_counts) if 2 <= c <= 5
    )
    rec_uncertain = sum(
        eq.record_count for eq, c in zip(sorted_eqs, sol_counts) if c > 5
    )

    # Quartiles
    if solvable:
        srt = sorted(solvable)
        q25 = srt[max(0, len(srt) // 4 - 1)]
        q75 = srt[min(len(srt) - 1, 3 * len(srt) // 4)]
        s_mean = mean(solvable)
        s_med = median(solvable)
        s_max = max(solvable)
    else:
        q25 = q75 = s_max = None
        s_mean = s_med = None

    # ---- CRA Ratio -----------------------------------------------------
    ratios: list[float] = []
    trivial_one = 0
    refined_to_one = 0
    for i, (eq, seg, sols) in enumerate(
        zip(sorted_eqs, eq_segments, eq_solutions)
    ):
        if not sols:
            continue
        scope_basics = set(grid.basics_in(seg))
        free_count = len(scope_basics - eq_specs_excluded[i])
        if free_count == 0:
            continue
        trivial = comb(eq.record_count + free_count - 1, free_count - 1)
        ratios.append(len(sols) / trivial)
        if trivial == 1:
            trivial_one += 1
        elif len(sols) == 1:
            refined_to_one += 1

    cra_mean = mean(ratios) if ratios else None
    cra_med = median(ratios) if ratios else None
    cra_min = min(ratios) if ratios else None
    cra_max = max(ratios) if ratios else None

    return EvalResult(
        dataset_file=dataset_path.name,
        m=ds.m,
        k=ds.k,
        n=ds.n,
        qi_names=ds.qi_names,
        grid_basics=grid.total_basics,
        eqs_pre_merge=pre_merge,
        eqs_merged=n_total,
        outlier_count=outlier_count,
        eqs_infeasible=n_infeasible,
        eqs_exact=n_exact,
        eqs_highrisk=n_highrisk,
        eqs_uncertain=n_uncertain,
        solvable_mean=s_mean,
        solvable_median=s_med,
        solvable_q25=q25,
        solvable_q75=q75,
        solvable_max=s_max,
        max_solutions_cap=max_solutions,
        eqs_capped=n_capped,
        records_in_eqs=ds.n - outlier_count,
        records_infeasible=rec_infeasible,
        records_exact=rec_exact,
        records_highrisk=rec_highrisk,
        records_uncertain=rec_uncertain,
        cra_ratio_mean=cra_mean,
        cra_ratio_median=cra_med,
        cra_ratio_min=cra_min,
        cra_ratio_max=cra_max,
        eqs_trivially_unique=trivial_one,
        eqs_refined_to_unique=refined_to_one,
        outlier_free_basics=outlier_free,
        outlier_closed_form=outlier_closed_form,
        outlier_enum_skipped=outlier_skipped,
        outlier_solutions_enumerated=len(outlier_solutions),
        wall_time_alg3=t_alg3,
        wall_time_alg4=t_alg4,
    )


# =====================================================================
# Text rendering of EvalResult (used by the CLI).
# =====================================================================


def _banner(s: str) -> None:
    print(f"\n=== {s} ===")


def _fmt_pct(num: int, denom: int) -> str:
    return f"{num} ({100.0 * num / denom:.1f}%)" if denom else f"{num} (n/a)"


def print_report(r: EvalResult, outlier_enum_threshold: int) -> None:
    """Pretty-print an EvalResult to stdout (the CLI's main output)."""
    _banner("Dataset")
    print(f"  file        : {r.dataset_file}")
    print(f"  m (QIs)     : {r.m}  {r.qi_names}")
    print(f"  k           : {r.k}")
    print(f"  n (records) : {r.n}")
    print(f"  grid basics : {r.grid_basics}")

    _banner("CRA run")
    print(f"  pre-merge EQs    : {r.eqs_pre_merge}")
    print(f"  merged EQs       : {r.eqs_merged}")
    print(f"  outliers (|O|)   : {r.outlier_count}")
    if r.outlier_count > 0:
        if r.outlier_enum_skipped:
            label = (
                f"{r.outlier_closed_form:,} solutions (closed-form; "
                f"enumeration skipped, threshold={outlier_enum_threshold:,})"
            )
        elif r.outlier_closed_form is not None:
            label = (
                f"{r.outlier_solutions_enumerated:,} enumerated "
                f"(closed-form check: {r.outlier_closed_form:,})"
            )
        else:
            label = f"{r.outlier_solutions_enumerated:,} enumerated"
        print(f"  outlier LP       : {label}")
    print(f"  wall time (Alg 3): {r.wall_time_alg3:.2f} s")
    print(f"  wall time (Alg 4): {r.wall_time_alg4:.2f} s")
    print(f"  wall time (total): {r.wall_time_total:.2f} s")

    _banner("Attack precision (per EQ)")
    print(f"  total EQs                       : {r.eqs_merged}")
    print(f"  infeasible    (= 0 solutions)   : {_fmt_pct(r.eqs_infeasible, r.eqs_merged)}")
    print(f"  Exact Breach  (= 1 solution)    : {_fmt_pct(r.eqs_exact, r.eqs_merged)}")
    print(f"  High-Risk     (2-5 solutions)   : {_fmt_pct(r.eqs_highrisk, r.eqs_merged)}")
    print(f"  Uncertain     (> 5 solutions)   : {_fmt_pct(r.eqs_uncertain, r.eqs_merged)}")
    if r.max_solutions_cap is not None and r.eqs_capped:
        print(
            f"  (of those, {r.eqs_capped} EQs hit the per-EQ cap of "
            f"{r.max_solutions_cap}; their solution counts are LOWER BOUNDS)"
        )
    if r.solvable_mean is not None:
        print(
            f"  solutions per solvable EQ  -> "
            f"mean={r.solvable_mean:.2f}  median={r.solvable_median:.1f}  "
            f"Q25={r.solvable_q25}  Q75={r.solvable_q75}  max={r.solvable_max}"
        )

    _banner("Attack precision (records in standard EQs)")
    print(f"  records in standard EQs                : {r.records_in_eqs}")
    print(f"  in infeasible EQs                      : {_fmt_pct(r.records_infeasible, r.records_in_eqs)}")
    print(f"  in Exact-Breach EQs (1 solution)       : {_fmt_pct(r.records_exact, r.records_in_eqs)}")
    print(f"  in High-Risk EQs (2-5 solutions)       : {_fmt_pct(r.records_highrisk, r.records_in_eqs)}")
    print(f"  in Uncertain EQs (>5 solutions)        : {_fmt_pct(r.records_uncertain, r.records_in_eqs)}")

    if r.cra_ratio_mean is not None:
        _banner("CRA Ratio (enumerated / trivial stars-and-bars)")
        print(f"  EQs counted                            : {r.eqs_merged - r.eqs_infeasible}")
        print(
            f"  EQs already unique without CRA         : {r.eqs_trivially_unique} "
            f"(trivial==1, no LP refinement possible)"
        )
        print(f"  EQs refined CRA (trivial>1 -> CRA==1)  : {r.eqs_refined_to_unique}")
        print(f"  mean ratio                             : {r.cra_ratio_mean:.4f}")
        print(f"  median ratio                           : {r.cra_ratio_median:.4f}")
        print(f"  min ratio                              : {r.cra_ratio_min:.4f}  (strongest refinement)")
        print(f"  max ratio                              : {r.cra_ratio_max:.4f}")

    _banner("Outliers (Algorithm 4)")
    print(f"  |O|                                    : {r.outlier_count}")
    if r.outlier_count > 0:
        print(f"  free outlier basics                    : {r.outlier_free_basics}")
        if r.outlier_closed_form is not None:
            print(f"  closed-form solution count             : {r.outlier_closed_form:,}")
        if r.outlier_enum_skipped:
            print(
                f"  enumeration skipped (closed-form > "
                f"--outlier-enum-threshold={outlier_enum_threshold:,})"
            )
        else:
            print(f"  outlier-LP solutions enumerated        : {r.outlier_solutions_enumerated:,}")
            if r.outlier_solutions_enumerated == 1:
                print("  --> outlier set is UNIQUELY determined (exact breach).")

    _banner("Summary line")
    cra_ratio = f"{r.cra_ratio_mean:.4f}" if r.cra_ratio_mean is not None else "n/a"
    print(
        f"  m={r.m}, k={r.k}, n={r.n}, "
        f"eqs={r.eqs_merged}, |O|={r.outlier_count}, "
        f"exact_eqs%={r.exact_eq_pct:.1f}, "
        f"exact_records%={r.exact_record_pct:.1f}, "
        f"cra_ratio_mean={cra_ratio}, "
        f"wall={r.wall_time_total:.1f}s"
    )


# =====================================================================
# CLI.
# =====================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    req = p.add_argument_group("required")
    req.add_argument(
        "--dataset", required=True, type=Path, metavar="CSV",
        help="Path to the ARX-anonymised CSV (D_gen).",
    )
    req.add_argument(
        "--hier-dir", required=True, type=Path, metavar="DIR",
        help="Directory containing the generalisation-hierarchy CSVs "
             "(one per QI, named hierarchy_<qi>.csv).",
    )
    req.add_argument(
        "-k", required=True, type=int, metavar="K",
        help="The k-anonymity parameter used when ARX produced D_gen.",
    )
    req.add_argument(
        "--qi",
        nargs="+",
        required=True,
        help="Names of quasi-identifier columns in the dataset"
    )

    opt = p.add_argument_group("optional")
    opt.add_argument(
        "--qi-columns", nargs="+", default=None, metavar="NAME",
        help="Override QI auto-detection. List the QI column header names "
             "in the order they should be treated. By default any column "
             "whose every value is an interval like \"[a, b[\" or the "
             "suppression marker '*' is treated as a QI.",
    )
    opt.add_argument(
        "--max-solutions", type=int, default=None, metavar="N",
        help="Cap CP-SAT enumeration to N solutions per EQ. Default: no cap. "
             "REQUIRED in practice for m >= 4 (we recommend N=10000); "
             "without it a single broad-state EQ can take hours.",
    )
    opt.add_argument(
        "--max-outlier-solutions", type=int, default=None, metavar="N",
        help="Cap CP-SAT enumeration for the outlier LP (Algorithm 4). "
             "Ignored when the closed-form bypass triggers.",
    )
    opt.add_argument(
        "--outlier-enum-threshold", type=int, default=200_000, metavar="N",
        help="If the outlier LP is in the closed-form regime "
             "(halves disabled AND |O| <= k-1) AND its stars-and-bars "
             "solution count exceeds N, skip enumeration and report only "
             "the count. Default 200000.",
    )
    opt.add_argument(
        "--verbose", action="store_true",
        help="Print per-EQ progress lines while Algorithm 3 runs.",
    )
    opt.add_argument(
        "--progress-every", type=int, default=20, metavar="N",
        help="When --verbose, emit a progress line every N EQs. Default 20.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    progress_cb = None
    if args.verbose:
        every = max(1, args.progress_every)
        def progress_cb(i: int, total: int, elapsed: float) -> None:
            if i % every == 0:
                print(
                    f"  progress: {i}/{total}  elapsed={elapsed:.1f}s",
                    flush=True,
                )

    result = evaluate_dataset(
        dataset_path=args.dataset,
        hier_dir=args.hier_dir,
        k=args.k,
        qi_columns=args.qi,
        max_solutions=args.max_solutions,
        max_outlier_solutions=args.max_outlier_solutions,
        outlier_enum_threshold=args.outlier_enum_threshold,
        progress_callback=progress_cb,
    )
    print_report(result, outlier_enum_threshold=args.outlier_enum_threshold)


if __name__ == "__main__":
    main()
