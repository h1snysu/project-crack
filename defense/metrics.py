"""Run the CRA on an anonymized CSV and compute every metric the experiment
needs, including per-EQ detail the existing `eval_cra.EvalResult` does not
expose (LR vs CRA solution counts, z_i single-out proxy, integer-assignment
counts, per-EQ timing/timeouts).

This mirrors the loop in `eval_cra.evaluate_dataset` but keeps the per-EQ
solution lists so we can derive the z_i-based metrics, and threads through a
per-EQ wall-clock limit (`cra_timeout_seconds`) and an EQ cap (`max_eqs`).
"""
from __future__ import annotations

from math import comb
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Sequence

from cra import build_eq_spec, build_outlier_spec, run_algorithm4
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


def _pct(num: int, denom: int) -> float:
    return 100.0 * num / denom if denom else 0.0


def run_cra_eval(
    dataset_path: Path,
    hiers: Sequence[Hierarchy],
    k: int,
    *,
    qi_columns: Sequence[str] | None = None,
    max_solutions: int | None = None,
    max_eqs: int | None = None,
    cra_timeout_seconds: float | None = None,
    max_outlier_solutions: int | None = None,
    outlier_enum_threshold: int = 200_000,
) -> tuple[dict, list[dict]]:
    """Evaluate one anonymized CSV. Returns (summary_dict, per_eq_rows)."""
    ds = Dataset.from_csv(
        dataset_path, k=k,
        qi_columns=list(qi_columns) if qi_columns is not None else None,
    )
    grid = Grid(list(hiers))

    extraction = extract_equivalence_classes(ds, list(hiers))
    pre_merge = len(extraction.eqs)
    sorted_eqs = merge_by_snapped_segment(sort_equivalence_classes(extraction.eqs))

    eqs_total_after_merge = len(sorted_eqs)
    eqs_skipped = 0
    if max_eqs is not None and len(sorted_eqs) > max_eqs:
        eqs_skipped = len(sorted_eqs) - max_eqs
        sorted_eqs = sorted_eqs[:max_eqs]

    eq_segments = [Segment(eq.qi_nodes) for eq in sorted_eqs]
    eq_basics = [set(grid.basics_in(s)) for s in eq_segments]
    active_values = {s.qi_values for s in eq_segments}

    per_eq: list[dict] = []
    t_cra0 = perf_counter()
    for i, (eq, seg) in enumerate(zip(sorted_eqs, eq_segments)):
        spec = build_eq_spec(
            grid=grid, eq=eq, eq_segment=seg,
            prior_eq_basics=eq_basics[:i],
            active_segment_values=active_values, k=ds.k,
            max_solutions=max_solutions,
        )
        spec.max_time_seconds = cra_timeout_seconds
        t0 = perf_counter()
        sols = SegmentLP.solve(spec)
        dt = perf_counter() - t0

        scope_basics = set(grid.basics_in(seg))
        free_count = len(scope_basics - spec.excluded_basics)
        lr_solutions = comb(eq.record_count + free_count - 1, free_count - 1) if free_count > 0 else 1
        cra_solutions = len(sols)
        ratio = (lr_solutions / cra_solutions) if cra_solutions > 0 else None
        has_zi1 = any(v == 1 for s in sols for v in s.values())
        capped = max_solutions is not None and cra_solutions >= max_solutions
        timed_out = (
            cra_timeout_seconds is not None and dt >= 0.95 * cra_timeout_seconds
        )
        per_eq.append({
            "eq_index": i,
            "qi_values": "|".join(eq.qi_values),
            "qi_layers": "/".join(str(l) for l in eq.qi_layers),
            "record_count": eq.record_count,
            "is_basic": seg.is_basic,
            "info_loss": round(eq.info_loss, 6),
            "free_basics": free_count,
            "lr_solutions": lr_solutions,
            "cra_solutions": cra_solutions,
            "cra_ratio": round(ratio, 6) if ratio is not None else None,
            "cra_int_assignments": cra_solutions,
            "has_zi_eq_1": has_zi1,
            "capped": capped,
            "timed_out": timed_out,
            "solve_seconds": round(dt, 4),
        })
    t_cra = perf_counter() - t_cra0

    # Outliers (Algorithm 4) -- reuse the existing closed-form bypass logic.
    outlier_count = extraction.outlier_count
    outlier_closed_form = None
    outlier_enum = 0
    outlier_skipped = False
    if outlier_count > 0 and max_eqs is None:
        ospec = build_outlier_spec(
            grid=grid, outlier_count=outlier_count, eq_basics=eq_basics,
            active_segment_values=active_values, k=ds.k,
        )
        free_set = set(grid.basics_in(ospec.scope)) - ospec.excluded_basics
        if not ospec.halves and outlier_count <= ds.k - 1 and len(free_set) > 0:
            outlier_closed_form = comb(outlier_count + len(free_set) - 1, len(free_set) - 1)
        if outlier_closed_form is not None and outlier_closed_form > outlier_enum_threshold:
            outlier_skipped = True
        else:
            osols = run_algorithm4(
                grid=grid, outlier_count=outlier_count, eq_basics=eq_basics,
                active_segment_values=active_values, k=ds.k,
                max_solutions=max_outlier_solutions,
            )
            outlier_enum = len(osols)

    # ---- aggregate summary ----
    n_eqs = len(per_eq)
    n_basic = sum(1 for e in per_eq if e["is_basic"])
    n_compound = n_eqs - n_basic
    n_infeasible = sum(1 for e in per_eq if e["cra_solutions"] == 0)
    n_exact = sum(1 for e in per_eq if e["cra_solutions"] == 1)
    n_capped = sum(1 for e in per_eq if e["capped"])
    n_timed_out = sum(1 for e in per_eq if e["timed_out"])
    n_single_out = sum(1 for e in per_eq if e["has_zi_eq_1"])

    ratios = [e["cra_ratio"] for e in per_eq if e["cra_ratio"] is not None]
    cra_int = [e["cra_int_assignments"] for e in per_eq]
    losses = [e["info_loss"] for e in per_eq]
    rec_weighted_loss = (
        sum(e["info_loss"] * e["record_count"] for e in per_eq)
        / sum(e["record_count"] for e in per_eq)
        if per_eq else None
    )

    summary = {
        "n_records_anonymized": ds.n,
        "hierarchy_layers": "/".join(str(h.height) for h in hiers),
        "eqs_pre_merge": pre_merge,
        "eqs_total_after_merge": eqs_total_after_merge,
        "eqs_evaluated": n_eqs,
        "eqs_skipped_by_max_eqs": eqs_skipped,
        "basic_segment_eqs": n_basic,
        "compound_segment_eqs": n_compound,
        "outlier_count": outlier_count,
        "outlier_closed_form": outlier_closed_form,
        "outlier_enumerated": outlier_enum,
        "outlier_enum_skipped": outlier_skipped,
        "eqs_infeasible": n_infeasible,
        "eqs_exact": n_exact,
        "exact_eq_pct": round(_pct(n_exact, n_eqs), 3),
        "eqs_capped": n_capped,
        "eqs_timed_out": n_timed_out,
        "avg_info_loss_record_weighted": round(rec_weighted_loss, 6) if rec_weighted_loss is not None else None,
        "median_info_loss": round(median(losses), 6) if losses else None,
        "cra_ratio_mean": round(mean(ratios), 6) if ratios else None,
        "cra_ratio_median": round(median(ratios), 6) if ratios else None,
        "cra_ratio_max": round(max(ratios), 6) if ratios else None,
        "pct_eqs_ratio_le_1_05": round(_pct(sum(1 for r in ratios if r <= 1.05), len(ratios)), 3) if ratios else None,
        "pct_eqs_ratio_le_2": round(_pct(sum(1 for r in ratios if r <= 2.0), len(ratios)), 3) if ratios else None,
        "cra_int_assignments_median": round(median(cra_int), 3) if cra_int else None,
        "cra_int_assignments_mean": round(mean(cra_int), 3) if cra_int else None,
        "pct_eqs_with_single_out": round(_pct(n_single_out, n_eqs), 3),
        "single_out_risk_proxy": round(_pct(n_single_out, n_eqs), 3),
        "cra_seconds": round(t_cra, 4),
    }
    return summary, per_eq
