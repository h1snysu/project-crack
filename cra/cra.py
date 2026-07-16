"""End-to-end orchestrator for the Combinatorial Refinement Attack.

This module wires Phase 2 (EQ extraction) and Phase 3 (Grid + Solver) together
to execute Algorithm 3 of the paper for every standard EQ. Algorithm 4
(outliers) will be added in a follow-up step.

Per-EQ LPSpec construction (Algorithm 3 lines 6-19):

  * `scope`          := Segment(eq.qi_nodes)  -- auto-snaps any layer-0 dim
                        to its layer-1 parent.
  * `excluded_basics` := union over PRIOR EQs of (their basics ∩ scope.basics),
                        i.e. line 7 (basics outside B*) is implicit because
                        we only create variables for basics in scope, and line
                        10 (overlap with prior EQ') is the explicit exclusion.
  * `halves`         := scope.iter_half_segments() filtered to NON-active
                        sub-EQs (a half that exactly equals another EQ is
                        already accounted for by that EQ's own LP).
  * `sparse`         := scope.iter_descendant_segments(include_self=False)
                        filtered to NON-active sub-EQs (line 18 with the
                        "active sub-EQ" refinement).

Two segments are "the same EQ" when their snapped `qi_values` match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dataset import Dataset
from equivalence import (
    ExtractionResult,
    extract_equivalence_classes,
    merge_by_snapped_segment,
    sort_equivalence_classes,
)
from grid import Grid
from hierarchy import Hierarchy
from models import EquivalenceClass
from segment import Basic, Segment
from solver import LPSpec, SegmentLP, Solution


@dataclass
class CRAResult:
    """Output of one full CRA pass (Algorithm 3 for EQs + Algorithm 4 for outliers)."""

    sorted_eqs: list[EquivalenceClass]
    eq_segments: list[Segment]  # parallel to sorted_eqs (post-snap scopes)
    eq_solutions: list[list[Solution]]  # parallel to sorted_eqs
    outlier_count: int
    outlier_solutions: list[Solution] = field(default_factory=list)
    # Pre-computed per-EQ basic-segment sets, useful for downstream Alg 4.
    eq_basics: list[set[Basic]] = field(default_factory=list)
    # How many EQs ARX emitted before snap-merge (for diagnostics).
    pre_merge_eq_count: int = 0


def build_eq_spec(
    grid: Grid,
    eq: EquivalenceClass,
    eq_segment: Segment,
    prior_eq_basics: list[set[Basic]],
    active_segment_values: set[tuple[str, ...]],
    k: int,
    max_solutions: int | None = None,
) -> LPSpec:
    """Construct the LPSpec for one EQ per Algorithm 3 lines 6-19."""
    scope_basics = set(grid.basics_in(eq_segment))

    # Overlap (line 8-12): exclude basics already claimed by any prior EQ.
    excluded: set[Basic] = set()
    for prior in prior_eq_basics:
        excluded |= prior & scope_basics

    # Halves + sparse: walk sub-segments once, classify each.
    halves_set: set[Segment] = set(eq_segment.iter_half_segments())
    halves: list[Segment] = []
    sparse: list[Segment] = []
    for sub in eq_segment.iter_descendant_segments(include_self=False):
        # Active-sub-EQ exemption: another EQ in sorted_eqs exactly matches
        # this sub-segment -> it is accounted for by its own LP; skip both
        # halves and sparse constraints to avoid forced infeasibility.
        if sub.qi_values in active_segment_values:
            continue
        if sub in halves_set:
            halves.append(sub)
        sparse.append(sub)

    return LPSpec(
        grid=grid,
        scope=eq_segment,
        total=eq.record_count,
        k=k,
        excluded_basics=excluded,
        halves=halves,
        sparse=sparse,
        max_solutions=max_solutions,
    )


def build_outlier_spec(
    grid: Grid,
    outlier_count: int,
    eq_basics: list[set[Basic]],
    active_segment_values: set[tuple[str, ...]],
    k: int,
    max_solutions: int | None = None,
) -> LPSpec:
    """Construct the LPSpec for the outlier LP per Algorithm 4.

    * scope    := the entire grid G (every QI at root).
    * total    := outlier_count (= |O|).
    * excluded := union of every standard EQ's basics (outliers run last).
    * halves   := every grid-half ONLY IF outlier_count >= k; else [].
    * sparse   := every sub-segment of G not matching an active standard EQ,
                  capped at k - 1.
    """
    scope = grid.grid_segment

    # All basics claimed by any standard EQ are excluded.
    excluded: set[Basic] = set()
    for eq_b in eq_basics:
        excluded |= eq_b

    # Halves: only if there are at least k outliers (matches user spec).
    halves: list[Segment] = []
    if outlier_count >= k:
        for half in scope.iter_half_segments():
            # Mirror the Alg-3 exemption: skip halves that exactly match an
            # active standard EQ (those basics are already excluded; the
            # constraint would be infeasible by construction).
            if half.qi_values in active_segment_values:
                continue
            halves.append(half)

    halves_set: set[Segment] = set(halves)

    # Sparse: walk every sub-segment of G except G itself; skip active EQs.
    sparse: list[Segment] = []
    for sub in scope.iter_descendant_segments(include_self=False):
        if sub.qi_values in active_segment_values:
            continue
        sparse.append(sub)

    # Sanity: halves are a subset of sparse (each half is also a sub-segment).
    # No special handling needed; the LP allows both bounds on the same sub.

    return LPSpec(
        grid=grid,
        scope=scope,
        total=outlier_count,
        k=k,
        excluded_basics=excluded,
        halves=halves,
        sparse=sparse,
        max_solutions=max_solutions,
    )


def run_algorithm4(
    grid: Grid,
    outlier_count: int,
    eq_basics: list[set[Basic]],
    active_segment_values: set[tuple[str, ...]],
    k: int,
    max_solutions: int | None = None,
) -> list[Solution]:
    """Run Algorithm 4 on the outlier set. Returns [] if outlier_count == 0."""
    if outlier_count <= 0:
        return []
    spec = build_outlier_spec(
        grid=grid,
        outlier_count=outlier_count,
        eq_basics=eq_basics,
        active_segment_values=active_segment_values,
        k=k,
        max_solutions=max_solutions,
    )
    return SegmentLP.solve(spec)


def run_algorithm3(
    grid: Grid,
    sorted_eqs: list[EquivalenceClass],
    k: int,
    max_solutions: int | None = None,
) -> tuple[list[Segment], list[set[Basic]], list[list[Solution]]]:
    """Run Algorithm 3 over `sorted_eqs`. Returns per-EQ (segment, basics, solutions)."""
    eq_segments: list[Segment] = [Segment(eq.qi_nodes) for eq in sorted_eqs]
    eq_basics: list[set[Basic]] = [
        set(grid.basics_in(s)) for s in eq_segments
    ]
    active_values: set[tuple[str, ...]] = {s.qi_values for s in eq_segments}

    eq_solutions: list[list[Solution]] = []
    for i, (eq, seg) in enumerate(zip(sorted_eqs, eq_segments)):
        spec = build_eq_spec(
            grid=grid,
            eq=eq,
            eq_segment=seg,
            prior_eq_basics=eq_basics[:i],
            active_segment_values=active_values,
            k=k,
            max_solutions=max_solutions,
        )
        eq_solutions.append(SegmentLP.solve(spec))
    return eq_segments, eq_basics, eq_solutions


def run_cra(
    dataset: Dataset,
    hierarchies: list[Hierarchy],
    max_solutions_per_eq: int | None = None,
    max_outlier_solutions: int | None = None,
) -> CRAResult:
    """Full CRA driver: extract + sort + snap-merge EQs, run Algorithm 3 over
    EQs, then Algorithm 4 over the outliers."""
    extraction: ExtractionResult = extract_equivalence_classes(
        dataset, hierarchies
    )
    sorted_eqs = sort_equivalence_classes(extraction.eqs)
    pre_merge = len(sorted_eqs)
    merged_eqs = merge_by_snapped_segment(sorted_eqs)
    grid = Grid(hierarchies)
    eq_segments, eq_basics, eq_solutions = run_algorithm3(
        grid=grid,
        sorted_eqs=merged_eqs,
        k=dataset.k,
        max_solutions=max_solutions_per_eq,
    )
    active_values: set[tuple[str, ...]] = {s.qi_values for s in eq_segments}
    outlier_solutions = run_algorithm4(
        grid=grid,
        outlier_count=extraction.outlier_count,
        eq_basics=eq_basics,
        active_segment_values=active_values,
        k=dataset.k,
        max_solutions=max_outlier_solutions,
    )
    return CRAResult(
        sorted_eqs=merged_eqs,
        eq_segments=eq_segments,
        eq_solutions=eq_solutions,
        outlier_count=extraction.outlier_count,
        outlier_solutions=outlier_solutions,
        eq_basics=eq_basics,
        pre_merge_eq_count=pre_merge,
    )
