"""Unified CP-SAT enumerator for both Algorithm 3 (per EQ) and Algorithm 4
(outliers).

The two algorithms differ only in which segment is the LP "scope", which
basic segments are excluded (overlap), and whether the halves constraint is
applied. Everything else -- variable set, total sum, halves >= 1, sparse
<= k-1 -- has identical shape, so we express it once.

Usage (later phases will build the spec):

    spec = LPSpec(
        grid=grid,
        scope=eq_segment,
        total=eq.record_count,
        k=dataset.k,
        excluded_basics=overlap_basics,
        halves=[h for h in eq_segment.iter_half_segments() if ...],
        sparse=[s for s in eq_segment.iter_descendant_segments(include_self=False) if ...],
    )
    solutions = SegmentLP.solve(spec)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from grid import Grid
from segment import Basic, Segment


@dataclass
class LPSpec:
    """Inputs that fully determine one CP-SAT enumeration run."""

    grid: Grid
    scope: Segment
    total: int
    k: int
    # Basic segments inside `scope` that must be 0 (overlap with prior EQs).
    excluded_basics: set[Basic] = field(default_factory=set)
    # Half-segments of `scope` that must each sum to >= 1.
    halves: list[Segment] = field(default_factory=list)
    # Sub-segments of `scope` that must each sum to <= k - 1.
    sparse: list[Segment] = field(default_factory=list)
    # Optional cap; None means "enumerate all".
    max_solutions: int | None = None
    # If True, returned dicts include zero-valued basics too. Default: omit zeros.
    include_zeros: bool = False
    # Optional CP-SAT wall-clock limit (seconds) so one hard EQ cannot hang the
    # sweep. None means no limit. When the limit is hit enumeration stops early
    # and the returned list is a (possibly partial) lower bound on the count.
    max_time_seconds: float | None = None


# A solution maps each free basic to its non-negative integer count.
Solution = dict[Basic, int]


class _AllSolutionsCallback(cp_model.CpSolverSolutionCallback):
    """Collects every solution CP-SAT finds (under enumerate_all_solutions)."""

    def __init__(
        self,
        z_vars: dict[Basic, cp_model.IntVar],
        *,
        max_solutions: int | None,
        include_zeros: bool,
    ) -> None:
        super().__init__()
        self._z = z_vars
        self._max = max_solutions
        self._include_zeros = include_zeros
        self.solutions: list[Solution] = []

    def on_solution_callback(self) -> None:
        sol: Solution = {}
        for b, var in self._z.items():
            v = self.Value(var)
            if v > 0 or self._include_zeros:
                sol[b] = v
        self.solutions.append(sol)
        if self._max is not None and len(self.solutions) >= self._max:
            self.StopSearch()


class SegmentLP:
    """Stateless CP-SAT runner. Build an `LPSpec`, call `solve(spec)`."""

    @staticmethod
    def solve(spec: LPSpec) -> list[Solution]:
        if spec.total < 0:
            raise ValueError(f"LPSpec.total must be >= 0, got {spec.total}")
        if spec.k < 1:
            raise ValueError(f"LPSpec.k must be >= 1, got {spec.k}")

        scope_basics = set(spec.grid.basics_in(spec.scope))
        free_basics = sorted(scope_basics - spec.excluded_basics)

        # Degenerate: no free variables. Only consistent if total is also 0.
        if not free_basics:
            if spec.total == 0:
                return [{}] if not spec.halves else []
            return []

        model = cp_model.CpModel()
        z: dict[Basic, cp_model.IntVar] = {}
        for idx, b in enumerate(free_basics):
            z[b] = model.NewIntVar(0, spec.total, f"z_{idx}")

        # Algorithm 3 line 13 / Algorithm 4: total sum constraint.
        model.Add(sum(z.values()) == spec.total)

        # Halves: sum over half's basics >= 1.
        for half in spec.halves:
            half_basics = set(spec.grid.basics_in(half)) & set(free_basics)
            if not half_basics:
                # No free basic can satisfy this lower bound -> infeasible.
                return []
            model.Add(sum(z[b] for b in half_basics) >= 1)

        # Sparse: sum over sub-segment's basics <= k - 1.
        sparse_bound = spec.k - 1
        for sub in spec.sparse:
            sub_basics = set(spec.grid.basics_in(sub)) & set(free_basics)
            if not sub_basics:
                continue
            model.Add(sum(z[b] for b in sub_basics) <= sparse_bound)

        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        # Single-threaded enumeration is the only mode that yields every solution.
        solver.parameters.num_search_workers = 1
        if spec.max_time_seconds is not None:
            solver.parameters.max_time_in_seconds = float(spec.max_time_seconds)

        callback = _AllSolutionsCallback(
            z, max_solutions=spec.max_solutions,
            include_zeros=spec.include_zeros,
        )
        solver.Solve(model, callback)
        return callback.solutions
