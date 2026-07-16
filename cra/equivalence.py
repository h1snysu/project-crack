"""Equivalence-class extraction and sorting.

Implements Algorithm 3 lines 2-4 *plus* outlier separation (Algorithm 4 input):

  * Records whose generalisation state equals (h^1, ..., h^m) -- i.e. every QI
    at the root layer -- are ARX-LR "outliers" (fully-suppressed records).
    They are aggregated into `outlier_count` and removed from the normal EQ
    list so Algorithm 4 can process them on its own.

  * Metrics (info_loss, c1, c2) are functions of the generalisation state
    (qi_layers) alone, so they are computed once per unique state and reused
    for every EQ sharing that state.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Iterable

from dataset import Dataset
from hierarchy import Hierarchy
from models import EquivalenceClass, HierarchyNode
from segment import Segment


# (info_loss, c1, c2) keyed by qi_layers tuple.
StateMetrics = tuple[float, int, float]


def _compute_state_metrics(
    qi_layers: tuple[int, ...],
    qi_nodes: tuple[HierarchyNode, ...],
    hierarchies: list[Hierarchy],
) -> StateMetrics:
    """Eq. 2 (info_loss) + c1 + c2 for a single generalisation state."""
    m = len(qi_layers)
    if m == 0:
        return 0.0, 0, 0.0
    prod = 1.0
    for n in qi_nodes:
        prod *= (n.q_loss + 1.0)
    info_loss = prod ** (1.0 / m) - 1.0
    c1 = sum(qi_layers)
    c2 = sum(l / h.height for l, h in zip(qi_layers, hierarchies)) / m
    return info_loss, c1, c2


@dataclass
class ExtractionResult:
    """Output of `extract_equivalence_classes`.

    `eqs`            - standard EQs (already sorted is the caller's job)
    `outlier_count`  - |O|, total records at the fully-suppressed state
    `state_metrics`  - cache used during extraction, kept for downstream re-use
                       (e.g. printing or debugging).
    """

    eqs: list[EquivalenceClass]
    outlier_count: int
    state_metrics: dict[tuple[int, ...], StateMetrics]


def extract_equivalence_classes(
    dataset: Dataset, hierarchies: list[Hierarchy]
) -> ExtractionResult:
    """Group D_gen rows, score each unique generalisation state once,
    and split off fully-suppressed records as outliers."""
    if len(hierarchies) != dataset.m:
        raise ValueError(
            f"Number of hierarchies ({len(hierarchies)}) does not match "
            f"number of QIs in dataset ({dataset.m})"
        )

    counts: dict[tuple[str, ...], int] = {}
    nodes_cache: dict[tuple[str, ...], tuple[HierarchyNode, ...]] = {}

    for row in dataset.rows:
        if row not in nodes_cache:
            nodes_cache[row] = tuple(
                h.resolve(cell) for cell, h in zip(row, hierarchies)
            )
        counts[row] = counts.get(row, 0) + 1

    # The fully-suppressed generalisation state: every QI at its root layer.
    outlier_state: tuple[int, ...] = tuple(h.height for h in hierarchies)

    state_metrics: dict[tuple[int, ...], StateMetrics] = {}
    eqs: list[EquivalenceClass] = []
    outlier_count = 0

    for key, cnt in counts.items():
        nodes = nodes_cache[key]
        qi_layers = tuple(n.layer for n in nodes)
        if qi_layers == outlier_state:
            outlier_count += cnt
            continue
        metrics = state_metrics.get(qi_layers)
        if metrics is None:
            metrics = _compute_state_metrics(qi_layers, nodes, hierarchies)
            state_metrics[qi_layers] = metrics
        info_loss, c1, c2 = metrics
        eqs.append(
            EquivalenceClass(
                qi_values=key,
                qi_nodes=nodes,
                record_count=cnt,
                info_loss=info_loss,
                c1=c1,
                c2=c2,
            )
        )

    # Sanity: every record is accounted for.
    total = sum(eq.record_count for eq in eqs) + outlier_count
    if total != dataset.n:
        raise ValueError(
            f"EQ extraction lost records: {total} vs dataset n={dataset.n}"
        )
    # Outliers are *expected* to fall under k; warn only on normal EQs.
    under_k = [eq for eq in eqs if eq.record_count < dataset.k]
    if under_k:
        warnings.warn(
            f"{len(under_k)} non-outlier EQ(s) have |EQ| < k={dataset.k}; "
            f"smallest = {min(eq.record_count for eq in under_k)}. "
            f"Input is not strictly k-anonymous."
        )
    bad_loss = [eq for eq in eqs if not (0.0 - 1e-9 <= eq.info_loss <= 1.0 + 1e-9)]
    if bad_loss:
        warnings.warn(
            f"{len(bad_loss)} EQ(s) have info_loss outside [0,1]; "
            f"check hierarchy q_loss values."
        )

    return ExtractionResult(
        eqs=eqs, outlier_count=outlier_count, state_metrics=state_metrics
    )


def sort_equivalence_classes(
    eqs: Iterable[EquivalenceClass],
) -> list[EquivalenceClass]:
    """Sort by (info_loss, c1, c2) ascending; final tie-break on qi_layers so
    EQs sharing the same generalisation state stay grouped (matches ARX-LR's
    state-by-state iteration loop)."""
    return sorted(eqs, key=lambda e: (e.info_loss, e.c1, e.c2, e.qi_layers))


def merge_by_snapped_segment(
    sorted_eqs: list[EquivalenceClass],
) -> list[EquivalenceClass]:
    """Collapse EQs whose snapped segment (basic-segment grid footprint) is
    identical into a single EQ whose `record_count` is the sum.

    ARX-LR can emit two distinct EQs whose pre-snap `qi_values` differ but
    whose layer-1 footprints coincide -- e.g. ('58', '123') (state 0/0) and
    ('[55, 60[', '[120, 130[') (state 1/1) both snap to the same basic
    `('[55, 60[', '[120, 130[')`. At the grid level they are the same EQ:
    their records all live in the same z_basic and the LP cannot tell them
    apart, so we merge them up-front.

    The retained EQ keeps the first (lowest sort-key) occurrence's
    `qi_values`, `qi_nodes`, `info_loss`, `c1`, `c2` -- only `record_count`
    accumulates. Input ordering is preserved.
    """
    seen: dict[tuple[str, ...], int] = {}
    out: list[EquivalenceClass] = []
    for eq in sorted_eqs:
        snap = Segment(eq.qi_nodes).qi_values
        if snap in seen:
            i = seen[snap]
            out[i] = replace(
                out[i], record_count=out[i].record_count + eq.record_count
            )
        else:
            seen[snap] = len(out)
            out.append(eq)
    return out
