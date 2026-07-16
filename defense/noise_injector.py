"""Fake-record injection strategies (the "defense").

A record is a dict keyed by CSV header. Injection adds fake rows and tags every
row with a marker column (`__fake__` = 0/1). The marker is never a QI -- the CRA
reader detects QIs by interval shape and we always pass `qi_columns` explicitly,
so a 0/1 column is ignored downstream. `fake_records.csv` is also written as a
sidecar, so the pipeline runs identically with or without reading the marker.

Strategies
----------
no_noise              : baseline, returns the input unchanged.
random_noise          : `noise_budget` fakes, each QI drawn uniformly from the
                        QI's valid raw domain. Naive baseline.
sparse_basic_noise    : preferred. Find basic segments with 0 < count < k and
                        inject up to `max_noise_per_segment` fakes into each
                        (ascending count first) until `noise_budget` is hit.
                        Fake QI values are drawn from the raw points *inside*
                        the chosen basic segment, so they stay valid and land
                        in that segment.
targeted_overlap_noise: like sparse_basic but prioritizes sparse segments that
                        share a layer-2 parent with populated neighbours (more
                        likely to create/modify overlaps during generalization).

All randomness flows through a seeded `random.Random` for reproducibility.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from hierarchy import Hierarchy
from .segment_utils import (
    all_raw_values,
    basic_segment_of,
    raw_values_under_basic,
    segment_counts,
    sparse_segments,
)

MARKER = "__fake__"
STRATEGIES = (
    "no_noise",
    "random_noise",
    "sparse_basic_noise",
    "targeted_overlap_noise",
)

Row = dict[str, str]


@dataclass
class InjectionResult:
    rows: list[Row]          # real rows + fake rows, each carrying MARKER
    fake_rows: list[Row]     # the fakes only
    header: list[str]        # header including MARKER
    target_segments: list[tuple[str, ...]] = field(default_factory=list)
    per_segment_added: dict[tuple[str, ...], int] = field(default_factory=dict)

    @property
    def n_real(self) -> int:
        return len(self.rows) - len(self.fake_rows)

    @property
    def n_fake(self) -> int:
        return len(self.fake_rows)


def _is_id_col(name: str) -> bool:
    n = name.strip().lower()
    return n in {"patient", "id", "name"} or n.endswith("_id") or "uuid" in n


def _fill_non_qi(
    header: Sequence[str],
    qi_cols: Sequence[str],
    real_rows: Sequence[Row],
    rng: random.Random,
    fake_idx: int,
) -> Row:
    """Build the non-QI part of a fake row: synthetic id for id-like columns,
    a value sampled from the real distribution otherwise."""
    row: Row = {}
    for col in header:
        if col in qi_cols or col == MARKER:
            continue
        if _is_id_col(col):
            row[col] = f"FAKE-{fake_idx:06d}"
        elif real_rows:
            row[col] = rng.choice(real_rows)[col]
        else:
            row[col] = ""
    return row


def _make_fake(
    seg_or_none: tuple[str, ...] | None,
    qi_cols: Sequence[str],
    hiers: Sequence[Hierarchy],
    header: Sequence[str],
    real_rows: Sequence[Row],
    rng: random.Random,
    fake_idx: int,
) -> Row:
    """One fake row. If `seg_or_none` is a basic segment, QI raw values are
    drawn from inside it; otherwise drawn from each QI's full domain."""
    row = _fill_non_qi(header, qi_cols, real_rows, rng, fake_idx)
    for qi_i, (col, h) in enumerate(zip(qi_cols, hiers)):
        if seg_or_none is not None:
            pool = raw_values_under_basic(h, seg_or_none[qi_i])
        else:
            pool = all_raw_values(h)
        row[col] = rng.choice(pool)
    row[MARKER] = "1"
    return row


def _layer2_parent(seg: tuple[str, ...], hiers: Sequence[Hierarchy]) -> tuple[str, ...]:
    """Layer-2 (one step coarser) footprint of a basic segment, per QI. Falls
    back to the basic value itself if a QI has no layer-2 ancestor."""
    out = []
    for qi_i, h in enumerate(hiers):
        node = h.layers[1].get(seg[qi_i])
        parent = node.parent if node is not None else None
        out.append(parent.value if parent is not None and parent.layer >= 1 else seg[qi_i])
    return tuple(out)


def inject(
    rows: Sequence[Row],
    header: Sequence[str],
    qi_cols: Sequence[str],
    hiers: Sequence[Hierarchy],
    strategy: str,
    k: int,
    noise_budget: int,
    max_noise_per_segment: int,
    seed: int,
) -> InjectionResult:
    """Apply a defense strategy. Returns real+fake rows with a MARKER column."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")

    rng = random.Random(seed)
    out_header = list(header) + ([MARKER] if MARKER not in header else [])
    # Tag every real row with marker 0 (copy so we never mutate caller data).
    real_rows: list[Row] = [{**r, MARKER: "0"} for r in rows]
    real_qi = [[r[q] for q in qi_cols] for r in rows]

    fakes: list[Row] = []
    targets: list[tuple[str, ...]] = []
    per_seg: dict[tuple[str, ...], int] = {}

    if strategy == "no_noise" or noise_budget <= 0:
        return InjectionResult(
            rows=real_rows, fake_rows=[], header=out_header,
        )

    if strategy == "random_noise":
        for j in range(noise_budget):
            fakes.append(_make_fake(None, qi_cols, hiers, out_header, rows, rng, j))

    else:  # sparse_basic_noise or targeted_overlap_noise
        counts = segment_counts(real_qi, hiers)
        sparse = sparse_segments(counts, k)

        if strategy == "targeted_overlap_noise":
            # Score each sparse segment by populated neighbours under its
            # layer-2 parent (more neighbours -> more likely to drive overlaps).
            parent_members: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
            for seg in counts:
                p = _layer2_parent(seg, hiers)
                parent_members.setdefault(p, []).append(seg)

            def score(seg: tuple[str, ...]) -> int:
                p = _layer2_parent(seg, hiers)
                return sum(
                    1 for s in parent_members.get(p, [])
                    if s != seg and counts.get(s, 0) > 0
                )

            sparse.sort(key=lambda s: (-score(s), counts[s], s))

        budget_left = noise_budget
        for seg in sparse:
            if budget_left <= 0:
                break
            take = min(max_noise_per_segment, budget_left)
            for _ in range(take):
                idx = len(fakes)
                fakes.append(
                    _make_fake(seg, qi_cols, hiers, out_header, rows, rng, idx)
                )
            per_seg[seg] = take
            targets.append(seg)
            budget_left -= take

    all_rows = real_rows + fakes
    return InjectionResult(
        rows=all_rows, fake_rows=fakes, header=out_header,
        target_segments=targets, per_segment_added=per_seg,
    )
