"""The m-dimensional basic-segment universe.

`Grid` is a thin orchestrator around the m hierarchies. It exposes:

* `grid_segment`           : the full-domain segment (every QI at root).
* `total_basics`           : |G| = product of leaf counts per QI.
* `basics_in(segment)`     : iterator of basic-segment leaf-tuples inside `segment`.
* `basics_in_set(segment)` : same as above, materialised into a set.

Basic segments are identified by their leaf-tuple `Basic = tuple[str, ...]`,
which is hashable and stable across LP runs -- so we use it directly as the
CP-SAT variable key (no separate integer index needed).
"""
from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Iterator

from hierarchy import Hierarchy
from segment import Basic, Segment


class Grid:
    def __init__(self, hierarchies: list[Hierarchy]) -> None:
        if not hierarchies:
            raise ValueError("Grid requires at least one hierarchy")
        for i, h in enumerate(hierarchies):
            if h.qi_index != i:
                raise ValueError(
                    f"Hierarchy at position {i} has qi_index={h.qi_index}; "
                    f"expected {i}"
                )
        self.hierarchies = hierarchies
        self.m = len(hierarchies)
        self.grid_segment = Segment(tuple(h.root for h in hierarchies))

    @property
    def total_basics(self) -> int:
        return reduce(mul, (len(h.leaves) for h in self.hierarchies), 1)

    def basics_in(self, segment: Segment) -> Iterator[Basic]:
        if segment.m != self.m:
            raise ValueError(
                f"Segment arity {segment.m} != grid arity {self.m}"
            )
        yield from segment.iter_basic_leaf_tuples()

    def basics_in_set(self, segment: Segment) -> set[Basic]:
        return set(self.basics_in(segment))
