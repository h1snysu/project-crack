"""Segment representation.

A `Segment` is an m-tuple of `HierarchyNode`s -- one node per QI. It defines a
region of the basic-segment grid as the Cartesian product of the basic-segment
representatives (the layer-1 nodes) under each of its nodes.

Vocabulary (mapping to the paper, 0-indexed layers):
    * basic segment : every dim at layer 1 (smallest interval, NOT the raw
      data points at layer 0). The LP variables z_j correspond to these.
    * compound segment : at least one dim above layer 1.
    * half-segment of S : lower exactly one QI's node by one layer (to one of
      that node's children); never descends to layer 0.
    * descendant segments of S : all segments T with T.nodes[i] in the
      layer >= 1 subtree of S.nodes[i] for every i (includes S itself).

Layer-0 snap:
    `Segment(...)` automatically maps any layer-0 nodes up to their layer-1
    parents. ARX-LR exports records as raw layer-0 values when it chose not
    to generalise; in the basic-segment grid we represent them by their
    containing layer-1 interval.

Containment uses the leaf-set (basic representative) inclusion test:
    S.contains(T) iff  T.nodes[i].leaves <= S.nodes[i].leaves  for every i.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Tuple

from models import HierarchyNode


# A basic segment is identified by its tuple of leaf values, one per QI.
Basic = Tuple[str, ...] # type alias


def _descendants(node: HierarchyNode) -> Iterator[HierarchyNode]:
    """Yield `node` and every node in its subtree (pre-order DFS), stopping
    at layer 1. Layer-0 children (raw data points) are below the basic-segment
    grid and are excluded."""
    if node.layer < 1:
        return
    yield node
    for child in node.children:
        if child.layer >= 1:
            yield from _descendants(child)


@dataclass(frozen=True)
class Segment:
    """Immutable m-tuple of hierarchy nodes.

    Any layer-0 node passed in is snapped to its layer-1 parent at construction
    time (see module docstring).
    """

    nodes: tuple[HierarchyNode, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            return
        if any(n.layer == 0 for n in self.nodes):
            snapped: list[HierarchyNode] = []
            for n in self.nodes:
                if n.layer == 0:
                    if n.parent is None or n.parent.layer != 1:
                        raise ValueError(
                            f"Cannot snap layer-0 node {n!r}: parent is "
                            f"{n.parent!r}"
                        )
                    snapped.append(n.parent)
                else:
                    snapped.append(n)
            # frozen=True -> bypass via object.__setattr__
            object.__setattr__(self, "nodes", tuple(snapped))

    @property
    def m(self) -> int:
        # number of QIs
        return len(self.nodes)

    @property
    def qi_layers(self) -> tuple[int, ...]:
        return tuple(n.layer for n in self.nodes)

    @property
    def qi_values(self) -> tuple[str, ...]:
        return tuple(n.value for n in self.nodes)

    @property
    def is_basic(self) -> bool:
        return all(n.layer == 1 for n in self.nodes)

    @property
    def is_grid(self) -> bool:
        """True iff every dim is the root of its hierarchy."""
        return all(n.parent is None for n in self.nodes)

    def size(self) -> int:
        """Number of basic segments inside this segment."""
        size = 1
        for n in self.nodes:
            size *= len(n.leaves)
        return size

    def contains(self, other: "Segment") -> bool:
        if self.m != other.m:
            raise ValueError(
                f"Segment arity mismatch: {self.m} vs {other.m}"
            )
        for a, b in zip(self.nodes, other.nodes):
            if a.qi_index != b.qi_index:
                raise ValueError("Segment QI-index mismatch")
            if not b.leaves.issubset(a.leaves):
                return False
        return True

    def iter_basic_leaf_tuples(self) -> Iterator[Basic]:
        """Cartesian product of the leaves under each dim, in stable order."""
        leaf_lists = [sorted(n.leaves) for n in self.nodes]
        yield from product(*leaf_lists)

    def iter_half_segments(self) -> Iterator["Segment"]:
        """Half-segments: lower exactly one QI by one layer.

        Never descends to layer 0 -- the basic-segment grid stops at layer 1.
        Yields nothing for dims that are already at layer 1.
        """
        nodes = self.nodes
        for i, node in enumerate(nodes):
            for child in node.children:
                if child.layer < 1:
                    continue
                yield Segment(nodes[:i] + (child,) + nodes[i + 1:])

    def iter_descendant_segments(
        self, *, include_self: bool = True
    ) -> Iterator["Segment"]:
        """Every segment whose nodes lie in the per-dim subtrees of self."""
        per_dim = [list(_descendants(n)) for n in self.nodes]
        for combo in product(*per_dim):
            seg = Segment(combo)
            if not include_self and seg == self:
                continue
            yield seg

    def __repr__(self) -> str:
        layers = "/".join(str(n.layer) for n in self.nodes)
        values = "/".join(repr(n.value) for n in self.nodes)
        return f"Seg(L={layers}, V={values})"
