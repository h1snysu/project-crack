"""Shared data models for the CRA implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class HierarchyNode:
    """One node in a QI's generalization tree.

    Layer convention (matches the paper):
        layer == 1   -> leaf
        layer == h^i -> root
    """

    value: str
    layer: int
    qi_index: int
    parent: Optional["HierarchyNode"] = None
    children: list["HierarchyNode"] = field(default_factory=list)
    leaves: frozenset[str] = field(default_factory=frozenset)
    q_loss: float = 0.0

    def __hash__(self) -> int:
        return hash((self.qi_index, self.layer, self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HierarchyNode):
            return NotImplemented
        return (
            self.qi_index == other.qi_index
            and self.layer == other.layer
            and self.value == other.value
        )

    def __repr__(self) -> str:
        return (
            f"Node(qi={self.qi_index}, L{self.layer}, "
            f"value={self.value!r}, qloss={self.q_loss:.4f})"
        )


@dataclass
class EquivalenceClass:
    """One equivalence class extracted from D_gen.

    `qi_values`  - the generalised tuple (one cell per QI) shared by every
                   record in this EQ; serves as a unique identity key.
    `qi_nodes`   - the resolved HierarchyNode per QI.
    `record_count` = |EQ|
    `info_loss`  - paper Eq. 2 : (prod_i (Q_loss^i + 1))^(1/m) - 1
    `c1`         - sum of layer indices (lower = less generalised)
    `c2`         - mean of lyr^i / h^i across QIs (normalised version of c1)
    """

    qi_values: Tuple[str, ...]
    qi_nodes: Tuple[HierarchyNode, ...]
    record_count: int
    info_loss: float
    c1: int
    c2: float

    @property
    def m(self) -> int:
        return len(self.qi_nodes)

    @property
    def qi_layers(self) -> Tuple[int, ...]:
        return tuple(n.layer for n in self.qi_nodes)

    def __repr__(self) -> str:
        layers = "/".join(str(l) for l in self.qi_layers)
        return (
            f"EQ(|EQ|={self.record_count}, layers={layers}, "
            f"loss={self.info_loss:.4f}, c1={self.c1}, c2={self.c2:.4f}, "
            f"values={self.qi_values!r})"
        )
