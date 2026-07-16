"""Generalization hierarchy parsing and queries.

Each hierarchy CSV row encodes one raw value plus its ancestry across columns:
    col 0 (raw point), col 1 (smallest interval), col 2, ..., col H (root)

Layer convention matches the paper:
    layer == col_idx     -- 0-indexed
    layer == 0           -- raw data point (ARX-LR's ungeneralised cells)
    layer == 1           -- smallest interval == BASIC SEGMENT representative
    layer == h^i         -- root (root_layer == num_columns - 1)

The `Basic` solver grid is built from layer-1 nodes (not layer 0). ARX exports
column 0 with one row per raw value, which can fan out 11-to-1 below an
interval like `[110, 120[`; pushing the basic-segment layer up to column 1
keeps the grid tractable and matches Algorithm 3.

Q_loss for a node follows Eq. 1 of the paper (numeric interval form):
    q_loss(node) = length(node.interval) / length(root.interval)
Layer-0 nodes are single points (length 0) so q_loss == 0 there automatically.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from models import HierarchyNode


# Matches "[lo, hi[", "[lo, hi]", "(lo, hi)", etc. Captures lo and hi as numbers.
_INTERVAL_RE = re.compile(
    r"""^\s*
        [\[\(]\s*
        ([+-]?\d+(?:\.\d+)?)
        \s*,\s*
        ([+-]?\d+(?:\.\d+)?)
        \s*[\]\)\[]\s*$
    """,
    re.VERBOSE,
)


def parse_interval_length(value: str) -> float:
    """Length of an interval string, or 0 for a single numeric value."""
    s = value.strip().strip('"')
    m = _INTERVAL_RE.match(s)
    if m:
        return float(m.group(2)) - float(m.group(1))
    try:
        float(s)
        return 0.0
    except ValueError as e:
        raise ValueError(
            f"Cannot parse {value!r} as an interval or numeric value"
        ) from e


class Hierarchy:
    """A QI's generalization tree, indexed by (layer, value)."""

    def __init__(self, qi_name: str, qi_index: int) -> None:
        self.qi_name = qi_name
        self.qi_index = qi_index
        # h^i (= root layer index = number of CSV columns - 1)
        self.height: int = 0
        self.root: HierarchyNode | None = None
        # layers[l][value] -> node at layer l, for l in 0..height (inclusive)
        self.layers: list[dict[str, HierarchyNode]] = []
        # Sorted list of basic-segment representatives (the layer-1 values).
        self.leaves: list[str] = []
        self._value_index: dict[str, HierarchyNode] = {}

    @classmethod
    def from_csv(
        cls, path: Path | str, qi_name: str, qi_index: int
    ) -> "Hierarchy":
        path = Path(path)
        h = cls(qi_name=qi_name, qi_index=qi_index)

        rows: list[list[str]] = []
        with path.open(newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                rows.append([c.strip() for c in row])
        if not rows:
            raise ValueError(f"Hierarchy file {path} is empty")

        num_columns = len(rows[0])
        if any(len(r) != num_columns for r in rows):
            raise ValueError(f"Hierarchy file {path}: inconsistent row widths")
        if num_columns < 2:
            raise ValueError(
                f"Hierarchy file {path}: needs >= 2 columns "
                f"(layer 0 = raw, layer 1 = basic segment); got {num_columns}"
            )
        height = num_columns - 1  # root layer index = h^i
        h.height = height
        h.layers = [dict() for _ in range(num_columns)]

        # Build / link nodes row by row.
        for row in rows:
            prev: HierarchyNode | None = None
            for col_idx, val in enumerate(row):
                layer = col_idx  # 0-indexed: col 0 -> layer 0 (raw)
                table = h.layers[layer]
                node = table.get(val)
                if node is None:
                    node = HierarchyNode(
                        value=val, layer=layer, qi_index=qi_index
                    )
                    table[val] = node
                if prev is not None:
                    if prev.parent is None:
                        prev.parent = node
                        node.children.append(prev)
                    elif prev.parent is not node:
                        raise ValueError(
                            f"Hierarchy {path}: node {prev} has conflicting "
                            f"parents {prev.parent} vs {node}"
                        )
                prev = node

        # Verify a single root at the top layer (h.height).
        top = h.layers[height]
        if len(top) != 1:
            raise ValueError(
                f"Hierarchy {path}: expected one root at layer {height}, "
                f"got {len(top)}: {list(top)}"
            )
        h.root = next(iter(top.values()))
        # Hierarchy.leaves := basic-segment representatives (layer 1).
        h.leaves = sorted(h.layers[1].keys(), key=_leaf_sort_key)

        # Propagate `leaves` (set of basic-segment values reachable from node).
        # Layer 1 seeds: each is its own basic representative.
        for node in h.layers[1].values():
            node.leaves = frozenset([node.value])
        # Layer 0 (raw): inherits parent's basic representative.
        for node in h.layers[0].values():
            if node.parent is None:
                raise ValueError(
                    f"Hierarchy {path}: layer-0 node {node.value!r} has no parent"
                )
            node.leaves = node.parent.leaves
        # Higher layers: union of children's basic representatives.
        for layer_idx in range(2, height + 1):
            for node in h.layers[layer_idx].values():
                s: set[str] = set()
                for c in node.children:
                    s.update(c.leaves)
                node.leaves = frozenset(s)

        # Sanity: root covers every layer-1 representative.
        if h.root.leaves != frozenset(h.layers[1].keys()):
            missing = frozenset(h.layers[1].keys()) - h.root.leaves
            raise ValueError(
                f"Hierarchy {path}: root does not cover all basic segments; "
                f"missing {sorted(missing)[:5]}..."
            )

        # Build a global value index; enforce no value appearing at two layers.
        for layer_idx in range(num_columns):
            for val, node in h.layers[layer_idx].items():
                if val in h._value_index:
                    other = h._value_index[val]
                    raise ValueError(
                        f"Hierarchy {path}: value {val!r} appears at both "
                        f"layer {other.layer} and layer {node.layer}; "
                        f"resolution would be ambiguous"
                    )
                h._value_index[val] = node

        # Compute q_loss = (length of node interval) / (length of root interval).
        # Layer-0 nodes are single points -> length 0 -> q_loss == 0 automatically.
        root_len = parse_interval_length(h.root.value)
        if root_len <= 0:
            for layer_idx in range(num_columns):
                for node in h.layers[layer_idx].values():
                    node.q_loss = 0.0
        else:
            for layer_idx in range(num_columns):
                for node in h.layers[layer_idx].values():
                    node.q_loss = parse_interval_length(node.value) / root_len

        return h

    def resolve(self, cell_value: str) -> HierarchyNode:
        """Map a D_gen cell to its hierarchy node.

        Tries the raw string first; falls back to numeric canonicalisation
        (e.g. dataset "60" vs hierarchy "60.0"). ARX's suppression marker
        '*' is treated as the root (fully-generalised value).
        """
        s = cell_value.strip().strip('"')
        if s == "*":
            assert self.root is not None
            return self.root
        node = self._value_index.get(s)
        if node is not None:
            return node

        # Numeric fallback.
        try:
            f = float(s)
        except ValueError:
            raise KeyError(
                f"Cell value {cell_value!r} not found in hierarchy for QI "
                f"{self.qi_name!r}"
            ) from None

        candidates: list[str] = [str(f)]
        if f.is_integer():
            candidates += [str(int(f)), f"{int(f)}.0"]
        for c in candidates:
            node = self._value_index.get(c)
            if node is not None:
                return node

        raise KeyError(
            f"Cell value {cell_value!r} not found in hierarchy for QI "
            f"{self.qi_name!r} (tried {candidates})"
        )

    def node(self, value: str, layer: int) -> HierarchyNode:
        return self.layers[layer][value]

    def all_nodes(self) -> Iterable[HierarchyNode]:
        for layer in self.layers:
            yield from layer.values()

    def __repr__(self) -> str:
        return (
            f"Hierarchy(qi={self.qi_index}, name={self.qi_name!r}, "
            f"height={self.height}, leaves={len(self.leaves)})"
        )


def _leaf_sort_key(v: str):
    """Sort leaves numerically when possible, lexicographically otherwise."""
    try:
        return (0, float(v))
    except ValueError:
        return (1, v)
