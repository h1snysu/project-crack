"""Helpers that map raw records onto the basic-segment grid of the CRA
hierarchies, and that sample *valid* raw values inside a chosen basic segment.

"Basic segment" == layer-1 node (the smallest interval), matching the paper and
the existing `cra/segment.py`. A record's basic segment is the tuple of layer-1
interval strings, one per QI.

All sampling draws from the *existing* layer-0 raw values registered in the
hierarchy (e.g. systolic skips 65/66/67), never arbitrary integers, so every
fake value is guaranteed to resolve cleanly through `Hierarchy.resolve`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from typing import Iterable, Sequence

# Make the sibling `cra/` package importable (it uses top-level imports).
_CRA_DIR = Path(__file__).resolve().parent.parent / "cra"
if str(_CRA_DIR) not in sys.path:
    sys.path.insert(0, str(_CRA_DIR))

from hierarchy import Hierarchy          # noqa: E402
from models import HierarchyNode         # noqa: E402
from segment import Segment              # noqa: E402

# Canonical QI name -> hierarchy filename. Mirrors eval_cra.QI_TO_HIER but is
# kept here so the defense package can register synthetic QIs independently.
QI_TO_HIER: dict[str, str] = {
    "Age": "hierarchy_age.csv",
    "Systolic Blood Pressure": "hierarchy_systolic-blood-pressure.csv",
    "Heart rate": "hierarchy_heart-rate.csv",
    "Body Weight": "hierarchy_body-weight.csv",
}


def load_hierarchies(qi_cols: Sequence[str], hier_dir: Path) -> list[Hierarchy]:
    """Resolve one Hierarchy per QI column, in order.

    Lookup is case-insensitive on the registered QI names; a QI that is not in
    `QI_TO_HIER` falls back to `hierarchy_<slug>.csv` where slug lower-cases the
    name and replaces spaces with '-'. This lets synthetic smoke datasets ship
    their own hierarchies without editing the registry.
    """
    hier_dir = Path(hier_dir)
    lookup = {k.lower(): v for k, v in QI_TO_HIER.items()}
    hiers: list[Hierarchy] = []
    for i, qi in enumerate(qi_cols):
        fname = lookup.get(qi.lower())
        if fname is None:
            slug = qi.lower().replace(" ", "-")
            fname = f"hierarchy_{slug}.csv"
        path = hier_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"No hierarchy file for QI {qi!r} (looked for {path}). "
                f"Register it in defense.segment_utils.QI_TO_HIER or name the "
                f"file hierarchy_<slug>.csv."
            )
        hiers.append(Hierarchy.from_csv(path, qi, i))
    return hiers


def ancestors_by_layer(hier: Hierarchy, raw_value: str) -> dict[int, str]:
    """Map layer index (1..height) -> ancestor interval value for a raw cell.

    Layer 1 is the basic segment; the root layer is `hier.height`.
    """
    node: HierarchyNode = hier.resolve(raw_value)
    out: dict[int, str] = {}
    # Walk up to the root, recording every layer >= 1.
    while node is not None:
        if node.layer >= 1:
            out[node.layer] = node.value
        node = node.parent
    return out


def basic_segment_of(
    raw_qi: Sequence[str], hiers: Sequence[Hierarchy]
) -> tuple[str, ...]:
    """Return the layer-1 basic-segment tuple for one record's QI values.

    `Segment` auto-snaps the resolved (possibly layer-0) nodes up to layer 1.
    """
    nodes = tuple(h.resolve(v) for v, h in zip(raw_qi, hiers))
    return Segment(nodes).qi_values


def segment_counts(
    records_qi: Iterable[Sequence[str]], hiers: Sequence[Hierarchy]
) -> Counter:
    """Count how many records fall in each basic segment."""
    c: Counter = Counter()
    for r in records_qi:
        c[basic_segment_of(r, hiers)] += 1
    return c


def sparse_segments(counts: Counter, k: int) -> list[tuple[str, ...]]:
    """Basic segments whose record count is below k (0 < count < k).

    Sorted by ascending count then segment value for determinism. Empty
    segments (count 0) are excluded -- they hold no real record to hide.
    """
    items = [(seg, n) for seg, n in counts.items() if 0 < n < k]
    items.sort(key=lambda x: (x[1], x[0]))
    return [seg for seg, _ in items]


def raw_values_under_basic(hier: Hierarchy, basic_value: str) -> list[str]:
    """Valid layer-0 raw values that live inside a given basic segment.

    These are exactly the children of the layer-1 node, so a fake value drawn
    from this list is guaranteed to resolve back into `basic_value`.
    """
    node = hier.layers[1].get(basic_value)
    if node is None:
        raise KeyError(
            f"{basic_value!r} is not a basic segment of QI {hier.qi_name!r}"
        )
    return [c.value for c in node.children if c.layer == 0]


def all_raw_values(hier: Hierarchy) -> list[str]:
    """Every valid layer-0 raw value of a QI (its full published domain)."""
    return list(hier.layers[0].keys())
