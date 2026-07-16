"""Phase 3 step-1 smoke test: Segment + Grid against the 2-QI dataset.

Validates the 0-indexed layer convention (col 0 = layer 0 = raw, col 1 =
layer 1 = basic segment, col h = root) and the layer-0 auto-snap.

Run from cra/:
    python smoke_phase3.py
"""
from __future__ import annotations

from pathlib import Path

from dataset import Dataset
from equivalence import extract_equivalence_classes, sort_equivalence_classes
from grid import Grid
from hierarchy import Hierarchy
from segment import Segment, _descendants

HIER_DIR = Path("../dataset/min1/hierarchies")
DATASET = Path("../dataset/min1/anonymized_2qi_k5.csv")
K = 5


def header(s: str) -> None:
    print(f"\n=== {s} ===")


def check(name: str, got, expected) -> None:
    ok = "PASS" if got == expected else "FAIL"
    print(f"  [{ok}] {name}: got={got!r} expected={expected!r}")
    if got != expected:
        raise SystemExit(1)


def main() -> None:
    ds = Dataset.from_csv(DATASET, k=K)
    age = Hierarchy.from_csv(HIER_DIR / "hierarchy_age.csv", "Age", 0)
    sbp = Hierarchy.from_csv(
        HIER_DIR / "hierarchy_systolic-blood-pressure.csv",
        "Systolic Blood Pressure", 1,
    )
    hiers = [age, sbp]
    grid = Grid(hiers)

    # ---- 0. Layer convention --------------------------------------------
    header("Layer convention (0-indexed)")
    check("age.height (root layer)", age.height, 5)
    check("sbp.height", sbp.height, 5)
    check("age root.layer", age.root.layer, age.height)
    check("# layer-0 nodes (raw)", len(age.layers[0]), 68)
    check("# layer-1 nodes (basics)", len(age.layers[1]), 15)
    check("len(age.leaves) is basic-segment count", len(age.leaves), 15)

    # ---- 1. Grid -------------------------------------------------------
    header("Grid")
    check("m", grid.m, 2)
    check(
        "total_basics",
        grid.total_basics,
        len(age.layers[1]) * len(sbp.layers[1]),
    )
    print(f"  grid_segment = {grid.grid_segment}")
    check("grid_segment.is_grid", grid.grid_segment.is_grid, True)
    check(
        "grid_segment.size()", grid.grid_segment.size(), grid.total_basics
    )

    # ---- 2. Pick a basic (layer 1/1) EQ; verify snap is a no-op ---------
    extraction = extract_equivalence_classes(ds, hiers)
    sorted_eqs = sort_equivalence_classes(extraction.eqs)
    eq = next(e for e in sorted_eqs if e.qi_layers == (1, 1))

    header(f"Basic EQ {eq.qi_values} (pre-snap layers {eq.qi_layers})")
    seg = Segment(eq.qi_nodes)
    print(f"  segment (post-snap) = {seg}")
    check("seg.qi_layers (post-snap)", seg.qi_layers, (1, 1))
    check("is_basic", seg.is_basic, True)
    check("is_grid", seg.is_grid, False)

    basics = list(seg.iter_basic_leaf_tuples())
    check("|basics| (1x1)", len(basics), 1)
    check("basic tuple == seg values", basics[0], eq.qi_values)

    # Halves: both dims at layer 1 -> no halves emitted.
    halves = list(seg.iter_half_segments())
    check("|halves| (already basic)", len(halves), 0)

    descs = list(seg.iter_descendant_segments(include_self=True))
    check("|descendants(self)|", len(descs), 1)

    # ---- 3. Pick a *real* compound EQ (layers > 1 somewhere) ------------
    compound = next(
        e for e in sorted_eqs if max(e.qi_layers) >= 2
    )
    header(
        f"Compound EQ {compound.qi_values} (pre-snap layers {compound.qi_layers})"
    )
    cseg = Segment(compound.qi_nodes)
    print(f"  segment = {cseg}")

    # Per-dim subtree sizes (down to layer 1, not layer 0).
    per_dim_descs = [list(_descendants(n)) for n in cseg.nodes]
    expected_subseg = per_dim_descs[0].__len__() * per_dim_descs[1].__len__()
    expected_basics = len(cseg.nodes[0].leaves) * len(cseg.nodes[1].leaves)
    expected_halves = sum(
        len([c for c in n.children if c.layer >= 1]) for n in cseg.nodes
    )

    check(
        "|basics| matches Cartesian-product of leaves",
        cseg.size(), expected_basics,
    )
    check(
        "|halves| equals layer-1+ children of each dim",
        len(list(cseg.iter_half_segments())), expected_halves,
    )
    check(
        "|descendants(self)| matches Cartesian subtree product",
        sum(1 for _ in cseg.iter_descendant_segments()), expected_subseg,
    )

    # A half-segment must reduce exactly one dim's layer by 1 and never below 1.
    for h in cseg.iter_half_segments():
        diffs = [
            (a.layer, b.layer)
            for a, b in zip(cseg.nodes, h.nodes)
            if a is not b
        ]
        assert len(diffs) == 1, f"half changed != 1 dim: {h}"
        old, new = diffs[0]
        assert new == old - 1 and new >= 1, f"bad half layer {old}->{new}"

    # ---- 4. Containment in 0-indexed world ------------------------------
    header("Containment + layer-0 snap")
    # Raw layer-0 nodes -> Segment auto-snap to layer 1.
    raw_age = age.resolve("71")        # raw point '71'  -> layer 0
    raw_sbp = sbp.resolve("123")       # raw point '123' (stored as '123.0')
    check("raw_age.layer", raw_age.layer, 0)
    check("raw_sbp.layer", raw_sbp.layer, 0)
    snapped = Segment((raw_age, raw_sbp))
    check("auto-snap age layer", snapped.nodes[0].layer, 1)
    check("auto-snap sbp layer", snapped.nodes[1].layer, 1)
    check("auto-snap age value", snapped.nodes[0].value, "[70, 75[")
    check("auto-snap sbp value", snapped.nodes[1].value, "[120, 130[")
    check("snapped.is_basic", snapped.is_basic, True)
    check(
        "grid contains snapped basic", grid.grid_segment.contains(snapped), True
    )

    # ---- 5. Grid summary ------------------------------------------------
    header("Grid sub-segment counts (no layer 0 reachable)")
    age_total = sum(
        len(age.layers[l]) for l in range(1, age.height + 1)
    )  # layers 1..5
    sbp_total = sum(
        len(sbp.layers[l]) for l in range(1, sbp.height + 1)
    )
    print(f"  per-QI layer 1..h node counts: age={age_total} sbp={sbp_total}")
    check(
        "|descendants(grid)|",
        sum(1 for _ in grid.grid_segment.iter_descendant_segments()),
        age_total * sbp_total,
    )
    grid_halves = list(grid.grid_segment.iter_half_segments())
    expected_grid_halves = sum(
        len([c for c in h.root.children if c.layer >= 1]) for h in hiers
    )
    check("|halves(grid)|", len(grid_halves), expected_grid_halves)
    for h in grid_halves:
        for n in h.nodes:
            assert n.layer >= 1, f"grid half went below layer 1: {h}"

    print("\nAll Phase 3 step-1 checks passed.")


if __name__ == "__main__":
    main()
