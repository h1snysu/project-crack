"""Show the generalization lattice and the per-round node selection of our
local-recoding anonymizer (builtin_lr), mirroring ARX's GUI "Explore" view so
the two can be compared node-by-node.

Node notation matches ARX: a transformation is a level vector [l1, l2, ...],
one generalization level per QI. Level 0 = raw leaf, level 1 = the smallest
interval (basic segment) -- since we anonymize with min level = 1, published
nodes have every level >= 1. The all-root vector is the fully-suppressed node.

Our anonymizer visits lattice nodes in ascending information loss and, at each
node, publishes every group of >= k still-unassigned records. A "round" is a
node at which at least one new equivalence class was published -- exactly the
sequence the GUI shows as "the node chosen each round".

Usage (from project root):
    cra/.venv/bin/python -m defense.lattice_trace \
        --input dataset/min1/example-raw-dataset.csv \
        --qi-cols "Age,Systolic Blood Pressure" --k 5 \
        [--hier-dir dataset/min1/hierarchies] [--json out/trace.json] \
        [--full-lattice]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

from .segment_utils import load_hierarchies, ancestors_by_layer
from .lr_anonymizer import _ordered_states, _state_info_loss, SUPPRESSED
from .domain_share import build_domain_shares, level_loss_weighted, arx_node_loss
from hierarchy import Hierarchy


def _qloss_by_layer(hiers):
    out = []
    for h in hiers:
        d = {}
        for layer in range(1, h.height + 1):
            d[layer] = max((n.q_loss for n in h.layers[layer].values()), default=0.0)
        out.append(d)
    return out


def build_lattice(hiers):
    """Every publishable node (level vector) with its info loss, plus the root.
    Returns list of dicts sorted by (info_loss, sum, vector)."""
    qbl = _qloss_by_layer(hiers)
    ranges = [range(1, h.height + 1) for h in hiers]
    nodes = []
    for s in product(*ranges):
        nodes.append({
            "node": list(s),
            "sum_levels": sum(s),
            "info_loss": round(_state_info_loss(s, qbl), 6),
            "is_root": all(l == h.height for l, h in zip(s, hiers)),
        })
    nodes.sort(key=lambda d: (d["info_loss"], d["sum_levels"], tuple(d["node"])))
    return nodes


def _arx_ordered_states(hiers, level_loss):
    """All publishable states ordered by ARX domain-share Loss (matches the
    GUI Explore ranking)."""
    ranges = [range(1, h.height + 1) for h in hiers]
    all_root = tuple(h.height for h in hiers)
    states = [s for s in product(*ranges) if s != all_root]
    states.sort(key=lambda s: (arx_node_loss(s, level_loss), s))
    return states


def trace_anonymize(records_qi, hiers, k, order="uniform", level_loss=None):
    """Run builtin_lr while recording the per-round node selections.
    order: 'uniform' (interval-width ratio; matches GUI record output for 2QI)
           or 'arx' (domain-share Loss; matches the GUI lattice node ranking).
    Returns (published_tuples, rounds, n_suppressed)."""
    n, m = len(records_qi), len(hiers)
    anc = [[ancestors_by_layer(hiers[qi], records_qi[i][qi]) for qi in range(m)]
           for i in range(n)]
    qbl = _qloss_by_layer(hiers)

    state_order = (_arx_ordered_states(hiers, level_loss) if order == "arx"
                   else _ordered_states(hiers))

    published = [None] * n
    remaining = set(range(n))
    rounds = []
    rnum = 0
    for state in state_order:
        if not remaining:
            break
        groups = defaultdict(list)
        for i in remaining:
            groups[tuple(anc[i][qi][state[qi]] for qi in range(m))].append(i)
        published_here = []
        for tup in sorted(groups):
            members = groups[tup]
            if len(members) >= k:
                for i in members:
                    published[i] = tup
                remaining.difference_update(members)
                published_here.append((tup, len(members)))
        if published_here:
            rnum += 1
            rounds.append({
                "round": rnum,
                "node": list(state),
                "info_loss": round(_state_info_loss(state, qbl), 6),
                "eqs_published": len(published_here),
                "records_published": sum(c for _, c in published_here),
                "remaining_after": len(remaining),
                "example_eqs": [
                    {"values": list(t), "count": c}
                    for t, c in published_here[:3]
                ],
            })
    supp = tuple([SUPPRESSED] * m)
    for i in remaining:
        published[i] = supp
    return published, rounds, len(remaining)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--qi-cols", required=True)
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--hier-dir", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--full-lattice", action="store_true",
                   help="Print every lattice node, not just the selected ones.")
    p.add_argument("--order", choices=["uniform", "arx"], default="uniform",
                   help="Node visiting order for the per-round trace. 'uniform' "
                        "(default) matches the GUI record output for 2QI; 'arx' "
                        "uses ARX's domain-share Loss (matches the GUI lattice "
                        "node ranking).")
    args = p.parse_args()

    qi = [c.strip() for c in args.qi_cols.split(",") if c.strip()]
    hier_dir = args.hier_dir or (args.input.parent / "hierarchies")
    hiers = load_hierarchies(qi, hier_dir)

    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rec = [[r[q] for q in qi] for r in rows]

    # ARX domain-share Loss -> lattice ranking that matches the GUI Explore view.
    shares = build_domain_shares(qi, hier_dir)
    level_loss = level_loss_weighted(rec, hiers, shares)

    lattice = build_lattice(hiers)
    for d in lattice:
        d["arx_loss"] = round(arx_node_loss(tuple(d["node"]), level_loss), 6)
    lattice.sort(key=lambda d: (d["arx_loss"], tuple(d["node"])))

    published, rounds, n_supp = trace_anonymize(
        rec, hiers, args.k, order=args.order, level_loss=level_loss)

    from collections import Counter
    eqs = Counter(published)

    print(f"=== builtin_lr local-recoding trace ===")
    print(f"QIs (levels): {qi}  (max level per QI: {[h.height for h in hiers]})")
    print(f"records={len(rec)}  k={args.k}  lattice nodes={len(lattice)}")
    print(f"published EQs={len([t for t in eqs if t[0]!=SUPPRESSED])}  suppressed records={n_supp}")

    print(f"\n--- Lattice ranked by ARX domain-share Loss (matches GUI Explore) ---")
    print(f"  {'node':<16} {'ARX_loss':>9} {'uniform':>8}")
    show = lattice if args.full_lattice else lattice[:20]
    for d in show:
        tag = " (root/suppress)" if d["is_root"] else ""
        print(f"  {str(d['node']):<16} {d['arx_loss']:>9.4f} {d['info_loss']:>8.4f}{tag}")
    if not args.full_lattice and len(lattice) > 20:
        print(f"  ... ({len(lattice)-20} more; use --full-lattice)")

    print(f"\n--- Per-round node selection (rounds that published >=1 EQ) ---")
    print(f"  {'rnd':>3} {'node':<16} {'loss':>8} {'newEQs':>6} {'recs':>6} {'remain':>7}")
    for r in rounds:
        print(f"  {r['round']:>3} {str(r['node']):<16} {r['info_loss']:>8.4f} "
              f"{r['eqs_published']:>6} {r['records_published']:>6} {r['remaining_after']:>7}")
    print(f"  (total rounds/selected nodes: {len(rounds)})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "qi": qi, "k": args.k, "levels_max": [h.height for h in hiers],
            "lattice": lattice, "rounds": rounds,
            "n_suppressed": n_supp,
            "n_published_eqs": len([t for t in eqs if t[0] != SUPPRESSED]),
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
