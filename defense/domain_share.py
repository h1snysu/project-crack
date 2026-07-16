"""ARX's exact domain-share Loss metric (the default "Loss"/Granularity model),
reverse-engineered from libarx-3.9.2 source (metric/v2/DomainShareMaterialized
+ MetricMDNMLoss).

DomainShareMaterialized:
    share(value, level) = (# hierarchy rows whose column[level] == value)
                          / (# hierarchy rows)
i.e. the fraction of the attribute's raw domain that a generalized value covers.
This is purely a function of the hierarchy CSV -- it is what makes generalizing
a coarse-domain attribute (e.g. Heart rate, 45 raw values) "cost" more per level
than a fine-domain one (e.g. Age, 68 raw values), which uniform interval-width
ratios miss.

Per-cell normalized Loss (Iyengar):
    nloss(value, level) = (share - 1/size) / (1 - 1/size)
so a leaf (share=1/size) -> 0 and the root (share=1) -> 1.

This module is used to RANK lattice nodes exactly as ARX's Explore view does,
so our tool's node ordering can be checked against the GUI. It reproduces the
GUI's node ranking and gets [1,1,1] ~= 0.123 (GUI shows 0.1216; the small gap
is ARX's record/gsFactor weighting, which does not change the ordering).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from hierarchy import Hierarchy
from .segment_utils import QI_TO_HIER


def _hier_path(qi_name: str, hier_dir: Path) -> Path:
    lookup = {k.lower(): v for k, v in QI_TO_HIER.items()}
    fname = lookup.get(qi_name.lower()) or f"hierarchy_{qi_name.lower().replace(' ', '-')}.csv"
    return Path(hier_dir) / fname


def build_domain_shares(qi_names: Sequence[str], hier_dir: Path) -> list[dict]:
    """For each QI return {'size': int, 'share': {level: {value: share}}}.

    share = count(rows with value at that level) / size, matching ARX's
    DomainShareMaterialized exactly.
    """
    out = []
    for qi in qi_names:
        rows = [r for r in csv.reader(_hier_path(qi, hier_dir).open()) if r]
        size = len(rows)
        ncol = len(rows[0])
        counts = [defaultdict(int) for _ in range(ncol)]
        for r in rows:
            for lvl in range(ncol):
                counts[lvl][r[lvl].strip().strip('"')] += 1
        share = {lvl: {v: c / size for v, c in counts[lvl].items()}
                 for lvl in range(ncol)}
        out.append({"size": size, "share": share})
    return out


def nloss(shares: dict, value: str, level: int) -> float:
    """Normalized per-cell Loss for a generalized value at a level."""
    size = shares["size"]
    s = shares["share"].get(level, {}).get(value, 0.0)
    return (s - 1.0 / size) / (1.0 - 1.0 / size) if size > 1 else 0.0


def level_loss_weighted(
    records_qi, hiers: Sequence[Hierarchy], shares: list[dict]
) -> list[dict[int, float]]:
    """Record-weighted mean normalized Loss per (QI, level) -- the separable
    scalar used to order lattice nodes the ARX way."""
    from .segment_utils import ancestors_by_layer
    n, m = len(records_qi), len(hiers)
    anc = [[ancestors_by_layer(hiers[qi], records_qi[i][qi]) for qi in range(m)]
           for i in range(n)]
    out: list[dict[int, float]] = [dict() for _ in range(m)]
    for qi in range(m):
        for lvl in range(1, hiers[qi].height + 1):
            out[qi][lvl] = sum(nloss(shares[qi], anc[i][qi][lvl], lvl)
                               for i in range(n)) / n
    return out


def arx_node_loss(state, level_loss: list[dict[int, float]]) -> float:
    """ARX Loss score of a lattice node = arithmetic mean over QIs of the
    per-(QI, level) record-weighted Loss. Matches the GUI Explore ranking."""
    m = len(state)
    return sum(level_loss[qi][state[qi]] for qi in range(m)) / m
