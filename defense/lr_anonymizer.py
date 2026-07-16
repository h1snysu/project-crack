"""builtin_lr: a small, deterministic local-recoding k-anonymizer.

This is an *approximation* of ARX-LR, written so the whole defense pipeline can
run end-to-end (and a 100+ run sweep can be automated) without driving the ARX
Java GUI. For final scientific claims, plug a real ARX export in via
`run_defense_experiment.py --skip-arx --anonymized <file>`; the builtin path is
for relative comparisons (no_noise vs defense, same anonymizer) and smoke tests.

Faithful implementation of the paper's ARX-LR local recoding (Algorithm 2),
min layer = 1:
  * Build the generalization lattice (one node per per-QI layer tuple).
  * Each round, on the CURRENT remaining records, find the node(s) that form at
    least one k-group; pick the one with the smallest ordering key
    (loss_g, c1, c2, c3) where loss_g is Eq. 2 (geometric-mean information loss),
    c1 = sum of layers, c2 = normalized layer sum, and c3 is the data-dependent
    distinct-interval retention criterion (Eq. 3), re-evaluated on the remaining
    records each round.
  * Apply the chosen node: publish all its k-groups as EQs, remove those records.
  * Repeat until no node forms a k-group; suppress the remaining records as
    outliers (every QI -> '*', which the CRA treats as the root/outlier state,
    i.e. the paper's "generalized to the root interval" == Algorithm-4 input).

NOTE on fidelity: the paper's loss uses interval-width ratios (Eq. 1); the real
ARX GUI uses a domain-share Loss metric, so this reproduces the paper exactly and
the GUI exactly for 2 QIs, with small differences at 3/4 QIs (see docs/memory).

Output preserves input row order. QI cells become interval strings or '*'.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Sequence

from hierarchy import Hierarchy

SUPPRESSED = "*"


def _state_info_loss(layers: tuple[int, ...], qloss_by_layer: list[dict[int, float]]) -> float:
    """Paper Eq. 2 info loss for a generalization state, using a representative
    q_loss per (QI, layer)."""
    m = len(layers)
    prod = 1.0
    for qi, l in enumerate(layers):
        prod *= (qloss_by_layer[qi][l] + 1.0)
    return prod ** (1.0 / m) - 1.0


def _ordered_states(hiers: Sequence[Hierarchy]) -> list[tuple[int, ...]]:
    """All publishable states (layer per QI, 1..height), excluding the all-root
    state, ordered by (info_loss, sum-of-layers, tuple) -- ascending."""
    # Representative q_loss per (QI, layer): take the max over nodes at that
    # layer (robust if a layer mixes interval widths; uniform hierarchies make
    # this exact).
    qloss_by_layer: list[dict[int, float]] = []
    for h in hiers:
        d: dict[int, float] = {}
        for layer in range(1, h.height + 1):
            d[layer] = max((n.q_loss for n in h.layers[layer].values()), default=0.0)
        qloss_by_layer.append(d)

    ranges = [range(1, h.height + 1) for h in hiers]
    all_root = tuple(h.height for h in hiers)
    states = [s for s in product(*ranges) if s != all_root]
    states.sort(key=lambda s: (_state_info_loss(s, qloss_by_layer), sum(s), s))
    return states


def _base_key(state: tuple[int, ...], qloss_by_layer, hiers) -> tuple:
    """Data-INDEPENDENT part of the paper's ordering: (loss_g, c1, c2).
      loss_g = Eq. 2 geometric-mean information loss
      c1     = sum of layers
      c2     = (1/m) * sum(layer_i / h_i)   (normalized layer sum)
    """
    m = len(state)
    loss = _state_info_loss(state, qloss_by_layer)
    c1 = sum(state)
    c2 = sum(l / h.height for l, h in zip(state, hiers)) / m
    return (loss, c1, c2)


def anonymize_qi(
    records_qi: Sequence[Sequence[str]],
    hiers: Sequence[Hierarchy],
    k: int,
) -> list[tuple[str, ...]]:
    """Anonymize the QI tuples of a dataset -- a faithful implementation of the
    paper's ARX-LR local recoding (Algorithm 2):

      repeat:
        among all lattice nodes, find those that form >=1 k-group on the
        CURRENT remaining records; pick the one with the smallest
        (loss_g, c1, c2, c3) -- where c3 is the data-dependent distinct-interval
        retention criterion, re-evaluated on the remaining records each round;
        apply it (publish all its k-groups), remove those records.
      until no node forms a k-group; suppress the remaining records as outliers.

    Returns one generalized tuple per input record, in input order.
    """
    n = len(records_qi)
    m = len(hiers)
    if any(len(r) != m for r in records_qi):
        raise ValueError("record QI arity does not match number of hierarchies")

    from .segment_utils import ancestors_by_layer  # local import: avoid cycle

    anc: list[list[dict[int, str]]] = [
        [ancestors_by_layer(hiers[qi], records_qi[i][qi]) for qi in range(m)]
        for i in range(n)
    ]

    qloss_by_layer = []
    for h in hiers:
        d = {layer: max((nd.q_loss for nd in h.layers[layer].values()), default=0.0)
             for layer in range(1, h.height + 1)}
        qloss_by_layer.append(d)

    # Candidate nodes = all states except the all-root state, pre-sorted by the
    # data-independent key (loss_g, c1, c2). Group into tie-groups; c3 breaks ties
    # within a group, re-evaluated each round on the remaining records.
    all_root = tuple(h.height for h in hiers)
    nodes = [s for s in product(*[range(1, h.height + 1) for h in hiers])
             if s != all_root]
    nodes.sort(key=lambda s: _base_key(s, qloss_by_layer, hiers))
    tie_groups: list[list[tuple[int, ...]]] = []
    last_key = object()
    for s in nodes:
        key = _base_key(s, qloss_by_layer, hiers)
        if key != last_key:
            tie_groups.append([])
            last_key = key
        tie_groups[-1].append(s)

    def groups_at(state, remaining):
        g: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i in remaining:
            g[tuple(anc[i][qi][state[qi]] for qi in range(m))].append(i)
        return g

    def c3(state, remaining, dst_D):
        # 1 - (1/m) * sum_i dst(g(D).Qi)/dst(D.Qi); lower is finer -> preferred.
        total = 0.0
        for qi in range(m):
            dst_g = len({anc[i][qi][state[qi]] for i in remaining})
            total += dst_g / dst_D[qi] if dst_D[qi] else 0.0
        return 1.0 - total / m

    published: list[tuple[str, ...] | None] = [None] * n
    remaining: set[int] = set(range(n))

    while remaining:
        dst_D = [len({records_qi[i][qi] for i in remaining}) for qi in range(m)]
        chosen = None
        for group in tie_groups:                       # ascending (loss_g, c1, c2)
            forming = []
            for s in group:
                g = groups_at(s, remaining)
                if any(len(v) >= k for v in g.values()):
                    forming.append((s, g))
            if forming:
                # tie-break by c3 (data-dependent), then tuple for determinism
                chosen = min(forming, key=lambda sg: (c3(sg[0], remaining, dst_D), sg[0]))
                break
        if chosen is None:
            break                                      # no node forms a k-group
        _state, g = chosen
        for tup, members in g.items():
            if len(members) >= k:
                for i in members:
                    published[i] = tup
                remaining.difference_update(members)

    suppressed = tuple([SUPPRESSED] * m)
    for i in remaining:
        published[i] = suppressed
    return [p for p in published]  # type: ignore[misc]
