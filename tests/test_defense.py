"""Unit / sanity tests for the defense modules.

Run from project root:
    cra/.venv/bin/python -m pytest tests/test_defense.py -q
or without pytest:
    cra/.venv/bin/python tests/test_defense.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from defense.segment_utils import (        # noqa: E402
    load_hierarchies, basic_segment_of, segment_counts, sparse_segments,
    raw_values_under_basic, all_raw_values, ancestors_by_layer,
)
from defense.noise_injector import inject, MARKER                # noqa: E402
from defense.lr_anonymizer import anonymize_qi                   # noqa: E402

HIER = _ROOT / "dataset/min1/hierarchies"
QI = ["Age", "Systolic Blood Pressure"]
K = 5


def _load():
    import csv
    rows = list(csv.DictReader((_ROOT / "dataset/min1/example-raw-dataset.csv").open()))
    header = list(rows[0].keys())
    hiers = load_hierarchies(QI, HIER)
    return rows, header, hiers


def test_basic_segment_mapping():
    _, _, hiers = _load()
    # Age 58 -> [55,60[ ; Systolic 123 -> [120,130[
    seg = basic_segment_of(["58", "123"], hiers)
    assert seg == ("[55, 60[", "[120, 130["), seg
    # ancestors include layer 1 and go up to root
    anc = ancestors_by_layer(hiers[0], "58")
    assert anc[1] == "[55, 60[" and 1 in anc and max(anc) == hiers[0].height


def test_fake_values_inside_basic_segment():
    _, _, hiers = _load()
    target = ("[30, 35[", "[130, 140[")
    age_pool = raw_values_under_basic(hiers[0], target[0])
    sys_pool = raw_values_under_basic(hiers[1], target[1])
    assert set(age_pool) <= {"30", "31", "32", "33", "34"}
    # every sampled raw value must map back into the target basic segment
    for a in age_pool:
        for s in sys_pool:
            assert basic_segment_of([a, s], hiers) == target


def test_respect_noise_budget():
    rows, header, hiers = _load()
    res = inject(rows, header, QI, hiers, "sparse_basic_noise", K,
                 noise_budget=13, max_noise_per_segment=2, seed=0)
    assert res.n_fake <= 13
    assert res.n_fake == 13  # enough sparse segments exist to spend the budget


def test_respect_max_noise_per_segment():
    rows, header, hiers = _load()
    res = inject(rows, header, QI, hiers, "sparse_basic_noise", K,
                 noise_budget=1000, max_noise_per_segment=1, seed=0)
    assert all(c <= 1 for c in res.per_segment_added.values())
    # fakes land only in segments that were sparse (count < k) originally
    counts = segment_counts([[r[q] for q in QI] for r in rows], hiers)
    sparse = set(sparse_segments(counts, K))
    assert set(res.per_segment_added) <= sparse


def test_fakes_are_valid_and_in_target_segments():
    rows, header, hiers = _load()
    res = inject(rows, header, QI, hiers, "sparse_basic_noise", K,
                 noise_budget=20, max_noise_per_segment=2, seed=1)
    for fr in res.fake_rows:
        seg = basic_segment_of([fr[q] for q in QI], hiers)
        assert seg in res.per_segment_added  # fake sits in a targeted segment
        assert fr[MARKER] == "1"


def test_output_rowcount_equals_original_plus_fakes():
    rows, header, hiers = _load()
    for strat in ("random_noise", "sparse_basic_noise", "targeted_overlap_noise"):
        res = inject(rows, header, QI, hiers, strat, K,
                     noise_budget=15, max_noise_per_segment=2, seed=2)
        assert len(res.rows) == len(rows) + res.n_fake
        assert res.n_real == len(rows)


def test_no_noise_unchanged():
    rows, header, hiers = _load()
    res = inject(rows, header, QI, hiers, "no_noise", K,
                 noise_budget=50, max_noise_per_segment=2, seed=0)
    assert res.n_fake == 0
    assert len(res.rows) == len(rows)
    for orig, out in zip(rows, res.rows):
        for q in QI:
            assert orig[q] == out[q]


def test_anonymizer_k_anonymous_and_min_layer_1():
    rows, _, hiers = _load()
    rec = [[r[q] for q in QI] for r in rows]
    gen = anonymize_qi(rec, hiers, K)
    from collections import Counter
    c = Counter(gen)
    supp = tuple(["*"] * len(QI))
    for tup, n in c.items():
        if tup == supp:
            continue
        assert n >= K, f"published EQ {tup} has {n} < k records"
        # min generalization layer = 1: no raw layer-0 point leaks through
        for v in tup:
            assert v.startswith("[") or v == "*", v


def test_determinism_same_seed():
    rows, header, hiers = _load()
    a = inject(rows, header, QI, hiers, "sparse_basic_noise", K, 20, 2, seed=7)
    b = inject(rows, header, QI, hiers, "sparse_basic_noise", K, 20, 2, seed=7)
    assert [r[q] for r in a.fake_rows for q in QI] == \
           [r[q] for r in b.fake_rows for q in QI]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
