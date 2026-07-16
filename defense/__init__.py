"""Defense pipeline for the CRA attack.

A small, self-contained extension on top of the existing `cra/` library that
injects synthetic ("fake") records before k-anonymization and measures whether
they weaken the Combinatorial Refinement Attack (CRA).

Stages:
    raw CSV
      -> optional fake-record injection      (noise_injector)
      -> ARX-LR k-anonymization              (lr_anonymizer = builtin_lr,
                                              or external/--skip-arx real ARX)
      -> CRA evaluation                      (metrics, reusing cra/)
      -> metrics report

Nothing in `cra/` is modified except one backward-compatible field added to
`solver.LPSpec` (`max_time_seconds`) so a single hard EQ cannot hang forever.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# The cra/ library uses top-level imports (it is meant to be run from inside
# cra/). Put it on sys.path here so every defense submodule can import it.
_CRA_DIR = _Path(__file__).resolve().parent.parent / "cra"
if str(_CRA_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CRA_DIR))
