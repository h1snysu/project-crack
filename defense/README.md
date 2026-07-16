# Defense pipeline against the CRA attack

A small extension on top of `cra/` that injects synthetic ("fake") records
**before** k-anonymization and measures whether they weaken the Combinatorial
Refinement Attack (CRA).

```
raw CSV
  -> optional fake-record injection   (defense/noise_injector.py)
  -> ARX-LR k-anonymization           (defense/lr_anonymizer.py = builtin_lr,
                                       or real ARX via --skip-arx)
  -> CRA evaluation                   (defense/metrics.py, reuses cra/)
  -> metrics report
```

Nothing in `cra/` was rewritten. The only change there is one backward-compatible
field on `solver.LPSpec` (`max_time_seconds`) so a single hard EQ cannot hang a
sweep.

## The anonymization backend

The repo's CRA runs on already-anonymized CSVs; ARX itself is an external Java
GUI and cannot be scripted for a 100+ run sweep. So this package ships
**`builtin_lr`**, a deterministic greedy local-recoding anonymizer that uses the
existing hierarchies.

> **Calibration:** on the 2-QI data, `builtin_lr` reproduces the real ARX export
> *exactly* — identical generalized-tuple multiset and identical CRA metrics for
> k=3 and k=5. For 2-QI work it is an ARX-faithful stand-in. For final claims you
> can still validate against a real ARX export of the noised input with
> `--skip-arx --anonymized <file>`. 3-/4-QI fidelity is not guaranteed and should
> be spot-checked against ARX.

## Install / environment

Use the existing venv (has OR-Tools):

```bash
cra/.venv/bin/python ...
```

Plots are optional and need matplotlib (not installed by default):

```bash
cra/.venv/bin/pip install matplotlib   # only if you want PNGs
```

## Run a single 2-QI experiment

Preferred defense:

```bash
cra/.venv/bin/python run_defense_experiment.py \
  --input dataset/min1/example-raw-dataset.csv \
  --output-dir out/defense_2qi_k5_sparse \
  --qi-cols "Age,Systolic Blood Pressure" \
  --k 5 --strategy sparse_basic_noise \
  --noise-budget 20 --max-noise-per-segment 2 \
  --seed 0 --cra-timeout-seconds 30
```

Baseline:

```bash
cra/.venv/bin/python run_defense_experiment.py \
  --input dataset/min1/example-raw-dataset.csv \
  --output-dir out/baseline_2qi_k5 \
  --qi-cols "Age,Systolic Blood Pressure" \
  --k 5 --strategy no_noise --seed 0 --cra-timeout-seconds 30
```

Validate against real ARX (skip the builtin anonymizer):

```bash
cra/.venv/bin/python run_defense_experiment.py \
  --input dataset/min1/example-raw-dataset.csv \
  --output-dir out/realarx_2qi_k5 \
  --qi-cols "Age,Systolic Blood Pressure" \
  --k 5 --strategy no_noise --skip-arx \
  --anonymized dataset/min1/anonymized_2qi_k5.csv
```

`--noise-budget` accepts an integer (`20`) or a percentage of dataset size
(`5%`).

## Run the sweep

```bash
cra/.venv/bin/python scripts/sweep_2qi_defense.py        # defaults below
# add --plots once matplotlib is installed
```

Defaults: strategies `no_noise, random_noise, sparse_basic_noise`; k `3,4,5`;
budgets `0,1%,2%,5%`; seeds `0,1,2`; 30 s/EQ timeout. Outputs:

- `out/defense_summary.csv` — every metric for every run
- `out/defense_summary.md` — trimmed comparison table + glossary + interpretation
- `out/sweep_2qi/<strategy>_k<k>_b<budget>_s<seed>/` — full per-run outputs

Override anything, e.g. `--k-values 3,4,5,6,7 --budgets 0,1%,2%,5%,10%`.

## Per-run output files

| file | contents |
|---|---|
| `raw_input.csv` | copy of the raw input |
| `noised_input.csv` | raw + fakes, with a `__fake__` 0/1 marker column |
| `fake_records.csv` | the injected fakes only (sidecar) |
| `anonymized_output.csv` | the k-anonymized dataset fed to the CRA |
| `cra_results.json` | summary + per-EQ solution detail |
| `metrics_summary.json` | one flat dict of all run metrics |
| `metrics_per_eq.csv` | one row per EQ (LR/CRA solutions, z_i flag, timing) |
| `run_config.json` | exact configuration of the run |
| `run.log` | timestamped stage log for debugging |

The `__fake__` marker is never treated as a QI (the CRA detects QIs by interval
shape and we pass `qi_columns` explicitly), so the public anonymized output is
unaffected by it; `fake_records.csv` is the sidecar for traceability.

## Defense strategies

- **no_noise** — baseline; input unchanged.
- **random_noise** — `noise_budget` fakes, each QI drawn uniformly from the QI's
  valid raw domain. Naive baseline.
- **sparse_basic_noise** *(preferred)* — find basic segments with `0 < count < k`
  and inject up to `max_noise_per_segment` fakes into each (ascending count
  first) until `noise_budget` is reached. Fake QI values are drawn from the raw
  points *inside* the chosen basic segment, so they stay valid and land there.
- **targeted_overlap_noise** *(optional)* — like sparse_basic but prioritizes
  sparse segments that share a layer-2 parent with populated neighbours (more
  likely to create/modify overlaps between equivalence classes).

All fakes use only raw values registered in the hierarchy (e.g. systolic skips
65/66/67), so every fake resolves cleanly. All randomness is seeded.

## Metric meanings

| metric | meaning |
|---|---|
| `n_fake_records`, `fake_record_pct` | injected fakes, absolute and % of original (utility/budget cost) |
| `eqs_evaluated`, `basic_segment_eqs`, `compound_segment_eqs` | EQ counts after snap-merge; basic = all QIs at layer 1 |
| `avg_info_loss_record_weighted`, `median_info_loss` | utility cost; lower is better |
| `lr_solutions` (per EQ) | stars-and-bars baseline: candidate integer assignments **before** CRA refinement |
| `cra_solutions` / `cra_int_assignments` (per EQ) | integer assignments CRA actually enumerates (with halves + sparse constraints) |
| `cra_ratio` = LR/CRA | >= 1; **higher = stronger attack** (CRA narrowed the space more). Closer to 1 = weaker attack |
| `cra_ratio_mean/median/max` | aggregated over solvable EQs |
| `pct_eqs_ratio_le_1_05`, `pct_eqs_ratio_le_2` | % EQs where CRA barely beat the baseline (higher = better defense) |
| `cra_int_assignments_median/mean` | attacker uncertainty per EQ (higher = better defense) |
| `exact_eq_pct` | % EQs pinned to a single solution. **Caveat:** a filled segment is "exact" but may be mostly fakes — not a real single-record breach |
| `single_out_risk_proxy` / `pct_eqs_with_single_out` | % EQs admitting a CRA solution with some basic `z_i = 1` (FPSO single-out proxy); **lower = better**. The most faithful privacy signal here |
| `injection_seconds`, `arx_seconds`, `cra_seconds`, `total_seconds` | per-stage runtime |
| `eqs_timed_out`, `eqs_capped` | EQs that hit `--cra-timeout-seconds` / `--max-solutions-per-eq` (their counts are lower bounds) |

`cra_ratio = LR_solutions / CRA_solutions` here is the **reciprocal** of the
"CRA Ratio" printed by `cra/eval_cra.py` (which reports CRA/baseline). We use
LR/CRA so that "more effective attack" reads as a larger number.

## Runtime control & known limitations

- **2-QI:** full CRA, fast (each run < 1 s). This is the priority path.
- **3-QI / 4-QI:** can explode. Use `--max-solutions-per-eq 10000`,
  `--cra-timeout-seconds`, and `--max-eqs`. A per-EQ timeout means that EQ's
  solution count is a *lower bound* (`eqs_timed_out` flags it).
- `builtin_lr` is calibrated exact only for 2-QI on this dataset; validate
  higher dimensions against real ARX.
- `max_noise_per_segment` caps how much of a percentage budget can actually be
  spent (a 5% budget may inject fewer if sparse-segment capacity is lower).

## Tests

```bash
cra/.venv/bin/python tests/test_defense.py          # or: python -m pytest tests/test_defense.py -q
```

Covers basic-segment mapping, fake validity inside intervals, budget and
per-segment caps, row-count conservation, no_noise no-op, determinism, and that
the anonymizer output is k-anonymous with min generalization layer 1.
