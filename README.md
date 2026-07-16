# project-crack

A defense study against the **Combinatorial Re-identification Attack (CRA)** on
k-anonymized clinical data (see the bundled paper, *Exposing Privacy Risks in
Anonymizing Clinical Data — Combinatorial*).

The pipeline takes a raw CSV, optionally injects fake records to raise the
attacker's ambiguity, k-anonymizes it (built-in LR anonymizer, or real ARX),
runs the CRA solver, and reports how much the attack was weakened versus the
utility cost.

```
raw CSV → fake-record injection → k-anonymization (LR / ARX) → CRA eval → metrics
```

## Layout

| Path                        | What it is                                             |
| --------------------------- | ------------------------------------------------------ |
| `run_defense_experiment.py` | Main entry point — one experiment end to end.          |
| `scripts/sweep_2qi_defense.py` | Sweeps strategy × k × noise-budget × seed and aggregates. |
| `defense/`                  | Noise injection, LR anonymizer, CRA metrics.           |
| `cra/`                      | The CRA attack solver library (has its own README).    |
| `arx/`                      | Headless real-ARX driver (Java + bundled JDK).         |
| `dataset/`                  | Raw + pre-anonymized example CSVs and QI hierarchies.  |
| `out/`                      | Generated results (git-ignored, regenerable).          |
| `tests/`                    | Sanity tests for the defense modules.                  |

## Setup

Requires Python 3.12. Create the virtualenv the scripts expect and install the
single dependency:

```bash
python3.12 -m venv cra/.venv
cra/.venv/bin/pip install -r cra/requirements.txt   # ortools
```

All commands below use `cra/.venv/bin/python` and are run **from the project
root**.

## Run

### Single experiment (preferred 2-QI sparse-noise defense)

```bash
cra/.venv/bin/python run_defense_experiment.py \
    --input dataset/min1/example-raw-dataset.csv \
    --output-dir out/defense_2qi_k5_sparse \
    --qi-cols "Age,Systolic Blood Pressure" \
    --k 5 --strategy sparse_basic_noise \
    --noise-budget 20 --max-noise-per-segment 2 \
    --seed 0 --cra-timeout-seconds 30
```

Results (anonymized CSV, `metrics_summary.json`, log) land in `--output-dir`.

### Baseline (no noise)

```bash
cra/.venv/bin/python run_defense_experiment.py \
    --input dataset/min1/example-raw-dataset.csv \
    --output-dir out/baseline_2qi_k5 \
    --qi-cols "Age,Systolic Blood Pressure" \
    --k 5 --strategy no_noise --seed 0 --cra-timeout-seconds 30
```

### Full sweep

Runs strategies × k × budget × seed and aggregates every run into
`out/defense_summary.csv` and `out/defense_summary.md`:

```bash
cra/.venv/bin/python scripts/sweep_2qi_defense.py
```

### Key options

- `--strategy` — `no_noise`, `random_noise`, `sparse_basic_noise`, `targeted_overlap_noise`
- `--qi-cols` — comma-separated quasi-identifier columns
- `--k` — k-anonymity level
- `--noise-budget` — fake records to add (integer, or `N%` of dataset size)
- `--backend` — `builtin_lr` (default) or `external` (real ARX, see below)
- `--skip-arx --anonymized <csv>` — feed a pre-anonymized CSV instead of anonymizing

See `cra/.venv/bin/python run_defense_experiment.py --help` for the full list.

## Tests

```bash
cra/.venv/bin/python tests/test_defense.py          # no pytest needed
# or, if pytest is installed:
cra/.venv/bin/pip install pytest
cra/.venv/bin/python -m pytest tests/test_defense.py -q
```

## Real ARX (optional)

The `builtin_lr` backend reproduces ARX's LR output for 2 QIs, so ARX is only
needed to cross-check against the real tool.

The bundled JDK (`arx/jdk21/`) and its tarball are git-ignored (too large for
git). On a fresh checkout, provision a JDK 21 first — extract your kept
`arx/jdk21.tar.gz`, or drop any JDK 21 at `arx/jdk21/` — then compile the driver:

```bash
cd arx
tar xzf jdk21.tar.gz                                     # if you kept the tarball
jdk21/bin/javac -cp libarx-3.9.2.jar ArxAnonymize.java   # rebuilds *.class (git-ignored)
```

Then run an experiment against real ARX:

```bash
cra/.venv/bin/python run_defense_experiment.py \
    --input dataset/min1/example-raw-dataset.csv \
    --output-dir out/realarx_2qi_k5 \
    --qi-cols "Age,Systolic Blood Pressure" --k 5 \
    --strategy no_noise --backend external \
    --arx-cmd 'arx/run_arx.sh {input} {output} {k} "Age|Systolic Blood Pressure" "hierarchy_age.csv|hierarchy_systolic-blood-pressure.csv" dataset/min1/hierarchies' \
    --cra-timeout-seconds 30
```

Validated ARX config: `mode=iter`, `gsFactor=0.0`, min level 1, local recoding on.
