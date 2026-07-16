# Combinatorial Refinement Attack (CRA) on ARX-LR

A Python implementation of **Algorithm 3 (per-EQ refinement)** and
**Algorithm 4 (outlier refinement)** from the paper
*Exposing Privacy Risks in Anonymizing Clinical Data: A Combinatorial
Refinement Attack on k-Anonymous Datasets Produced by ARX-LR*
([arXiv:2509.03350](https://arxiv.org/abs/2509.03350)).

The tool takes an ARX-LR-anonymised CSV and the generalisation hierarchies
used to produce it, then enumerates **every** integer record-to-basic-segment
assignment that is consistent with the published EQs. Each EQ that yields
exactly one such assignment is a 100%-confidence privacy breach.

## Quick start

```bash
git clone https://github.com/<your-org>/cra-arx-lr.git
cd cra-arx-lr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Single dataset
python eval_cra.py \
    --dataset dataset/min1/anonymized_2qi_k5.csv \
    --hier-dir dataset/min1/hierarchies \
    -k 5

# Full sweep across all (m, k) combinations
python run_sweep.py \
    --input-dir dataset/min1 \
    --hier-dir dataset/min1/hierarchies
```

> **Note:** The examples elsewhere in this README use `../dataset/min1/...`
> because the original development layout placed the code in `cra/` and the
> data in a sibling `dataset/`. For a fresh clone of this repository, drop
> the `../` prefix.

## Contents

- [What this tool does](#what-this-tool-does)
- [Requirements](#requirements)
- [Input file formats](#input-file-formats)
- [Critical ARX setting](#critical-arx-setting-min-generalisation-layer--1)
- [Usage](#usage)
- [Interpreting the metrics](#interpreting-the-metrics)
- [Project layout](#project-layout)
- [Citation](#citation)
- [License](#license)

## What this tool does

Given an anonymised dataset `D_gen` and its hierarchies, the pipeline:

1. **Extracts equivalence classes (EQs)** from `D_gen` — every distinct
   generalised tuple becomes one EQ.
2. **Sorts EQs** by ascending information loss, breaking ties by `c1`, `c2`,
   and finally the generalisation state (paper Eq. 2 + 3).
3. **Snap-merges EQs** whose post-snap grid footprints coincide (this
   handles ARX's local-recoding outputs cleanly).
4. **Builds and solves one CP-SAT linear program per EQ** (Algorithm 3):
   - Variables `z_b ∈ {0, 1, 2, …}` for each basic segment.
   - Sum constraint `∑ z_b = |EQ|`.
   - Overlap exclusion against prior EQs.
   - Halves constraint `≥ 1` per half-segment.
   - Sparse constraint `≤ k − 1` per sub-segment that is not itself an EQ.
5. **Runs Algorithm 4** for the outliers (records ARX fully suppressed),
   with a closed-form bypass when enumeration would explode combinatorially.
6. **Reports attack-precision metrics** (per-EQ + record-weighted).

## Requirements

- Python 3.10+
- Google OR-Tools (CP-SAT solver):

```bash
pip install ortools
```

That's the only third-party dependency. Everything else is in the standard
library.

A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install ortools
```

## Input file formats

### Anonymised CSV (`D_gen`)

One row per anonymised record. The header must include the QI column names
(matching the hierarchy filenames; see `eval_cra.py:QI_TO_HIER`). Other
columns — identifiers like `PATIENT`, sensitive attributes like
`Primary_Diagnosis` — are ignored at parse time. QI columns are
auto-detected: a column is treated as a QI iff every one of its values is
either an interval string (`[a, b[`) or the suppression marker `*`.

```csv
PATIENT,Age,Systolic Blood Pressure,Primary_Diagnosis
abc123,"[80, 85[","[120, 130[",Stress (finding)
def456,"[75, 80[","[130, 140[",Limited social contact (finding)
...
*,*,*,Suppressed_outlier_record
```

The `*,*,...,*` row format is ARX's standard suppression marker for
outliers; the tool collects all such rows into a single outlier set and
processes them via Algorithm 4.

### Raw CSV (used only by `verify_with_raw.py`)

The pre-anonymisation dataset, with identical column names for QIs. Used
only by the optional verification script to confirm the ground-truth
distribution appears in the enumerated CRA solutions.

```csv
PATIENT,Age,Systolic Blood Pressure,Primary_Diagnosis
abc123,82,123,Stress (finding)
def456,77,138,Limited social contact (finding)
```

Rows in the raw CSV do **not** need to be aligned with the anonymised CSV;
the verification works at the aggregate basic-segment level.

### Hierarchy CSVs (one per QI)

One row per **raw value**, columns ordered from most specific (raw point at
column 0) to most general (root in the last column). Every row must be the
same width.

```csv
18,"[15, 20[","[10, 20[","[10, 30[","[10, 50[","[10, 90["
19,"[15, 20[","[10, 20[","[10, 30[","[10, 50[","[10, 90["
20,"[20, 25[","[20, 30[","[10, 30[","[10, 50[","[10, 90["
...
```

Layer convention used internally:
- Column 0 → layer 0 (raw data point).
- Column 1 → layer 1 (**basic segment**; the LP variable index).
- Column `h` → layer `h` (root); `h^i = num_columns − 1`.

Hierarchy files must live in one directory, named:

```
hierarchy_age.csv
hierarchy_body-weight.csv
hierarchy_systolic-blood-pressure.csv
hierarchy_heart-rate.csv
```

(To register a new QI, edit the `QI_TO_HIER` dict at the top of
`eval_cra.py`.)

## Critical ARX setting: minimum generalisation layer = 1

When you generate the anonymised CSV in ARX, **set the minimum
generalisation level of every QI to 1**, not 0.

If ARX is allowed to emit layer-0 (raw) values in `D_gen`, two distinct EQs
can occupy the same basic segment (a layer-0 EQ at point `71` and a layer-1
EQ at interval `[70, 75[`), which violates the paper's overlap assumption
(`z_b = 0` for prior-claimed basics) and produces spurious infeasible EQs.
With `min layer = 1`, every record is generalised to at least the smallest
interval and the CRA framework applies cleanly.

The tool detects this case (`extraction.eqs` containing `qi_layers` with
`0`) but does **not** silently work around it; you'll see "EQs with no
solutions" anomalies in the report if you forget. Re-run ARX with `min
layer = 1` to fix.

## Usage

All examples assume you `cd cra/` first.

### Single-dataset evaluation

```bash
python eval_cra.py \
    --dataset ../dataset/min1/anonymized_2qi_k5.csv \
    --hier-dir ../dataset/min1/hierarchies \
    -k 5
```

For datasets with `m ≥ 4` QIs, **always pass `--max-solutions 10000`** (or
similar), otherwise a single broad-state EQ can take hours to enumerate:

```bash
python eval_cra.py \
    --dataset ../dataset/min1/anonymized_4qi_k5.csv \
    --hier-dir ../dataset/min1/hierarchies \
    -k 5 \
    --max-solutions 10000 \
    --verbose
```

Full CLI:

```text
python eval_cra.py --help
```

Output is a structured report broken into:
- **Dataset** — file, m, k, n, grid size, hierarchies per QI.
- **CRA run** — EQ counts (pre-merge / merged), outlier handling, wall
  times for Algorithm 3 / 4.
- **Attack precision (per EQ)** — Exact / High-Risk / Uncertain buckets +
  solution-count quartiles.
- **Attack precision (records in standard EQs)** — same buckets, weighted
  by `|EQ|`.
- **CRA Ratio** — `(enumerated solutions) / (stars-and-bars baseline)`.
- **Outliers (Algorithm 4)** — `|O|`, closed-form count, and enumerated
  count (if applicable).
- **Summary line** — one-liner for cross-dataset comparison.

### Sweep across many datasets

```bash
python run_sweep.py \
    --input-dir ../dataset/min1 \
    --hier-dir ../dataset/min1/hierarchies \
    --out-csv sweep_results.csv \
    --out-md sweep_results.md
```

Or with an explicit file list (e.g. to skip the slow 4-QI runs):

```bash
python run_sweep.py \
    --files ../dataset/min1/anonymized_2qi_k3.csv \
            ../dataset/min1/anonymized_2qi_k5.csv \
            ../dataset/min1/anonymized_2qi_k8.csv \
            ../dataset/min1/anonymized_3qi_k5.csv \
    --hier-dir ../dataset/min1/hierarchies \
    --out-csv quick.csv --out-md quick.md
```

Filenames in the input directory must match `anonymized_<m>qi_k<k>.csv`
(case-insensitive); `m` and `k` are parsed from the name. The driver writes:

- A full-detail **CSV** (every metric the EvalResult carries).
- A trimmed **Markdown** comparison table with the columns most useful for a
  report, followed by a glossary of every metric.

### Ground-truth verification (optional)

```bash
python verify_with_raw.py
```

Hard-codes paths to `dataset/min1/example-raw-dataset.csv` and the 2QI/k=5
anonymised file. Use it as a template if you want to confirm the
ground-truth distribution appears in the enumerated solutions on your own
data.

## Interpreting the metrics

A privacy breach in the CRA model is "the attacker enumerates a small set
of integer assignments and the true assignment is one of them." Fewer
solutions ⇒ higher attacker confidence.

| Metric | Meaning |
|---|---|
| **EQs (merged)** | Number of equivalence classes after snap-merge. |
| **\|O\|** | Number of records ARX fully suppressed (the outlier set). |
| **Exact EQ%** | EQs whose LP yields exactly 1 solution — **100%-confidence breach** for everyone inside. |
| **Exact Records%** | Same, weighted by `|EQ|`. The headline privacy number. |
| **High-Risk EQ%** | EQs with 2–5 solutions — attacker has ≥ 20% blind-guess probability. |
| **Uncertain EQ%** | EQs with > 5 solutions. Includes any EQ that hit the per-EQ cap (`Capped EQs`). |
| **CRA Ratio (mean)** | Mean of `(enumerated solutions) / (stars-and-bars baseline)` over solvable EQs. 1.0 means halves + sparse constraints didn't cut anything; smaller numbers mean the LP genuinely refined the candidate space. |
| **Outlier closed-form** | When `|O| ≤ k − 1` and halves are disabled (`|O| < k`), the outlier LP's solution count is exactly `C(\|O\| + free − 1, free − 1)` — we report this rather than enumerate millions of solutions. |

### Interpreting CRA Ratio carefully

`CRA Ratio = 1.0` does **not** mean "the attack failed." It means
"halves + sparse contributed nothing to the refinement, because the LP was
already trivial." Most basic-sized EQs fall into this category (`trivial ==
1`), so a high mean ratio for low-m datasets just reflects that almost
every EQ collapsed to a single basic.

For broad-state EQs the ratio drops sharply — that's where the algorithm's
combinatorial power shows.

### Privacy story across k

As you sweep `k=3 → 5 → 8`, Exact Records% should drop monotonically: more
generalisation forces ARX to publish broader EQs, and each broader EQ has
more candidate refinements. The sweep table makes this trend visible at a
glance.

## Project layout

```
cra/
├── README.md             <- you are here
├── eval_cra.py           <- single-dataset CLI + evaluate_dataset() API
├── run_sweep.py          <- multi-dataset driver, writes CSV + Markdown
├── verify_with_raw.py    <- ground-truth verification (optional)
│
├── cra.py                <- Algorithm 3 + Algorithm 4 orchestrators
├── solver.py             <- CP-SAT model builder + all-solutions enumerator
├── grid.py               <- m-dimensional basic-segment universe
├── segment.py            <- Segment / half-segment / descendant iteration
├── equivalence.py        <- EQ extraction, sorting, snap-merge
├── dataset.py            <- D_gen CSV reader (auto-detects QI columns)
├── hierarchy.py          <- Hierarchy CSV parser, layer-0 -> layer-1 snap
├── models.py             <- HierarchyNode, EquivalenceClass dataclasses
│
├── main.py               <- legacy Phase-1 smoke test (kept for compat)
├── smoke_phase3.py       <- Segment + Grid checks
├── smoke_phase3_solver.py<- isolated CP-SAT enumeration tests
└── smoke_phase3_cra.py   <- full pipeline + brute-force cross-check
```

`cra.py`, `solver.py`, `grid.py`, `segment.py`, `equivalence.py`,
`dataset.py`, `hierarchy.py`, `models.py` form the importable library;
`eval_cra.py`, `run_sweep.py`, `verify_with_raw.py` are the user-facing
scripts. The `smoke_phase3*.py` files are developer regression tests and
not needed for normal use.

## Citation

If you use this implementation in academic work, please cite the original
paper:

```bibtex
@misc{cra-arx-lr,
  title  = {Exposing Privacy Risks in Anonymizing Clinical Data:
            A Combinatorial Refinement Attack on k-Anonymous Datasets
            Produced by ARX-LR},
  eprint = {2509.03350},
  archivePrefix = {arXiv},
  year   = {2025},
  url    = {https://arxiv.org/abs/2509.03350}
}
```

## License

Released under the MIT License. See `LICENSE` for the full text.

The example clinical dataset under `dataset/min1/` is derived from
synthetic records (Synthea-style) and is included for reproducibility of
the metrics tables. Do not apply this tool to real patient data without
appropriate IRB approval and informed consent.
