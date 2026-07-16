#!/bin/bash
# Headless ARX-LR anonymizer wrapper (real ARX, no GUI).
#
# Usage:
#   run_arx.sh <input.csv> <output.csv> <k> <qi_pipe> <hier_pipe> <hier_dir>
#
# where qi_pipe / hier_pipe are '|'-separated and parallel, e.g.
#   "Age|Systolic Blood Pressure"  "hierarchy_age.csv|hierarchy_systolic-blood-pressure.csv"
#
# Designed to slot into run_defense_experiment.py:
#   --backend external --arx-cmd 'arx/run_arx.sh {input} {output} {k} "Age|Systolic Blood Pressure" "hierarchy_age.csv|hierarchy_systolic-blood-pressure.csv" dataset/min1/hierarchies'
#
# Validated config: mode=iter, gsFactor=0.0, min level=1, local recoding on.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAVA="$HERE/jdk21/bin/java"
JAR="$HERE/libarx-3.9.2.jar"

INPUT="$1"; OUTPUT="$2"; K="$3"; QI="$4"; HIER="$5"; HIER_DIR="$6"

"$JAVA" -cp "$JAR:$HERE" ArxAnonymize \
    --input "$INPUT" --output "$OUTPUT" --k "$K" \
    --qi "$QI" --hier "$HIER" --hier-dir "$HIER_DIR" \
    --mode iter --gsfactor 0.0 --suppression 1.0 --min-level 1 \
    --max-iter 1000 --adaption 0.05
