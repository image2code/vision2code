#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
mkdir -p paper_assets/tables
"${PYTHON}" - <<'PYHUMAN'
from pathlib import Path
from vision2code.utils.io import read_csv, write_csv
root=Path('results/paper_outputs/human_validation'); out=Path('paper_assets/tables')
for name in ['human_alignment_correlations.csv','human_alignment_bootstrap_deltas.csv']:
    write_csv(out/name, read_csv(root/name)); print(out/name)
PYHUMAN
