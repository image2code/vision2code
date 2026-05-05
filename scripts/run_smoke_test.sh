#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
DATA_DIR="${VISION2CODE_DATA_DIR:-data/fixture_kaggle}"; NUM_SAMPLES=3
while [[ $# -gt 0 ]]; do case "$1" in --data_dir) DATA_DIR="$2"; shift 2 ;; --num_samples) NUM_SAMPLES="$2"; shift 2 ;; *) echo "unknown argument: $1" >&2; exit 2 ;; esac; done
"${PYTHON}" -m vision2code.data.validate_manifest --data_dir "${DATA_DIR}" --allow-small
"${PYTHON}" - <<'PYSMOKE'
from pathlib import Path
from vision2code.rendering.render_python import render_matplotlib_code
code = """import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1])
plt.savefig(OUTPUT_PATH)
"""
r=render_matplotlib_code(code,Path('/tmp/vision2code_smoke_render.png'))
print(r)
if not r['render_success']: raise SystemExit(1)
PYSMOKE
"${PYTHON}" -m vision2code.figures.make_leaderboard_tables --output-dir paper_assets/tables >/tmp/vision2code_smoke_tables.log
printf 'Smoke test completed with data_dir=%s num_samples=%s
' "${DATA_DIR}" "${NUM_SAMPLES}"
