#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
DATA_DIR="${VISION2CODE_DATA_DIR:-}"
DATASET_SLUG="${VISION2CODE_KAGGLE_DATASET:-image2code/vision2code}"
while [[ $# -gt 0 ]]; do case "$1" in --data_dir) DATA_DIR="$2"; shift 2 ;; *) echo "unknown argument: $1" >&2; exit 2 ;; esac; done
if [[ -z "${DATA_DIR}" ]]; then DATA_DIR="data/kaggle/vision2code"; fi
mkdir -p "${DATA_DIR}"
if [[ -f "${DATA_DIR}/manifest.csv" ]]; then echo "Using existing Vision2Code data at ${DATA_DIR}"; exit 0; fi
if ! "${PYTHON}" -c "import kaggle.cli" >/dev/null 2>&1; then
  echo "Kaggle Python package is not installed in the active environment. Run: pip install -e \".[dev,eval]\"" >&2
  exit 1
fi
"${PYTHON}" -c "from kaggle.cli import main; raise SystemExit(main())" datasets download -d "${DATASET_SLUG}" -p "${DATA_DIR}" --unzip
