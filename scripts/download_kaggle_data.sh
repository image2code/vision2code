#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${IMAGE2CODE_DATA_DIR:-}"
while [[ $# -gt 0 ]]; do case "$1" in --data_dir) DATA_DIR="$2"; shift 2 ;; *) echo "unknown argument: $1" >&2; exit 2 ;; esac; done
if [[ -z "${DATA_DIR}" ]]; then DATA_DIR="data/kaggle/image2code-neurips-2026"; fi
mkdir -p "${DATA_DIR}"
if [[ -f "${DATA_DIR}/manifest.csv" ]]; then echo "Using existing Image2Code data at ${DATA_DIR}"; exit 0; fi
if command -v kaggle >/dev/null 2>&1; then kaggle datasets download -d image2code/image2code-neurips-2026 -p "${DATA_DIR}" --unzip; else echo "Kaggle CLI not found. Download manually and set IMAGE2CODE_DATA_DIR or pass --data_dir." >&2; exit 1; fi
