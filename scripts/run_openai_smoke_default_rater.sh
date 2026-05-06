#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-.env}"

for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--env-file" && "${i}" -lt "$#" ]]; then
    next=$((i + 1))
    ENV_FILE="${!next}"
    break
  fi
done

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

PYTHON="${PYTHON:-python3}"
DATA_DIR="${VISION2CODE_DATA_DIR:-}"
SPLIT="${SPLIT:-test_mini}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MODEL="${MODEL:-gpt-5.4-mini}"
MODEL_SLUG="${MODEL_SLUG:-gpt_5_4_mini_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/outputs}"
RATER_MODEL="${RATER_MODEL:-Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}"
RATER_BASE_URL="${RATER_BASE_URL:-http://127.0.0.1:8000/v1}"
RATER_API_KEY="${RATER_API_KEY:-EMPTY}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-slug) MODEL_SLUG="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --rater-model) RATER_MODEL="$2"; shift 2 ;;
    --rater-base-url) RATER_BASE_URL="$2"; shift 2 ;;
    --rater-api-key) RATER_API_KEY="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${DATA_DIR}" ]]; then
  echo "Set VISION2CODE_DATA_DIR or pass --data_dir." >&2
  exit 1
fi

"${PYTHON}" scripts/check_local_rater.py \
  --base-url "${RATER_BASE_URL}" \
  --model "${RATER_MODEL}" \
  --api-key "${RATER_API_KEY}"

"${PYTHON}" -m vision2code.benchmark.run_benchmark \
  --provider openai \
  --model "${MODEL}" \
  --model-slug "${MODEL_SLUG}" \
  --data_dir "${DATA_DIR}" \
  --split "${SPLIT}" \
  --num_samples "${NUM_SAMPLES}" \
  --output_root "${OUTPUT_ROOT}" \
  --rater-provider local_vllm \
  --rater-model "${RATER_MODEL}" \
  --rater-base-url "${RATER_BASE_URL}" \
  --rater-api-key "${RATER_API_KEY}" \
  --env-file "${ENV_FILE}"
