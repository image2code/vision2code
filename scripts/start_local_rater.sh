#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
MODEL_PATH="${RATER_MODEL_PATH:-${RATER_MODEL:-${DEFAULT_MODEL}}}"
SERVED_MODEL_NAME="${RATER_SERVED_MODEL_NAME:-${DEFAULT_MODEL}}"
HOST="${RATER_HOST:-127.0.0.1}"
PORT="${RATER_PORT:-8000}"
API_KEY="${RATER_API_KEY:-EMPTY}"
VLLM_BIN="${VLLM_BIN:-vllm}"

if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
  echo "vLLM executable not found: ${VLLM_BIN}" >&2
  echo "Install vLLM in this environment or set VLLM_BIN=/path/to/vllm." >&2
  exit 1
fi

echo "Starting local rater server"
echo "Model path/id: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Base URL: http://${HOST}:${PORT}/v1"

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --api-key "${API_KEY}" \
  ${VLLM_EXTRA_ARGS:-}

