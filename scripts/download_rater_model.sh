#!/usr/bin/env bash
set -euo pipefail

MODEL="${RATER_MODEL:-Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is not installed. Install it with: pip install huggingface_hub" >&2
  exit 1
fi

ARGS=(download "${MODEL}")
if [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
  ARGS+=(--cache-dir "${HUGGINGFACE_HUB_CACHE}")
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  ARGS+=(--token "${HF_TOKEN}")
fi

echo "Downloading rater model: ${MODEL}"
echo "Cache: ${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-Hugging Face default cache}}"
huggingface-cli "${ARGS[@]}"
