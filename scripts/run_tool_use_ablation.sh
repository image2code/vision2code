#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
"${PYTHON}" -m vision2code.ablations.tool_use.run_tool_ablation "$@"

