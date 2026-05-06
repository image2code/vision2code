#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
"${PYTHON}" -m vision2code.ablations.tool_use.render_outputs "$@"
