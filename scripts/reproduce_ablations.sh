#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
"${PYTHON}" -m vision2code.tables.reproduce --group ablations
