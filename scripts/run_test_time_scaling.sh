#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
"${PYTHON}" -m vision2code.ablations.test_time_scaling.run_test_time_scaling "$@"

