#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${ROOT}:${ROOT}/third_party/alignn-main:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/phononscore-mplconfig}"

"${PYTHON_BIN}" -m phononscore.inference.score_cifs \
  --input "${ROOT}/examples/cifs" \
  --out "${ROOT}/examples/example_scores.csv" \
  --model phononscore_dft \
  --device cpu

echo "Smoke test output: ${ROOT}/examples/example_scores.csv"
