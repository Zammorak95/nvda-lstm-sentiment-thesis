#!/usr/bin/env bash
set -euo pipefail

# NVIDIA convenience wrapper for the generic stock thesis pipeline.
# This runs the main modelling route unless another phase is supplied:
#   random_search -> walk_bestparams -> report_bestparams -> summary

ROOT="${ROOT:-/home/zammorak/thesis}"
export SYMBOL="${SYMBOL:-NVDA}"
export KEYWORD="${KEYWORD:-NVIDIA stock}"

exec bash "$ROOT/scripts/run_stock_full_pipeline.sh" "${1:-model_main}"
