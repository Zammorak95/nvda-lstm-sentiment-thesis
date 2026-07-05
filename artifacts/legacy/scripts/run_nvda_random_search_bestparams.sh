#!/usr/bin/env bash
set -euo pipefail

# Legacy NVIDIA random-search wrapper archived after introducing scripts/run_stock_full_pipeline.sh.
# Use the generic model route instead:
#   SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh model_main

ROOT="${ROOT:-/home/zammorak/thesis}"
export SYMBOL="${SYMBOL:-NVDA}"
export KEYWORD="${KEYWORD:-NVIDIA stock}"

exec bash "$ROOT/scripts/run_stock_full_pipeline.sh" "${1:-model_main}"
