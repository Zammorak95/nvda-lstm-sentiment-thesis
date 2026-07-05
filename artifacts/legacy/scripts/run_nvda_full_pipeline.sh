#!/usr/bin/env bash
set -euo pipefail

# Legacy NVIDIA convenience wrapper archived after introducing scripts/run_stock_full_pipeline.sh.
# Use the generic pipeline instead:
#   SYMBOL=NVDA KEYWORD="NVIDIA stock" bash scripts/run_stock_full_pipeline.sh all

ROOT="${ROOT:-/home/zammorak/thesis}"
export SYMBOL="${SYMBOL:-NVDA}"
export KEYWORD="${KEYWORD:-NVIDIA stock}"

exec bash "$ROOT/scripts/run_stock_full_pipeline.sh" "${1:-help}"
