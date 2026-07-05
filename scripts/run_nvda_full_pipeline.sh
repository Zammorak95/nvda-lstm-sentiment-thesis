#!/usr/bin/env bash
set -euo pipefail

# NVIDIA convenience wrapper for the generic stock thesis pipeline.
# The generic pipeline handles data acquisition/processing, reduced dataset
# construction, random search, best-parameter walk-forward evaluation, gross
# thesis reports, and summary tables.

ROOT="${ROOT:-/home/zammorak/thesis}"
export SYMBOL="${SYMBOL:-NVDA}"
export KEYWORD="${KEYWORD:-NVIDIA stock}"

exec bash "$ROOT/scripts/run_stock_full_pipeline.sh" "${1:-help}"
