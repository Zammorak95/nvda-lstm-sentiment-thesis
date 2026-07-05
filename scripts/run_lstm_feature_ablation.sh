#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SYMBOL="${SYMBOL:-NVDA}"
SYMBOL_LOWER="$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')"
GPU="${GPU:-0}"
WALK_EPOCHS="${WALK_EPOCHS:-20}"

DATASET="${DATASET:-$ROOT/data/model_feed/${SYMBOL_LOWER}_model_dataset_clean.csv}"
OUTDIR="${OUTDIR:-$ROOT/artifacts/models/${SYMBOL_LOWER}_lstm_feature_ablation}"
BEST_META="${BEST_META:-$ROOT/artifacts/models/${SYMBOL_LOWER}_random_search_reduced_features/best/meta.json}"

FORCE_ARG=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARG=(--force)
fi

echo "ROOT=$ROOT"
echo "PYTHON=$PYTHON"
echo "SYMBOL=$SYMBOL"
echo "DATASET=$DATASET"
echo "BEST_META=$BEST_META"
echo "OUTDIR=$OUTDIR"
echo "GPU=$GPU"
echo "WALK_EPOCHS=$WALK_EPOCHS"
echo

"$PYTHON" -m thesis.eval.run_lstm_feature_ablation \
  --dataset "$DATASET" \
  --outdir "$OUTDIR" \
  --best-meta "$BEST_META" \
  --python "$PYTHON" \
  --epochs "$WALK_EPOCHS" \
  --gpu "$GPU" \
  "${FORCE_ARG[@]}"
