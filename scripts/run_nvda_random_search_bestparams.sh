#!/usr/bin/env bash
set -euo pipefail

# Run NVIDIA random search and then a walk-forward run with the best parameters.
# Usage:
#   bash scripts/run_nvda_random_search_bestparams.sh all
#   TRIALS=20 GPU=0 bash scripts/run_nvda_random_search_bestparams.sh all
#   bash scripts/run_nvda_random_search_bestparams.sh walk_best

ROOT="${ROOT:-/home/zammorak/thesis}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

GPU="${GPU:-0}"
TRIALS="${TRIALS:-50}"
RANDOM_EPOCHS="${RANDOM_EPOCHS:-50}"
WALK_EPOCHS="${WALK_EPOCHS:-30}"
COST_BPS="${COST_BPS:-5}"

DATASET="$ROOT/data/model_feed/nvda_model_dataset_clean.csv"
RANDOM_OUT="$ROOT/artifacts/models/nvda_random_search_reduced_features"
BEST_OUT="$ROOT/artifacts/models/nvda_walk_forward_random_search_bestparams"
RANDOM_SCRIPT="$ROOT/src/thesis/model_training/optimalisation/random_search_lstm_direction_v2.py"
WALK_SCRIPT="$ROOT/src/thesis/model_training/optimalisation/walk_forward_lstm_direction_rocm.py"
REPORT_SCRIPT="$ROOT/src/thesis/eval/thesis_walkforward_report.py"

phase="${1:-help}"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

run() {
  echo
  echo "$ $*"
  "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

phase_random_search() {
  require_file "$DATASET"
  run "$PYTHON" -u "$RANDOM_SCRIPT" \
    --data "$DATASET" \
    --outdir "$RANDOM_OUT" \
    --trials "$TRIALS" \
    --max_epochs "$RANDOM_EPOCHS" \
    --auto_threshold \
    --gpu "$GPU"
}

phase_walk_best() {
  require_file "$RANDOM_OUT/best/meta.json"
  require_file "$DATASET"
  run "$PYTHON" - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path('/home/zammorak/thesis')
python = root / '.venv/bin/python'
if not python.exists():
    python = 'python3'

meta_path = root / 'artifacts/models/nvda_random_search_reduced_features/best/meta.json'
meta = json.loads(meta_path.read_text())
params = meta['params']
threshold = meta.get('threshold', 0.5)

cmd = [
    str(python), '-u',
    str(root / 'src/thesis/model_training/optimalisation/walk_forward_lstm_direction_rocm.py'),
    '--data', str(root / 'data/model_feed/nvda_model_dataset_clean.csv'),
    '--outdir', str(root / 'artifacts/models/nvda_walk_forward_random_search_bestparams'),
    '--lookback', str(params['lookback']),
    '--initial_train', '700',
    '--val_size', '126',
    '--test_horizon', '63',
    '--step', '63',
    '--epochs', '30',
    '--batch', str(params['batch']),
    '--lr', str(params['lr']),
    '--lstm_units', str(params['lstm_units']),
    '--dense_units', str(params.get('dense_units', 64)),
    '--dropout', str(params['dropout']),
    '--recurrent_dropout', str(params['recurrent_dropout']),
    '--threshold', str(threshold),
]
cmd += ['--gpu', '0']
print('$ ' + ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
PY
}

phase_report_best() {
  require_file "$BEST_OUT/walk_forward_oos_predictions.csv"
  run "$PYTHON" -u "$REPORT_SCRIPT" \
    --oos "$BEST_OUT/walk_forward_oos_predictions.csv" \
    --summary "$BEST_OUT/walk_forward_summary.json" \
    --outdir "$BEST_OUT/thesis_report_cost_${COST_BPS}bps" \
    --cost_bps "$COST_BPS"
}

phase_summary() {
  run "$PYTHON" - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path('/home/zammorak/thesis')
items = {
    'nvda_reduced_fixed_params': root / 'artifacts/models/nvda_walk_forward_reduced_features/walk_forward_summary.json',
    'nvda_full_fixed_params': root / 'artifacts/models/nvda_walk_forward_full_features/walk_forward_summary.json',
    'nvda_random_search_meta': root / 'artifacts/models/nvda_random_search_reduced_features/best/meta.json',
    'nvda_walk_forward_random_bestparams': root / 'artifacts/models/nvda_walk_forward_random_search_bestparams/walk_forward_summary.json',
}
rows = []
for name, path in items.items():
    if not path.exists():
        rows.append({'run': name, 'status': 'missing', 'path': str(path)})
        continue
    data = json.loads(path.read_text())
    if 'overall' in data:
        overall = data.get('overall', {})
        strategy = overall.get('strategy', {}) or {}
        rows.append({
            'run': name,
            'status': 'ok',
            'auc': overall.get('auc'),
            'acc': overall.get('acc'),
            'sharpe': strategy.get('sharpe_long_only'),
            'trade_rate': strategy.get('trade_rate_long_only'),
            'feature_count': data.get('feature_count'),
            'path': str(path),
        })
    else:
        rows.append({
            'run': name,
            'status': 'ok',
            'val_auc': data.get('val_auc'),
            'val_acc': data.get('val_acc'),
            'test_auc': data.get('test_auc'),
            'test_acc': data.get('test_acc'),
            'threshold': data.get('threshold'),
            'params': json.dumps(data.get('params', {})),
            'path': str(path),
        })
summary = pd.DataFrame(rows)
out = root / 'artifacts/models/nvda_random_search_pipeline_summary.csv'
summary.to_csv(out, index=False)
print(summary.to_string(index=False))
print('\nSaved:', out)
PY
}

phase_help() {
  cat <<EOF
NVIDIA random-search + best-params pipeline

Usage:
  bash scripts/run_nvda_random_search_bestparams.sh <phase>

Phases:
  random_search   Run random search on nvda_model_dataset_clean.csv
  walk_best       Run walk-forward using best random-search hyperparameters
  report_best     Build cost/report output for best-params walk-forward
  summary         Save comparison CSV
  all             random_search -> walk_best -> report_best -> summary

Useful:
  TRIALS=20 GPU=0 bash scripts/run_nvda_random_search_bestparams.sh all
EOF
}

case "$phase" in
  random_search) phase_random_search ;;
  walk_best) phase_walk_best ;;
  report_best) phase_report_best ;;
  summary) phase_summary ;;
  all)
    phase_random_search
    phase_walk_best
    phase_report_best
    phase_summary
    ;;
  help|--help|-h) phase_help ;;
  *) echo "Unknown phase: $phase" >&2; phase_help; exit 2 ;;
esac
