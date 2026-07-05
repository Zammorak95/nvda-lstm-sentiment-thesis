#!/usr/bin/env bash
set -euo pipefail

# End-to-end NVIDIA thesis pipeline.
#
# Intended modelling logic:
#   1) build the dataset
#   2) build the reduced feature dataset used for the main model comparison
#   3) run random search on the reduced dataset
#   4) run walk-forward with the best random-search hyperparameters
#   5) also run fixed-parameter full/reduced variants as comparison checks
#   6) build cost-adjusted thesis reports and summary tables
#
# Usage examples:
#   bash scripts/run_nvda_full_pipeline.sh data
#   bash scripts/run_nvda_full_pipeline.sh reduced
#   TRIALS=50 GPU=0 bash scripts/run_nvda_full_pipeline.sh random_bestparams
#   END=2026-03-01 TRIALS=50 GPU=0 bash scripts/run_nvda_full_pipeline.sh all

ROOT="${ROOT:-/home/zammorak/thesis}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

SYMBOL="NVDA"
KEYWORD="${KEYWORD:-NVIDIA stock}"
SCAN_START="${SCAN_START:-2018-01-01}"
END="${END:-$(date -d yesterday +%F)}"
NEWS_START="${NEWS_START:-auto}"
NEWS_LIMIT_PER_DAY="${NEWS_LIMIT_PER_DAY:-10}"
MARKET_BUFFER_DAYS="${MARKET_BUFFER_DAYS:-45}"
REFRESH_MACRO="${REFRESH_MACRO:-0}"
FORCE="${FORCE:-0}"
GPU="${GPU:-0}"
TRIALS="${TRIALS:-50}"
RANDOM_EPOCHS="${RANDOM_EPOCHS:-50}"
WALK_EPOCHS="${WALK_EPOCHS:-30}"
COST_BPS="${COST_BPS:-5}"

DATASET="$ROOT/data/model_feed/nvda_model_dataset.csv"
REDUCED_DATASET="$ROOT/data/model_feed/nvda_model_dataset_clean.csv"

FULL_OUT="$ROOT/artifacts/models/nvda_walk_forward_full_features"
REDUCED_OUT="$ROOT/artifacts/models/nvda_walk_forward_reduced_features"
RANDOM_OUT="$ROOT/artifacts/models/nvda_random_search_reduced_features"
BEST_OUT="$ROOT/artifacts/models/nvda_walk_forward_random_search_bestparams"

AUTO_PIPELINE="$ROOT/src/thesis/pipelines/run_stock_pipeline_auto_window.py"
WALK_SCRIPT="$ROOT/src/thesis/model_training/optimalisation/walk_forward_lstm_direction_rocm.py"
RANDOM_SCRIPT="$ROOT/src/thesis/model_training/optimalisation/random_search_lstm_direction_v2.py"
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

pipeline_flags=()
if [[ "$NEWS_START" != "auto" ]]; then
  pipeline_flags+=(--news-start "$NEWS_START")
fi
if [[ "$REFRESH_MACRO" == "1" ]]; then
  pipeline_flags+=(--refresh-macro)
fi
if [[ "$FORCE" == "1" ]]; then
  pipeline_flags+=(--force)
fi

phase_env() {
  echo "ROOT=$ROOT"
  echo "PYTHON=$PYTHON"
  echo "SYMBOL=$SYMBOL"
  echo "KEYWORD=$KEYWORD"
  echo "SCAN_START=$SCAN_START"
  echo "END=$END"
  echo "NEWS_START=$NEWS_START"
  echo "NEWS_LIMIT_PER_DAY=$NEWS_LIMIT_PER_DAY"
  echo "GPU=$GPU"
  echo "TRIALS=$TRIALS"
  echo "RANDOM_EPOCHS=$RANDOM_EPOCHS"
  echo "WALK_EPOCHS=$WALK_EPOCHS"
  run "$PYTHON" --version
  require_file "$AUTO_PIPELINE"
  require_file "$WALK_SCRIPT"
  require_file "$RANDOM_SCRIPT"
  require_file "$REPORT_SCRIPT"
}

phase_data() {
  phase_env
  run "$PYTHON" -u "$AUTO_PIPELINE" all \
    --symbol "$SYMBOL" \
    --keyword "$KEYWORD" \
    --scan-start "$SCAN_START" \
    --end "$END" \
    --market-buffer-days "$MARKET_BUFFER_DAYS" \
    --news-limit-per-day "$NEWS_LIMIT_PER_DAY" \
    "${pipeline_flags[@]}"
}

phase_reduced() {
  require_file "$DATASET"
  run "$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('/home/zammorak/thesis')
input_path = root / 'data/model_feed/nvda_model_dataset.csv'
output_path = root / 'data/model_feed/nvda_model_dataset_clean.csv'

cols = [
    'date',
    'log_return',
    'overnight_return',
    'momentum_5d',
    'momentum_20d',
    'volatility_20d',
    'volume_change',
    'volume_20d_avg',
    'avg_sentiment',
    'spy_return',
    'soxx_return',
    'ief_return',
    'trends_zscore_30d',
    'trends_momentum_7d',
    'target_direction',
    'target_next_return',
]

df = pd.read_csv(input_path)
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f'Missing reduced feature columns: {missing}')

out = df[cols].replace([float('inf'), float('-inf')], pd.NA).dropna().reset_index(drop=True)
output_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(output_path, index=False)

print('Saved:', output_path)
print('Rows:', len(out))
print('Columns:', len(out.columns))
print('Range:', out['date'].min(), '->', out['date'].max())
print('Target balance:')
print(out['target_direction'].value_counts(normalize=True).round(4).sort_index())
PY
}

phase_random_search() {
  require_file "$REDUCED_DATASET"
  run "$PYTHON" -u "$RANDOM_SCRIPT" \
    --data "$REDUCED_DATASET" \
    --outdir "$RANDOM_OUT" \
    --trials "$TRIALS" \
    --max_epochs "$RANDOM_EPOCHS" \
    --auto_threshold \
    --gpu "$GPU"
}

phase_walk_bestparams() {
  require_file "$RANDOM_OUT/best/meta.json"
  require_file "$REDUCED_DATASET"
  run "$PYTHON" - <<PY
import json
import os
import subprocess
from pathlib import Path

root = Path('$ROOT')
python = Path('$PYTHON') if Path('$PYTHON').exists() else 'python3'
meta_path = root / 'artifacts/models/nvda_random_search_reduced_features/best/meta.json'
meta = json.loads(meta_path.read_text())
params = meta['params']
threshold = meta.get('threshold', 0.5)

def s(key):
    return str(params[key])

cmd = [
    str(python), '-u', str(root / 'src/thesis/model_training/optimalisation/walk_forward_lstm_direction_rocm.py'),
    '--data', str(root / 'data/model_feed/nvda_model_dataset_clean.csv'),
    '--outdir', str(root / 'artifacts/models/nvda_walk_forward_random_search_bestparams'),
    '--lookback', s('lookback'),
    '--initial_train', '700',
    '--val_size', '126',
    '--test_horizon', '63',
    '--step', '63',
    '--epochs', os.environ.get('WALK_EPOCHS', '30'),
    '--batch', s('batch'),
    '--lr', s('lr'),
    '--lstm_units', s('lstm_units'),
    '--dense_units', str(params.get('dense_units', 64)),
    '--dropout', s('dropout'),
    '--recurrent_dropout', s('recurrent_dropout'),
    '--threshold', str(threshold),
    '--gpu', os.environ.get('GPU', '0'),
]
print('Best random-search parameters:', params)
print('Validation-selected threshold:', threshold)
print('$ ' + ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
PY
}

phase_random_bestparams() {
  phase_random_search
  phase_walk_bestparams
  phase_report_bestparams
}

phase_walk_full() {
  require_file "$DATASET"
  run "$PYTHON" -u "$WALK_SCRIPT" \
    --data "$DATASET" \
    --outdir "$FULL_OUT" \
    --lookback 60 \
    --initial_train 700 \
    --val_size 126 \
    --test_horizon 63 \
    --step 63 \
    --epochs "$WALK_EPOCHS" \
    --batch 32 \
    --lr 0.0002 \
    --lstm_units 32 \
    --dense_units 64 \
    --dropout 0.05 \
    --recurrent_dropout 0.2 \
    --auto_threshold \
    --gpu "$GPU"
}

phase_walk_reduced() {
  require_file "$REDUCED_DATASET"
  run "$PYTHON" -u "$WALK_SCRIPT" \
    --data "$REDUCED_DATASET" \
    --outdir "$REDUCED_OUT" \
    --lookback 60 \
    --initial_train 700 \
    --val_size 126 \
    --test_horizon 63 \
    --step 63 \
    --epochs "$WALK_EPOCHS" \
    --batch 32 \
    --lr 0.0002 \
    --lstm_units 32 \
    --dense_units 64 \
    --dropout 0.05 \
    --recurrent_dropout 0.2 \
    --auto_threshold \
    --gpu "$GPU"
}

phase_report_full() {
  require_file "$FULL_OUT/walk_forward_oos_predictions.csv"
  run "$PYTHON" -u "$REPORT_SCRIPT" \
    --oos "$FULL_OUT/walk_forward_oos_predictions.csv" \
    --summary "$FULL_OUT/walk_forward_summary.json" \
    --outdir "$FULL_OUT/thesis_report_cost_${COST_BPS}bps" \
    --cost_bps "$COST_BPS"
}

phase_report_reduced() {
  require_file "$REDUCED_OUT/walk_forward_oos_predictions.csv"
  run "$PYTHON" -u "$REPORT_SCRIPT" \
    --oos "$REDUCED_OUT/walk_forward_oos_predictions.csv" \
    --summary "$REDUCED_OUT/walk_forward_summary.json" \
    --outdir "$REDUCED_OUT/thesis_report_cost_${COST_BPS}bps" \
    --cost_bps "$COST_BPS"
}

phase_report_bestparams() {
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
    'nvda_random_search_meta': root / 'artifacts/models/nvda_random_search_reduced_features/best/meta.json',
    'nvda_walk_forward_random_search_bestparams': root / 'artifacts/models/nvda_walk_forward_random_search_bestparams/walk_forward_summary.json',
    'nvda_reduced_fixed_params': root / 'artifacts/models/nvda_walk_forward_reduced_features/walk_forward_summary.json',
    'nvda_full_fixed_params': root / 'artifacts/models/nvda_walk_forward_full_features/walk_forward_summary.json',
}

rows = []
for name, path in items.items():
    if not path.exists():
        rows.append({'run': name, 'status': 'missing', 'path': str(path)})
        continue
    data = json.loads(path.read_text())
    if 'overall' in data:
        overall = data.get('overall', {})
        strat = overall.get('strategy', {}) or {}
        rows.append({
            'run': name,
            'status': 'ok',
            'auc': overall.get('auc'),
            'acc': overall.get('acc'),
            'sharpe': strat.get('sharpe_long_only'),
            'trade_rate': strat.get('trade_rate_long_only'),
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

out = pd.DataFrame(rows)
out_path = root / 'artifacts/models/nvda_results_pipeline_summary.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
print('\nSaved:', out_path)
PY
}

phase_help() {
  cat <<EOF
NVIDIA full thesis pipeline

Usage:
  bash scripts/run_nvda_full_pipeline.sh <phase>

Main phases:
  env                 Check paths and Python
  data                Fetch/process stock, news, trends; build dataset; validate; audit
  reduced             Build reduced feature dataset
  random_search       Run LSTM random search on reduced dataset
  walk_bestparams     Run walk-forward using best random-search parameters
  random_bestparams   random_search -> walk_bestparams -> report_bestparams
  report_bestparams   Build report for best random-search walk-forward
  summary             Collect key metrics into one CSV

Comparison phases:
  walk_full           Fixed-parameter walk-forward on full feature dataset
  walk_reduced        Fixed-parameter walk-forward on reduced feature dataset
  report_full         Report for fixed full-feature walk-forward
  report_reduced      Report for fixed reduced-feature walk-forward

Bundles:
  all                 data -> reduced -> random_search -> walk_bestparams -> report_bestparams -> fixed comparison runs -> summary
  model_main          random_search -> walk_bestparams -> report_bestparams -> summary
  fixed_compare       walk_full -> walk_reduced -> report_full -> report_reduced -> summary
  fast                reduced -> walk_reduced -> report_reduced -> summary

Useful overrides:
  END=2026-03-01 TRIALS=50 GPU=0 bash scripts/run_nvda_full_pipeline.sh all
  TRIALS=20 GPU=0 bash scripts/run_nvda_full_pipeline.sh model_main
  NEWS_START=2020-01-01 END=2026-03-01 bash scripts/run_nvda_full_pipeline.sh data
  COST_BPS=5 bash scripts/run_nvda_full_pipeline.sh report_bestparams
EOF
}

case "$phase" in
  env) phase_env ;;
  data) phase_data ;;
  reduced) phase_reduced ;;
  random_search) phase_random_search ;;
  walk_bestparams) phase_walk_bestparams ;;
  random_bestparams) phase_random_bestparams ;;
  walk_full) phase_walk_full ;;
  walk_reduced) phase_walk_reduced ;;
  report_full) phase_report_full ;;
  report_reduced) phase_report_reduced ;;
  report_bestparams) phase_report_bestparams ;;
  summary) phase_summary ;;
  all)
    phase_data
    phase_reduced
    phase_random_search
    phase_walk_bestparams
    phase_report_bestparams
    phase_walk_full
    phase_walk_reduced
    phase_report_full
    phase_report_reduced
    phase_summary
    ;;
  model_main)
    phase_random_search
    phase_walk_bestparams
    phase_report_bestparams
    phase_summary
    ;;
  fixed_compare)
    phase_walk_full
    phase_walk_reduced
    phase_report_full
    phase_report_reduced
    phase_summary
    ;;
  fast)
    phase_reduced
    phase_walk_reduced
    phase_report_reduced
    phase_summary
    ;;
  help|--help|-h) phase_help ;;
  *) echo "Unknown phase: $phase" >&2; phase_help; exit 2 ;;
esac
