#!/usr/bin/env bash
set -euo pipefail

# End-to-end NVIDIA thesis pipeline.
# Usage examples:
#   bash scripts/run_nvda_full_pipeline.sh data
#   bash scripts/run_nvda_full_pipeline.sh reduced
#   bash scripts/run_nvda_full_pipeline.sh walk_reduced
#   bash scripts/run_nvda_full_pipeline.sh report_reduced
#   bash scripts/run_nvda_full_pipeline.sh all
#
# Optional environment overrides:
#   END=2026-03-01 bash scripts/run_nvda_full_pipeline.sh all
#   NEWS_START=2020-01-01 END=2026-03-01 bash scripts/run_nvda_full_pipeline.sh data
#   TRIALS=20 GPU=0 bash scripts/run_nvda_full_pipeline.sh random_search

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
AUDIT="$ROOT/data/model_feed/nvda_model_dataset_audit.xlsx"

FULL_OUT="$ROOT/artifacts/models/nvda_walk_forward_full_features"
REDUCED_OUT="$ROOT/artifacts/models/nvda_walk_forward_reduced_features"
RANDOM_OUT="$ROOT/artifacts/models/nvda_random_search_reduced_features"

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

phase_summary() {
  run "$PYTHON" - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path('/home/zammorak/thesis')
items = {
    'nvda_full_features': root / 'artifacts/models/nvda_walk_forward_full_features/walk_forward_summary.json',
    'nvda_reduced_features': root / 'artifacts/models/nvda_walk_forward_reduced_features/walk_forward_summary.json',
    'nvda_random_search_reduced': root / 'artifacts/models/nvda_random_search_reduced_features/best/meta.json',
}

rows = []
for name, path in items.items():
    if not path.exists():
        rows.append({'run': name, 'path': str(path), 'status': 'missing'})
        continue
    data = json.loads(path.read_text())
    if 'overall' in data:
        overall = data.get('overall', {})
        strat = overall.get('strategy', {}) or {}
        rows.append({
            'run': name,
            'path': str(path),
            'status': 'ok',
            'auc': overall.get('auc'),
            'acc': overall.get('acc'),
            'sharpe': strat.get('sharpe_long_only'),
            'trade_rate': strat.get('trade_rate_long_only'),
            'feature_count': data.get('feature_count'),
        })
    else:
        rows.append({
            'run': name,
            'path': str(path),
            'status': 'ok',
            'val_auc': data.get('val_auc'),
            'val_acc': data.get('val_acc'),
            'test_auc': data.get('test_auc'),
            'test_acc': data.get('test_acc'),
            'threshold': data.get('threshold'),
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

Phases:
  env             Check paths and Python
  data            Fetch/process stock, news, trends; build dataset; validate; audit
  reduced         Build reduced feature dataset
  random_search   Run LSTM random search on reduced dataset
  walk_full       Walk-forward LSTM on full feature dataset
  walk_reduced    Walk-forward LSTM on reduced feature dataset
  report_full     Build figures/tables/stat report for full walk-forward
  report_reduced  Build figures/tables/stat report for reduced walk-forward
  summary         Collect key metrics into one CSV
  all             data -> reduced -> random_search -> walk_full -> walk_reduced -> reports -> summary
  fast            reduced -> walk_reduced -> report_reduced -> summary

Useful overrides:
  END=2026-03-01 bash scripts/run_nvda_full_pipeline.sh all
  NEWS_START=2020-01-01 END=2026-03-01 bash scripts/run_nvda_full_pipeline.sh data
  TRIALS=20 GPU=0 bash scripts/run_nvda_full_pipeline.sh random_search
  COST_BPS=5 bash scripts/run_nvda_full_pipeline.sh report_reduced
EOF
}

case "$phase" in
  env) phase_env ;;
  data) phase_data ;;
  reduced) phase_reduced ;;
  random_search) phase_random_search ;;
  walk_full) phase_walk_full ;;
  walk_reduced) phase_walk_reduced ;;
  report_full) phase_report_full ;;
  report_reduced) phase_report_reduced ;;
  summary) phase_summary ;;
  all)
    phase_data
    phase_reduced
    phase_random_search
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
