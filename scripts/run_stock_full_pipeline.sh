#!/usr/bin/env bash
set -euo pipefail

# Generic end-to-end thesis pipeline for one stock ticker.
#
# Main research route:
#   data -> reduced -> random_search -> walk_bestparams -> report_bestparams
#   -> baselines -> model_comparison -> summary
#
# Optional historical NVDA check:
#   legacy_05506 runs the old fixed best-parameter specification that produced
#   approximately OOS AUC 0.5506 on the historical NVDA thesis dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

SYMBOL="${SYMBOL:-NVDA}"
SYMBOL_UPPER="$(echo "$SYMBOL" | tr '[:lower:]' '[:upper:]')"
SYMBOL_LOWER="$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')"
KEYWORD="${KEYWORD:-${SYMBOL_UPPER} stock}"
SCAN_START="${SCAN_START:-2018-01-01}"
END="${END:-$(date -d yesterday +%F)}"
NEWS_START="${NEWS_START:-auto}"
NEWS_LIMIT_PER_DAY="${NEWS_LIMIT_PER_DAY:-25}"
MARKET_BUFFER_DAYS="${MARKET_BUFFER_DAYS:-45}"
FORCE="${FORCE:-0}"
GPU="${GPU:-0}"
TRIALS="${TRIALS:-50}"
RANDOM_EPOCHS="${RANDOM_EPOCHS:-50}"
WALK_EPOCHS="${WALK_EPOCHS:-30}"

TARGET_RAW_DIR="$ROOT/data/raw/stock_data/$SYMBOL_UPPER"
NEWS_RAW_DIR="$ROOT/data/raw/news_headlines_$SYMBOL_LOWER"
TRENDS_INTERIM="$ROOT/data/interim/${SYMBOL_LOWER}_trends_daily_consistent.csv"
TRENDS_PROCESSED="$ROOT/data/processed/${SYMBOL_LOWER}_trends_processed.csv"
NEWS_SENTIMENT="$ROOT/data/processed/${SYMBOL_LOWER}_news_daily_sentiment.csv"
TARGET_PROCESSED="$ROOT/data/processed/${SYMBOL_UPPER}_eod_processed.csv"
DATASET="$ROOT/data/model_feed/${SYMBOL_LOWER}_model_dataset.csv"
REDUCED_DATASET="$ROOT/data/model_feed/${SYMBOL_LOWER}_model_dataset_clean.csv"
AUDIT="$ROOT/data/model_feed/${SYMBOL_LOWER}_model_dataset_audit.xlsx"

RANDOM_OUT="$ROOT/artifacts/models/${SYMBOL_LOWER}_random_search_reduced_features"
BEST_OUT="$ROOT/artifacts/models/${SYMBOL_LOWER}_walk_forward_random_search_bestparams"
LEGACY_OUT="$ROOT/artifacts/models/${SYMBOL_LOWER}_walk_forward_legacy_05506_params"
FIXED_REDUCED_OUT="$ROOT/artifacts/models/${SYMBOL_LOWER}_walk_forward_reduced_fixedparams"
FIXED_FULL_OUT="$ROOT/artifacts/models/${SYMBOL_LOWER}_walk_forward_full_fixedparams"
BASELINE_OUT="$ROOT/artifacts/reports/${SYMBOL_LOWER}_baseline_models_linear_svm_ablations"
MODEL_COMPARISON_OUT="$ROOT/artifacts/reports/${SYMBOL_LOWER}_model_comparison"
SUMMARY_OUT="$ROOT/artifacts/reports/${SYMBOL_LOWER}_full_pipeline_summary.csv"

AUTO_PIPELINE="$ROOT/src/thesis/pipelines/run_stock_pipeline_auto_window.py"
GENERIC_PIPELINE="$ROOT/src/thesis/pipelines/run_stock_pipeline.py"
STOCK_FETCHER="$ROOT/src/thesis/One doc/nvda_stockdata_fetch_combined.py"
DATA_PIPELINE="$ROOT/src/thesis/One doc/data_pipeline_combined.py"
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

first_existing_csv() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -type f -name '*.csv' | sort | head -n 1
  fi
}

infer_news_start() {
  if [[ "$NEWS_START" != "auto" ]]; then
    echo "$NEWS_START"
    return 0
  fi

  "$PYTHON" - <<PY
from pathlib import Path
import pandas as pd
import sys

sent = Path('$NEWS_SENTIMENT')
raw_dir = Path('$NEWS_RAW_DIR')
scan = raw_dir / '${SYMBOL_UPPER}_news_availability_scan.csv'

candidates = []
if sent.exists():
    df = pd.read_csv(sent)
    for col in ['trading_date', 'date']:
        if col in df.columns and len(df):
            candidates.append(pd.to_datetime(df[col], errors='coerce').min())

if raw_dir.exists():
    for p in sorted(raw_dir.glob('*.csv')):
        if p.name.endswith('_news_fetch_progress.csv') or p.name.endswith('_news_availability_scan.csv'):
            continue
        try:
            df = pd.read_csv(p, usecols=lambda c: c in {'published_at', 'api_day'})
        except Exception:
            continue
        if 'api_day' in df.columns and len(df):
            candidates.append(pd.to_datetime(df['api_day'], errors='coerce').min())
        elif 'published_at' in df.columns and len(df):
            candidates.append(pd.to_datetime(df['published_at'], errors='coerce', utc=True).dt.date.min())

if scan.exists():
    try:
        df = pd.read_csv(scan)
        if 'scan_level' in df.columns and 'found' in df.columns:
            day = df[df['scan_level'].astype(str).eq('day')].copy()
            day['found_num'] = pd.to_numeric(day['found'], errors='coerce').fillna(0)
            hits = day[day['found_num'] > 0]
            if len(hits):
                candidates.append(pd.to_datetime(hits['range_start'], errors='coerce').min())
    except Exception:
        pass

candidates = [pd.Timestamp(x).date().isoformat() for x in candidates if pd.notna(x)]
if candidates:
    print(min(candidates))
else:
    sys.exit(3)
PY
}

resolve_window() {
  local inferred
  if inferred="$(infer_news_start 2>/dev/null)"; then
    RESOLVED_NEWS_START="$inferred"
    echo "[WINDOW] Inferred news_start locally: $RESOLVED_NEWS_START"
  else
    echo "[WINDOW] No local news_start found. Running API-conscious hierarchical news scan."
    run "$PYTHON" -u "$AUTO_PIPELINE" scan-news \
      --symbol "$SYMBOL_UPPER" \
      --keyword "$KEYWORD" \
      --scan-start "$SCAN_START" \
      --end "$END" \
      --news-limit-per-day "$NEWS_LIMIT_PER_DAY"
    RESOLVED_NEWS_START="$(infer_news_start)"
    echo "[WINDOW] Scanned news_start: $RESOLVED_NEWS_START"
  fi

  MARKET_START="$($PYTHON - <<PY
from datetime import datetime, timedelta
start = datetime.strptime('$RESOLVED_NEWS_START', '%Y-%m-%d').date()
print((start - timedelta(days=int('$MARKET_BUFFER_DAYS'))).isoformat())
PY
)"
  echo "[WINDOW] market/trends start=$MARKET_START news/model start=$RESOLVED_NEWS_START end=$END"
}

phase_env() {
  echo "ROOT=$ROOT"
  echo "PYTHON=$PYTHON"
  echo "SYMBOL=$SYMBOL_UPPER"
  echo "KEYWORD=$KEYWORD"
  echo "SCAN_START=$SCAN_START"
  echo "END=$END"
  echo "NEWS_START=$NEWS_START"
  echo "NEWS_LIMIT_PER_DAY=$NEWS_LIMIT_PER_DAY"
  echo "TRIALS=$TRIALS"
  echo "GPU=$GPU"
  run "$PYTHON" --version
  require_file "$AUTO_PIPELINE"
  require_file "$GENERIC_PIPELINE"
  require_file "$STOCK_FETCHER"
  require_file "$DATA_PIPELINE"
  require_file "$RANDOM_SCRIPT"
  require_file "$WALK_SCRIPT"
  require_file "$REPORT_SCRIPT"
}

clean_stock_file() {
  local symbol="$1"
  local input="$2"
  local output="$3"
  local pipeline_symbol="$symbol"
  case "$symbol" in
    NVDA|SPY|SOXX|IEF) pipeline_symbol="$symbol" ;;
    *) pipeline_symbol="NVDA" ;;
  esac
  run "$PYTHON" "$DATA_PIPELINE" stock-clean \
    --symbol "$pipeline_symbol" \
    --input "$input" \
    --output "$output"
}

ensure_stock_processed() {
  local symbol="$1"
  local raw_dir="$2"
  local processed="$3"

  if [[ "$FORCE" != "1" && -f "$processed" ]]; then
    echo "[SKIP] Processed EOD exists for $symbol: $processed"
    return 0
  fi

  local raw_csv
  raw_csv="$(first_existing_csv "$raw_dir" || true)"
  if [[ -n "$raw_csv" && "$FORCE" != "1" ]]; then
    echo "[REUSE] Raw EOD exists for $symbol: $raw_csv"
    clean_stock_file "$symbol" "$raw_csv" "$processed"
    return 0
  fi

  mkdir -p "$raw_dir"
  run "$PYTHON" "$STOCK_FETCHER" \
    --mode eod \
    --symbol "$symbol" \
    --start "$MARKET_START" \
    --end "$END" \
    --outdir "$raw_dir" \
    --csv \
    --continue_on_empty

  raw_csv="$(first_existing_csv "$raw_dir" || true)"
  if [[ -z "$raw_csv" ]]; then
    echo "No raw EOD CSV found after fetch for $symbol in $raw_dir" >&2
    exit 1
  fi
  clean_stock_file "$symbol" "$raw_csv" "$processed"
}

phase_data() {
  phase_env
  resolve_window

  ensure_stock_processed "$SYMBOL_UPPER" "$TARGET_RAW_DIR" "$TARGET_PROCESSED"
  ensure_stock_processed "SPY"  "$ROOT/data/raw/macro_stock_data/SPY"  "$ROOT/data/processed/SPY_eod_processed.csv"
  ensure_stock_processed "SOXX" "$ROOT/data/raw/macro_stock_data/SOXX" "$ROOT/data/processed/SOXX_eod_processed.csv"
  ensure_stock_processed "IEF"  "$ROOT/data/raw/macro_stock_data/IEF"  "$ROOT/data/processed/IEF_eod_processed.csv"

  if [[ "$FORCE" != "1" && -f "$TRENDS_PROCESSED" ]]; then
    echo "[SKIP] Processed Trends exists: $TRENDS_PROCESSED"
  else
    if [[ -f "$TRENDS_INTERIM" && "$FORCE" != "1" ]]; then
      echo "[REUSE] Trends interim exists: $TRENDS_INTERIM"
    else
      run "$PYTHON" -u "$GENERIC_PIPELINE" fetch-trends \
        --symbol "$SYMBOL_UPPER" \
        --keyword "$KEYWORD" \
        --start "$MARKET_START" \
        --end "$END"
    fi
    run "$PYTHON" "$GENERIC_PIPELINE" clean-trends \
      --symbol "$SYMBOL_UPPER" \
      --start "$MARKET_START" \
      --end "$END"
  fi

  if [[ "$FORCE" != "1" && -f "$NEWS_SENTIMENT" ]]; then
    echo "[SKIP] Processed daily news sentiment exists: $NEWS_SENTIMENT"
  else
    if find "$NEWS_RAW_DIR" -maxdepth 1 -type f -name '*.csv' ! -name '*progress.csv' ! -name '*availability_scan.csv' | grep -q . 2>/dev/null; then
      echo "[REUSE] Raw news CSVs exist in $NEWS_RAW_DIR. No StockData news API calls."
    else
      run "$PYTHON" -u "$GENERIC_PIPELINE" fetch-news \
        --symbol "$SYMBOL_UPPER" \
        --keyword "$KEYWORD" \
        --start "$MARKET_START" \
        --end "$END" \
        --news-start "$RESOLVED_NEWS_START" \
        --news-end "$END" \
        --news-limit-per-day "$NEWS_LIMIT_PER_DAY"
    fi
    run "$PYTHON" "$GENERIC_PIPELINE" build-sentiment \
      --symbol "$SYMBOL_UPPER" \
      --start "$MARKET_START" \
      --end "$END"
  fi

  run "$PYTHON" "$GENERIC_PIPELINE" build-model \
    --symbol "$SYMBOL_UPPER" \
    --keyword "$KEYWORD" \
    --start "$MARKET_START" \
    --end "$END"

  run "$PYTHON" "$GENERIC_PIPELINE" validate-audit \
    --symbol "$SYMBOL_UPPER" \
    --keyword "$KEYWORD" \
    --start "$MARKET_START" \
    --end "$END"
}

phase_reduced() {
  require_file "$DATASET"
  run "$PYTHON" - <<PY
from pathlib import Path
import pandas as pd

input_path = Path('$DATASET')
output_path = Path('$REDUCED_DATASET')
cols = [
    'date',
    'log_return', 'overnight_return', 'momentum_5d', 'momentum_20d',
    'volatility_20d', 'volume_change', 'volume_20d_avg',
    'avg_sentiment', 'sentiment_std', 'news_count',
    'spy_return', 'soxx_return', 'ief_return',
    'trends_zscore_30d', 'trends_momentum_7d', 'trends_spike',
    'target_direction', 'target_next_return',
]
df = pd.read_csv(input_path)
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f'Missing thesis feature columns: {missing}')
out = df[cols].replace([float('inf'), float('-inf')], pd.NA).dropna().reset_index(drop=True)
output_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(output_path, index=False)
print('Saved:', output_path)
print('Rows:', len(out))
print('Feature columns:', len([c for c in out.columns if c not in {'date', 'target_direction', 'target_next_return'}]))
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

python = Path('$PYTHON') if Path('$PYTHON').exists() else 'python3'
meta_path = Path('$RANDOM_OUT') / 'best/meta.json'
meta = json.loads(meta_path.read_text())
params = meta['params']
cmd = [
    str(python), '-u', '$WALK_SCRIPT',
    '--data', '$REDUCED_DATASET',
    '--outdir', '$BEST_OUT',
    '--lookback', str(params['lookback']),
    '--initial_train', '700',
    '--val_size', '126',
    '--test_horizon', '63',
    '--step', '63',
    '--epochs', os.environ.get('WALK_EPOCHS', '$WALK_EPOCHS'),
    '--batch', str(params['batch']),
    '--lr', str(params['lr']),
    '--lstm_units', str(params['lstm_units']),
    '--dense_units', str(params.get('dense_units', 64)),
    '--dropout', str(params['dropout']),
    '--recurrent_dropout', str(params['recurrent_dropout']),
    '--auto_threshold',
    '--gpu', os.environ.get('GPU', '$GPU'),
]
print('Best random-search hyperparameters:', params)
print('$ ' + ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
PY
}

phase_report_bestparams() {
  require_file "$BEST_OUT/walk_forward_oos_predictions.csv"
  run "$PYTHON" -u "$REPORT_SCRIPT" \
    --oos "$BEST_OUT/walk_forward_oos_predictions.csv" \
    --summary "$BEST_OUT/walk_forward_summary.json" \
    --outdir "$BEST_OUT/thesis_report_gross"
}

phase_walk_legacy_05506() {
  require_file "$REDUCED_DATASET"
  echo "[LEGACY] Running the historical NVDA best-parameter specification."
  echo "[LEGACY] Expected historical NVDA OOS AUC was approximately 0.5506, but retraining is stochastic."
  run "$PYTHON" -u "$WALK_SCRIPT" \
    --data "$REDUCED_DATASET" \
    --outdir "$LEGACY_OUT" \
    --lookback 90 \
    --initial_train 700 \
    --val_size 126 \
    --test_horizon 63 \
    --step 63 \
    --epochs "$WALK_EPOCHS" \
    --batch 64 \
    --lr 0.0003 \
    --lstm_units 96 \
    --dense_units 64 \
    --dropout 0.10 \
    --recurrent_dropout 0.20 \
    --auto_threshold \
    --gpu "$GPU"
  run "$PYTHON" -u "$REPORT_SCRIPT" \
    --oos "$LEGACY_OUT/walk_forward_oos_predictions.csv" \
    --summary "$LEGACY_OUT/walk_forward_summary.json" \
    --outdir "$LEGACY_OUT/thesis_report_gross"
}

phase_walk_reduced_fixed() {
  require_file "$REDUCED_DATASET"
  run "$PYTHON" -u "$WALK_SCRIPT" \
    --data "$REDUCED_DATASET" \
    --outdir "$FIXED_REDUCED_OUT" \
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

phase_walk_full_fixed() {
  require_file "$DATASET"
  run "$PYTHON" -u "$WALK_SCRIPT" \
    --data "$DATASET" \
    --outdir "$FIXED_FULL_OUT" \
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

phase_report_fixed() {
  if [[ -f "$FIXED_REDUCED_OUT/walk_forward_oos_predictions.csv" ]]; then
    run "$PYTHON" -u "$REPORT_SCRIPT" \
      --oos "$FIXED_REDUCED_OUT/walk_forward_oos_predictions.csv" \
      --summary "$FIXED_REDUCED_OUT/walk_forward_summary.json" \
      --outdir "$FIXED_REDUCED_OUT/thesis_report_gross"
  fi
  if [[ -f "$FIXED_FULL_OUT/walk_forward_oos_predictions.csv" ]]; then
    run "$PYTHON" -u "$REPORT_SCRIPT" \
      --oos "$FIXED_FULL_OUT/walk_forward_oos_predictions.csv" \
      --summary "$FIXED_FULL_OUT/walk_forward_summary.json" \
      --outdir "$FIXED_FULL_OUT/thesis_report_gross"
  fi
}

phase_baselines() {
  require_file "$REDUCED_DATASET"
  run "$PYTHON" -m thesis.eval.run_baseline_models_linear_svm \
    --dataset "$REDUCED_DATASET" \
    --run-ablations \
    --outdir "$BASELINE_OUT"
}

phase_model_comparison() {
  require_file "$BASELINE_OUT/tables/baseline_model_metrics.csv"
  local lstm_summary="$BEST_OUT/thesis_report_gross/report_summary.json"
  if [[ ! -f "$lstm_summary" ]]; then
    lstm_summary="$BEST_OUT/walk_forward_summary.json"
  fi
  require_file "$lstm_summary"
  run "$PYTHON" -m thesis.eval.make_model_comparison_table \
    --baseline-metrics "$BASELINE_OUT/tables/baseline_model_metrics.csv" \
    --feature-set all_features \
    --lstm-summary "$lstm_summary" \
    --outdir "$MODEL_COMPARISON_OUT"
}

phase_summary() {
  run "$PYTHON" - <<PY
import json
from pathlib import Path
import pandas as pd

items = {
    '${SYMBOL_LOWER}_random_search_meta': Path('$RANDOM_OUT') / 'best/meta.json',
    '${SYMBOL_LOWER}_walk_forward_bestparams': Path('$BEST_OUT') / 'walk_forward_summary.json',
    '${SYMBOL_LOWER}_walk_forward_bestparams_report': Path('$BEST_OUT') / 'thesis_report_gross/report_summary.json',
    '${SYMBOL_LOWER}_legacy_05506_params': Path('$LEGACY_OUT') / 'walk_forward_summary.json',
    '${SYMBOL_LOWER}_legacy_05506_report': Path('$LEGACY_OUT') / 'thesis_report_gross/report_summary.json',
    '${SYMBOL_LOWER}_fixed_reduced': Path('$FIXED_REDUCED_OUT') / 'walk_forward_summary.json',
    '${SYMBOL_LOWER}_fixed_full': Path('$FIXED_FULL_OUT') / 'walk_forward_summary.json',
}
rows = []
for run, path in items.items():
    if not path.exists():
        rows.append({'run': run, 'status': 'missing', 'path': str(path)})
        continue
    data = json.loads(path.read_text())
    row = {'run': run, 'status': 'ok', 'path': str(path)}
    if 'overall' in data:
        overall = data.get('overall', {})
        strat = overall.get('strategy', {}) or {}
        row.update({
            'auc': overall.get('auc'),
            'acc': overall.get('acc'),
            'sharpe': strat.get('sharpe_long_only'),
            'trade_rate': strat.get('trade_rate_long_only'),
            'feature_count': data.get('feature_count'),
        })
    elif 'classification' in data:
        cls = data.get('classification', {})
        trd = data.get('trading', {})
        row.update({
            'auc': cls.get('oos_auc'),
            'acc': cls.get('oos_acc'),
            'trade_rate': cls.get('trade_rate'),
            'sharpe': trd.get('annualized_sharpe'),
            'max_drawdown': trd.get('max_drawdown'),
            'mean_daily_return': trd.get('mean_daily_return'),
            'num_trades': trd.get('num_trades'),
        })
    else:
        row.update({
            'val_auc': data.get('val_auc'),
            'val_acc': data.get('val_acc'),
            'test_auc': data.get('test_auc'),
            'test_acc': data.get('test_acc'),
            'threshold': data.get('threshold'),
            'params': json.dumps(data.get('params', {})),
        })
    rows.append(row)

out = pd.DataFrame(rows)
out_path = Path('$SUMMARY_OUT')
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
print('\nSaved:', out_path)
print('Baseline metrics:', Path('$BASELINE_OUT') / 'tables/baseline_model_metrics.csv')
print('Model comparison:', Path('$MODEL_COMPARISON_OUT') / 'model_comparison_table.csv')
PY
}

phase_help() {
  cat <<EOF
Generic stock thesis pipeline

Configure with environment variables:
  SYMBOL=NVDA or AMD
  KEYWORD="NVIDIA stock" or "AMD stock"
  SCAN_START=2018-01-01
  END=2026-03-01
  NEWS_START=auto or explicit YYYY-MM-DD
  NEWS_LIMIT_PER_DAY=25
  TRIALS=50
  GPU=0

Main phases:
  env                 Check configuration and required scripts
  data                Raw checks/fetch -> process -> model dataset -> validate/audit
  reduced             Build thesis 16-feature clean dataset
  random_search       Hyperparameter search on reduced dataset
  walk_bestparams     Walk-forward using best random-search hyperparameters
  report_bestparams   Gross thesis report, tables and figures for bestparams run
  baselines           Majority/logistic/linear SVM/RF + feature ablations
  model_comparison    Combined LSTM + baseline table/figures
  summary             One CSV with key metrics

Optional comparison phases:
  legacy_05506        Old fixed NVDA bestparams route: lookback=90, LSTM=96, lr=0.0003
  walk_reduced_fixed  Fixed-parameter reduced walk-forward
  walk_full_fixed     Fixed-parameter full-feature walk-forward
  report_fixed        Gross reports for fixed comparison runs if present

Bundles:
  all                 data -> reduced -> random_search -> walk_bestparams -> report_bestparams -> baselines -> model_comparison -> fixed comparisons -> summary
  main                data -> reduced -> random_search -> walk_bestparams -> report_bestparams -> baselines -> model_comparison -> summary
  model_main          random_search -> walk_bestparams -> report_bestparams -> baselines -> model_comparison -> summary
  fixed_compare       walk_reduced_fixed -> walk_full_fixed -> report_fixed -> summary

Examples:
  SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 bash scripts/run_stock_full_pipeline.sh all
  SYMBOL=AMD  KEYWORD="AMD stock"    END=2026-02-26 TRIALS=50 GPU=0 bash scripts/run_stock_full_pipeline.sh all
  SYMBOL=NVDA KEYWORD="NVIDIA stock" GPU=0 bash scripts/run_stock_full_pipeline.sh legacy_05506
EOF
}

case "$phase" in
  env) phase_env ;;
  data) phase_data ;;
  reduced) phase_reduced ;;
  random_search) phase_random_search ;;
  walk_bestparams) phase_walk_bestparams ;;
  report_bestparams) phase_report_bestparams ;;
  baselines) phase_baselines ;;
  model_comparison) phase_model_comparison ;;
  summary) phase_summary ;;
  legacy_05506) phase_walk_legacy_05506 ;;
  walk_reduced_fixed) phase_walk_reduced_fixed ;;
  walk_full_fixed) phase_walk_full_fixed ;;
  report_fixed) phase_report_fixed ;;
  all)
    phase_data
    phase_reduced
    phase_random_search
    phase_walk_bestparams
    phase_report_bestparams
    phase_baselines
    phase_model_comparison
    phase_walk_reduced_fixed
    phase_walk_full_fixed
    phase_report_fixed
    phase_summary
    ;;
  main)
    phase_data
    phase_reduced
    phase_random_search
    phase_walk_bestparams
    phase_report_bestparams
    phase_baselines
    phase_model_comparison
    phase_summary
    ;;
  model_main)
    phase_random_search
    phase_walk_bestparams
    phase_report_bestparams
    phase_baselines
    phase_model_comparison
    phase_summary
    ;;
  fixed_compare)
    phase_walk_reduced_fixed
    phase_walk_full_fixed
    phase_report_fixed
    phase_summary
    ;;
  help|--help|-h) phase_help ;;
  *) echo "Unknown phase: $phase" >&2; phase_help; exit 2 ;;
esac
