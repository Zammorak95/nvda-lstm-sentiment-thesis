# Historical NVDA result and current generic pipeline

## Historical result

A previous final NVDA thesis run produced approximately:

| Metric | Value |
|---|---:|
| OOS AUC | 0.5506 |
| OOS accuracy | 0.5179 |
| Strategy Sharpe | 0.9958 |
| Trade rate | 0.5397 |

The corresponding LSTM specification was:

```text
lookback=90
lstm_units=96
dense_units=64
learning_rate=0.0003
batch=64
dropout=0.10
recurrent_dropout=0.20
auto_threshold=True
```

## How it is handled now

The main pipeline remains generic. It does not hard-code the old metric and it does not treat the old NVDA result as a constant.

Instead, the historical parameter setting is available as an optional rerun:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 GPU=0 \
  bash scripts/run_stock_full_pipeline.sh legacy_05506
```

Output:

```text
artifacts/models/nvda_walk_forward_legacy_05506_params/
artifacts/models/nvda_walk_forward_legacy_05506_params/thesis_report_gross/
```

This lets the thesis retain a transparent connection to the historical result while keeping the current pipeline usable for AMD or any other ticker.

## Why the result may differ when rerun

A fresh rerun may not exactly reproduce `0.5506` because:

- TensorFlow/LSTM training is stochastic;
- CPU/GPU and ROCm/TensorFlow versions can change training behaviour;
- the data window or fetched raw data may differ slightly;
- Google Trends and news availability may differ depending on the collection date and source responses.

Therefore, the old value should be described as the historical final NVDA thesis run. The `legacy_05506` phase is the reproducible way to rerun the old parameter setting, not a guarantee of bit-for-bit identical results.

## Recommended thesis wording

> The final NVDA LSTM specification used in the original thesis run achieved an out-of-sample AUC of approximately 0.5506. To preserve reproducibility while keeping the repository generic, this parameter setting is retained as an optional legacy reproduction phase. Fresh reruns may differ slightly due to stochastic neural-network training and differences in the local software/hardware environment.
