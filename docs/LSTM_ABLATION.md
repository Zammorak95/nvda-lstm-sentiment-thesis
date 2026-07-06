# LSTM feature-group ablation

This ablation follows Wouter's suggestion directly. It compares the LSTM under four feature-group specifications:

1. `market_only`: conventional market variables only, including NVDA market features plus SPY, SOXX and IEF returns;
2. `market_sentiment`: market variables plus news sentiment variables;
3. `market_attention`: market variables plus Google Trends attention variables;
4. `full_model`: market variables, sentiment variables and Google Trends attention variables.

The purpose is to assess whether sentiment and attention variables add predictive value beyond conventional market information.

## Why the same LSTM hyperparameters are reused

The ablation reuses the best hyperparameters from the main random-search run. This keeps the comparison focused on the feature groups. Running a separate random search for every feature set would mix two effects: the value of the feature group and the advantage of additional hyperparameter tuning.

## Command

Run this after the normal main pipeline has produced the clean dataset and the random-search `best/meta.json` file:

```bash
cd ~/thesis
source .venv/bin/activate
git pull origin main_v2
chmod +x scripts/run_lstm_feature_ablation.sh

SYMBOL=NVDA GPU=0 WALK_EPOCHS=20 \
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_lstm_feature_ablation.sh
```

For AMD:

```bash
SYMBOL=AMD GPU=0 WALK_EPOCHS=20 \
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_lstm_feature_ablation.sh
```

Force rerun existing outputs:

```bash
SYMBOL=NVDA GPU=0 WALK_EPOCHS=20 FORCE=1 \
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_lstm_feature_ablation.sh
```

## Output

For NVDA, the summary is saved to:

```text
artifacts/models/nvda_lstm_feature_ablation/lstm_feature_ablation_summary.csv
```

Each feature set also receives its own walk-forward output folder and gross thesis report:

```text
artifacts/models/nvda_lstm_feature_ablation/market_only/
artifacts/models/nvda_lstm_feature_ablation/market_sentiment/
artifacts/models/nvda_lstm_feature_ablation/market_attention/
artifacts/models/nvda_lstm_feature_ablation/full_model/
```

## Thesis wording

A cautious thesis sentence could be:

> To assess the incremental value of alternative data, an LSTM feature-group ablation was performed following the supervisor's recommendation. The same walk-forward LSTM procedure and hyperparameter setting were applied to four specifications: market-only, market plus sentiment, market plus attention, and the full model. This design isolates whether sentiment and Google Trends variables improve out-of-sample performance beyond conventional market variables.

If the full model performs best, this supports incremental value. If the market-only model performs similarly or better, the conclusion should be more cautious: the alternative variables do not provide stable additional predictive value in this sample.
