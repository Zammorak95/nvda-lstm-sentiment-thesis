#!/usr/bin/env python3
"""Run LSTM feature-group ablations using Wouter's suggested design.

The goal is to test whether sentiment and attention variables add predictive
information beyond conventional market variables, while keeping the LSTM
training/evaluation procedure fixed and comparable across feature sets.

Feature sets:
- market_only: market + macro/sector/bond variables
- market_sentiment: market_only + news sentiment variables
- market_attention: market_only + Google Trends attention variables
- full_model: market_only + sentiment + attention variables

By default, the script reuses the best hyperparameters from the main random
search meta.json. This keeps the ablation focused on the feature groups rather
than giving every feature set a separate hyperparameter search advantage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


MARKET_FEATURES = [
    "log_return",
    "overnight_return",
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "volume_change",
    "volume_20d_avg",
]
MACRO_FEATURES = ["spy_return", "soxx_return", "ief_return"]
SENTIMENT_FEATURES = ["avg_sentiment", "sentiment_std", "news_count"]
ATTENTION_FEATURES = ["trends_zscore_30d", "trends_momentum_7d", "trends_spike"]
TARGET_COLUMNS = ["target_direction", "target_next_return"]


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def default_params() -> dict[str, Any]:
    return {
        "lookback": 60,
        "lstm_units": 32,
        "dense_units": 64,
        "dropout": 0.05,
        "recurrent_dropout": 0.20,
        "lr": 0.0002,
        "batch": 32,
    }


def load_params(best_meta: Path | None) -> dict[str, Any]:
    params = default_params()
    if best_meta is not None and best_meta.exists():
        meta = json.loads(best_meta.read_text(encoding="utf-8"))
        params.update(meta.get("params", {}))
    return params


def feature_sets() -> dict[str, list[str]]:
    conventional_market = MARKET_FEATURES + MACRO_FEATURES
    return {
        "market_only": conventional_market,
        "market_sentiment": conventional_market + SENTIMENT_FEATURES,
        "market_attention": conventional_market + ATTENTION_FEATURES,
        "full_model": conventional_market + SENTIMENT_FEATURES + ATTENTION_FEATURES,
    }


def write_subset(df: pd.DataFrame, name: str, columns: list[str], outdir: Path) -> Path:
    missing = [c for c in ["date", *columns, *TARGET_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {name}: {missing}")

    subset = df[["date", *columns, *TARGET_COLUMNS]].copy()
    subset = subset.replace([float("inf"), float("-inf")], pd.NA).dropna().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"Feature set {name} produced an empty dataset after dropna().")

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.csv"
    subset.to_csv(path, index=False)
    return path


def parse_report_summary(path: Path) -> dict[str, float | int | str | None]:
    data = json.loads(path.read_text(encoding="utf-8"))

    classification = data.get("classification", {})
    trading = data.get("trading", {})
    overall = data.get("overall", {})

    auc = classification.get("auc", overall.get("auc"))
    acc = classification.get("accuracy", classification.get("acc", overall.get("acc")))

    strategy = overall.get("strategy", {}) if isinstance(overall, dict) else {}
    sharpe = trading.get("sharpe", trading.get("strategy_sharpe", strategy.get("sharpe_long_only")))
    trade_rate = trading.get("trade_rate", strategy.get("trade_rate_long_only"))
    max_drawdown = trading.get("max_drawdown")
    mean_daily_return = trading.get("mean_daily_return", strategy.get("avg_daily_return_long_only"))
    num_trades = trading.get("num_trades")

    return {
        "auc": auc,
        "accuracy": acc,
        "sharpe": sharpe,
        "trade_rate": trade_rate,
        "max_drawdown": max_drawdown,
        "mean_daily_return": mean_daily_return,
        "num_trades": num_trades,
    }


def run_command(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    root = find_project_root()
    parser = argparse.ArgumentParser(description="Run LSTM feature-group ablations with walk-forward validation.")
    parser.add_argument("--dataset", type=Path, required=True, help="Clean model dataset CSV.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for LSTM ablation results.")
    parser.add_argument("--best-meta", type=Path, default=None, help="Random-search best/meta.json to reuse hyperparameters.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocesses.")
    parser.add_argument("--walk-script", type=Path, default=root / "src/thesis/model_training/optimalisation/walk_forward_lstm_direction_rocm.py")
    parser.add_argument("--report-script", type=Path, default=root / "src/thesis/eval/thesis_walkforward_report.py")
    parser.add_argument("--initial-train", type=int, default=700)
    parser.add_argument("--val-size", type=int, default=126)
    parser.add_argument("--test-horizon", type=int, default=63)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true", help="Rerun existing feature-set outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.outdir / "datasets"
    params = load_params(args.best_meta)

    print("=== LSTM FEATURE-GROUP ABLATION ===", flush=True)
    print("Dataset  :", args.dataset, flush=True)
    print("Outdir   :", args.outdir, flush=True)
    print("Best meta:", args.best_meta, flush=True)
    print("Params   :", params, flush=True)
    print("Epochs   :", args.epochs, flush=True)

    df = pd.read_csv(args.dataset)
    rows: list[dict[str, Any]] = []

    for name, cols in feature_sets().items():
        print(f"\n=== Feature set: {name} ({len(cols)} features) ===", flush=True)
        subset_path = write_subset(df, name, cols, dataset_dir)
        feature_out = args.outdir / name
        report_out = feature_out / "thesis_report_gross"
        report_summary = report_out / "report_summary.json"

        if args.force or not report_summary.exists():
            walk_cmd = [
                args.python,
                "-u",
                str(args.walk_script),
                "--data",
                str(subset_path),
                "--outdir",
                str(feature_out),
                "--lookback",
                str(params["lookback"]),
                "--initial_train",
                str(args.initial_train),
                "--val_size",
                str(args.val_size),
                "--test_horizon",
                str(args.test_horizon),
                "--step",
                str(args.step),
                "--epochs",
                str(args.epochs),
                "--batch",
                str(params["batch"]),
                "--lr",
                str(params["lr"]),
                "--lstm_units",
                str(params["lstm_units"]),
                "--dense_units",
                str(params.get("dense_units", 64)),
                "--dropout",
                str(params["dropout"]),
                "--recurrent_dropout",
                str(params["recurrent_dropout"]),
                "--auto_threshold",
            ]
            if args.gpu != "":
                walk_cmd += ["--gpu", str(args.gpu)]
            run_command(walk_cmd)

            report_cmd = [
                args.python,
                "-u",
                str(args.report_script),
                "--oos",
                str(feature_out / "walk_forward_oos_predictions.csv"),
                "--summary",
                str(feature_out / "walk_forward_summary.json"),
                "--outdir",
                str(report_out),
            ]
            run_command(report_cmd)
        else:
            print(f"Using existing report: {report_summary}", flush=True)

        metrics = parse_report_summary(report_summary)
        rows.append(
            {
                "feature_set": name,
                "feature_count": len(cols),
                "dataset": str(subset_path),
                "report": str(report_summary),
                **metrics,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = args.outdir / "lstm_feature_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    try:
        summary.to_markdown(args.outdir / "lstm_feature_ablation_summary.md", index=False)
    except Exception:
        (args.outdir / "lstm_feature_ablation_summary.md").write_text(summary.to_string(index=False), encoding="utf-8")

    print("\n=== LSTM ABLATION DONE ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("Saved:", summary_path, flush=True)


if __name__ == "__main__":
    main()
