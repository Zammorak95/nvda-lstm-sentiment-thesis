#!/usr/bin/env python3
"""
logistic_regression_benchmark.py

Reproducible logistic-regression benchmark for the NVDA thesis.

What this script does
---------------------
- Loads model_dataset_clean.csv
- Sorts observations chronologically
- Uses the same target as the LSTM: target_direction
- Uses train-only scaling to prevent leakage
- Runs expanding-window walk-forward evaluation
- Saves out-of-sample predictions
- Reports AUC, accuracy, confusion matrix, Sharpe ratio and trade rate

Default walk-forward settings match the thesis setup:
- initial_train = 700 trading days
- val_size      = 126 trading days
- test_horizon  = 63 trading days
- step          = 63 trading days

Example
-------
python logistic_regression_benchmark.py \
  --data /home/zammorak/thesis/data/model_feed/model_dataset_clean.csv \
  --outdir /home/zammorak/thesis/models/logistic_regression_benchmark \
  --threshold 0.5

If you want to overwrite/recreate the older thesis output file name:
python logistic_regression_benchmark.py \
  --data /home/zammorak/thesis/data/model_feed/model_dataset_clean.csv \
  --outdir /home/zammorak/thesis/models \
  --out_csv /home/zammorak/thesis/models/logistic_reduced_t0525_oos_predictions.csv \
  --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA = "/home/zammorak/thesis/data/model_feed/model_dataset_clean.csv"
DEFAULT_OUTDIR = "/home/zammorak/thesis/models/logistic_regression_benchmark"


def annualized_sharpe(daily_returns: np.ndarray, periods: int = 252) -> float:
    """Annualized Sharpe ratio using daily returns."""
    daily_returns = np.asarray(daily_returns, dtype=float)
    std = daily_returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(daily_returns.mean() / std * np.sqrt(periods))


def best_threshold_youden(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Select threshold by maximizing Youden's J = TPR - FPR on validation data."""
    best_t, best_j = 0.5, -1e9

    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob >= t).astype(int)
        labels = np.array([0, 1])
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=labels).ravel()

        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        j = float(tpr - fpr)

        if j > best_j:
            best_j = j
            best_t = float(t)

    return best_t, best_j


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and prepare thesis model dataset."""
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the dataset.")
    if "target_direction" not in df.columns:
        raise ValueError("Expected 'target_direction' in the dataset.")
    if "target_next_return" not in df.columns:
        raise ValueError("Expected 'target_next_return' in the dataset for trading metrics.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)

    drop_cols = {"date", "target_next_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_df = df[feature_cols].select_dtypes(include=[np.number]).copy()
    if X_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")

    # Defensive cleanup only. The clean dataset should already contain no missing values.
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    if X_df.isna().any().any():
        X_df = X_df.fillna(X_df.median(numeric_only=True))

    X = X_df.to_numpy(dtype=float)
    y = df["target_direction"].astype(int).to_numpy()

    return df, X_df, X, y


def run_walk_forward_logistic(
    df: pd.DataFrame,
    X_df: pd.DataFrame,
    X_all: np.ndarray,
    y_all: np.ndarray,
    *,
    initial_train: int,
    val_size: int,
    test_horizon: int,
    step: int,
    threshold: float,
    auto_threshold: bool,
    class_weight: Optional[str],
    max_iter: int,
    C: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Expanding-window walk-forward benchmark.

    The validation window is used only when --auto_threshold is enabled.
    The scaler is always fitted on training data only.
    """
    n = len(df)

    if initial_train + test_horizon > n:
        raise ValueError(
            f"Not enough rows: initial_train({initial_train}) + "
            f"test_horizon({test_horizon}) > n({n})."
        )
    if initial_train - val_size <= 0:
        raise ValueError("val_size is too large for the initial_train window.")

    records: list[pd.DataFrame] = []
    fold_summaries: list[dict] = []

    fold = 0
    train_end = initial_train

    while True:
        test_end = train_end + test_horizon
        if test_end > n:
            break

        fold += 1
        val_start = train_end - val_size

        # Fit scaler on training only. Validation and test are transformed using training-fitted scaler.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_all[:val_start])
        y_train = y_all[:val_start]

        X_val = scaler.transform(X_all[val_start:train_end])
        y_val = y_all[val_start:train_end]

        X_test = scaler.transform(X_all[train_end:test_end])
        y_test = y_all[train_end:test_end]

        model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=42,
        )
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan")

        if auto_threshold:
            fold_threshold, youden_j = best_threshold_youden(y_val, val_prob)
        else:
            fold_threshold, youden_j = float(threshold), None

        val_pred = (val_prob >= fold_threshold).astype(int)
        val_acc = float(accuracy_score(y_val, val_pred))

        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= fold_threshold).astype(int)

        test_auc = float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else float("nan")
        test_acc = float(accuracy_score(y_test, test_pred))
        test_cm = confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist()

        out = pd.DataFrame({
            "fold": fold,
            "date": df["date"].iloc[train_end:test_end].to_numpy(),
            "y_true": y_test,
            "y_prob_up": test_prob,
            "y_pred": test_pred,
            "threshold": fold_threshold,
            "target_next_return": df["target_next_return"].iloc[train_end:test_end].to_numpy(),
        })
        records.append(out)

        fold_summaries.append({
            "fold": fold,
            "train_start_date": str(df["date"].iloc[0].date()),
            "train_end_date": str(df["date"].iloc[val_start - 1].date()),
            "validation_start_date": str(df["date"].iloc[val_start].date()),
            "validation_end_date": str(df["date"].iloc[train_end - 1].date()),
            "test_start_date": str(df["date"].iloc[train_end].date()),
            "test_end_date": str(df["date"].iloc[test_end - 1].date()),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "val_auc": val_auc,
            "val_acc": val_acc,
            "test_auc": test_auc,
            "test_acc": test_acc,
            "test_cm": test_cm,
            "threshold": float(fold_threshold),
            "youden_j": None if youden_j is None else float(youden_j),
            "feature_count": int(X_df.shape[1]),
            "features": X_df.columns.tolist(),
        })

        print(
            f"[fold {fold:02d}] "
            f"val_auc={val_auc:.4f} val_acc={val_acc:.4f} "
            f"thr={fold_threshold:.3f} | "
            f"test_auc={test_auc:.4f} test_acc={test_acc:.4f} cm={test_cm}"
        )

        train_end += step

    if not records:
        raise RuntimeError("No folds were produced. Check initial_train/test_horizon/step settings.")

    oos = pd.concat(records, ignore_index=True).sort_values("date").reset_index(drop=True)
    return oos, fold_summaries


def summarize_oos(oos: pd.DataFrame) -> dict:
    """Compute aggregate out-of-sample classification and trading metrics."""
    y_true = oos["y_true"].astype(int).to_numpy()
    y_prob = oos["y_prob_up"].astype(float).to_numpy()
    y_pred = oos["y_pred"].astype(int).to_numpy()

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    strategy_returns = oos["target_next_return"].astype(float).to_numpy() * y_pred
    trade_rate = float(np.mean(y_pred))

    return {
        "auc": auc,
        "accuracy": acc,
        "confusion_matrix": cm,
        "sharpe_long_only": annualized_sharpe(strategy_returns),
        "avg_daily_return_long_only": float(np.mean(strategy_returns)),
        "trade_rate_long_only": trade_rate,
        "rows": int(len(oos)),
        "mean_probability": float(np.mean(y_prob)),
        "mean_threshold": float(np.mean(oos["threshold"].astype(float).to_numpy())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible logistic-regression walk-forward benchmark for the NVDA thesis."
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to model_dataset_clean.csv.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional exact output CSV path. If omitted, saves to outdir/logistic_regression_walkforward_oos_predictions.csv.",
    )

    parser.add_argument("--initial_train", type=int, default=700)
    parser.add_argument("--val_size", type=int, default=126)
    parser.add_argument("--test_horizon", type=int, default=63)
    parser.add_argument("--step", type=int, default=63)

    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold if --auto_threshold is not used.")
    parser.add_argument("--auto_threshold", action="store_true", help="Select threshold per fold on validation using Youden's J.")
    parser.add_argument(
        "--class_weight",
        choices=["none", "balanced"],
        default="none",
        help="Use 'balanced' if you want class-weighted logistic regression.",
    )
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    class_weight = None if args.class_weight == "none" else "balanced"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv) if args.out_csv else outdir / "logistic_regression_walkforward_oos_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df, X_df, X_all, y_all = load_dataset(args.data)

    print("=== LOGISTIC REGRESSION WALK-FORWARD BENCHMARK ===")
    print(f"Data: {args.data}")
    print(f"Rows: {len(df)} | Features: {X_df.shape[1]}")
    print(f"Date range: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    print(f"Class balance: up={int(y_all.sum())}, down={int(len(y_all) - y_all.sum())}")
    print(
        f"Walk-forward: initial_train={args.initial_train}, val_size={args.val_size}, "
        f"test_horizon={args.test_horizon}, step={args.step}"
    )
    print(f"Threshold mode: {'Youden validation threshold' if args.auto_threshold else f'fixed {args.threshold}'}")
    print(f"Class weight: {class_weight}")

    oos, folds = run_walk_forward_logistic(
        df,
        X_df,
        X_all,
        y_all,
        initial_train=args.initial_train,
        val_size=args.val_size,
        test_horizon=args.test_horizon,
        step=args.step,
        threshold=args.threshold,
        auto_threshold=args.auto_threshold,
        class_weight=class_weight,
        max_iter=args.max_iter,
        C=args.C,
    )

    summary = {
        "model": "LogisticRegression",
        "data": str(args.data),
        "output_csv": str(out_csv),
        "settings": {
            "initial_train": args.initial_train,
            "val_size": args.val_size,
            "test_horizon": args.test_horizon,
            "step": args.step,
            "threshold": args.threshold,
            "auto_threshold": bool(args.auto_threshold),
            "class_weight": class_weight,
            "max_iter": args.max_iter,
            "C": args.C,
            "scaling": "StandardScaler fitted on training window only",
        },
        "feature_columns": X_df.columns.tolist(),
        "overall": summarize_oos(oos),
        "folds": folds,
    }

    oos.to_csv(out_csv, index=False)

    summary_path = outdir / "logistic_regression_walkforward_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== OVERALL OOS RESULTS ===")
    for key, value in summary["overall"].items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(f"Predictions: {out_csv}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
