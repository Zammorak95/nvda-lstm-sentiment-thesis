#!/usr/bin/env python3
"""
Walk-forward backtest for a ROCm-safe LSTM direction classifier.

This script keeps the original thesis workflow, but improves two things:
1. --gpu is applied before TensorFlow is imported.
2. Default paths are based on thesis.paths/environment variables, so the script is usable on Windows/Linux.

Usage examples:
  python -u walk_forward_lstm_direction_rocm.py --auto_threshold --gpu 0
  python -u walk_forward_lstm_direction_rocm.py --data data/model_feed/model_dataset_clean.csv --outdir artifacts/models/walk_forward_direction
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[4]


def _ensure_src_on_path() -> None:
    root = _find_project_root()
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

try:
    from thesis.paths import DATA_DIR, MODELS
except Exception:
    _ROOT = _find_project_root()
    DATA_DIR = Path(os.getenv("THESIS_DATA_DIR", _ROOT / "data")).resolve()
    MODELS = Path(os.getenv("THESIS_MODELS_DIR", _ROOT / "artifacts" / "models")).resolve()


DEFAULT_DATA = DATA_DIR / "model_feed" / "model_dataset_clean.csv"
DEFAULT_OUTDIR = Path(os.getenv("THESIS_MODELS_DIR", MODELS)) / "walk_forward_direction"


def _set_gpu(gpu_index: str | None) -> None:
    if gpu_index is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_index)


def _import_tensorflow(gpu_index: str | None):
    """Import TensorFlow only after HIP_VISIBLE_DEVICES has been set."""
    _set_gpu(gpu_index)
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow.keras import callbacks, layers, models  # noqa: PLC0415

    return tf, layers, models, callbacks


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    Xs: list[np.ndarray] = []
    ys: list[int] = []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(int(y[i]))
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


def build_model(
    tf: Any,
    layers: Any,
    models: Any,
    lookback: int,
    n_features: int,
    lr: float,
    lstm_units: int,
    dropout: float,
    rec_dropout: float,
    dense_units: int,
):
    model = models.Sequential(
        [
            layers.Input(shape=(lookback, n_features)),
            layers.LSTM(
                lstm_units,
                dropout=dropout,
                recurrent_dropout=rec_dropout,
                implementation=1,  # ROCm-safe non-fused path
            ),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def compute_class_weight(y_train: np.ndarray) -> dict[int, float] | None:
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if pos == 0 or neg == 0:
        return None
    total = pos + neg
    return {0: float(total / (2 * neg)), 1: float(total / (2 * pos))}


def best_threshold_youden(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_t, best_j = 0.5, -1e9
    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        j = float(tpr - fpr)
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, best_j


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    daily_returns = np.asarray(daily_returns, dtype=float)
    std = daily_returns.std()
    if std == 0:
        return float("nan")
    return float(daily_returns.mean() / std * np.sqrt(252))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--initial_train", type=int, default=700)
    ap.add_argument("--val_size", type=int, default=126)
    ap.add_argument("--test_horizon", type=int, default=63)
    ap.add_argument("--step", type=int, default=63)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lstm_units", type=int, default=32)
    ap.add_argument("--dense_units", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--recurrent_dropout", type=float, default=0.2)
    ap.add_argument("--auto_threshold", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gpu", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tf, layers, models, callbacks = _import_tensorflow(args.gpu)

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=== WALK-FORWARD LSTM (ROCm-safe) ===", flush=True)
    print("Data   :", args.data, flush=True)
    print("Outdir :", args.outdir, flush=True)
    print("HIP_VISIBLE_DEVICES:", os.environ.get("HIP_VISIBLE_DEVICES", None), flush=True)
    print("TF GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

    df = pd.read_csv(args.data)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in dataset.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "target_direction" not in df.columns:
        raise ValueError("target_direction missing in dataset.")

    drop_cols = {"date", "target_next_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].select_dtypes(include=[np.number]).copy()
    if X_df.shape[1] == 0:
        raise ValueError("No numeric features found after dropping meta/targets.")

    X_all = X_df.to_numpy(dtype=np.float32)
    y_all = df["target_direction"].astype(int).to_numpy()

    n = len(df)
    print(
        f"Rows={n} | Features={X_df.shape[1]} | Range={df['date'].iloc[0].date()}→{df['date'].iloc[-1].date()}",
        flush=True,
    )

    if args.initial_train + args.test_horizon + args.lookback >= n:
        raise ValueError(
            f"Not enough rows for settings: initial_train({args.initial_train}) + "
            f"test_horizon({args.test_horizon}) + lookback({args.lookback}) >= n({n})."
        )

    oos_records: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []

    fold = 0
    train_end = args.initial_train

    while True:
        test_end = train_end + args.test_horizon
        if test_end > n:
            break

        fold += 1
        fold_dir = args.outdir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_start = train_end - args.val_size
        if val_start <= args.lookback:
            raise ValueError("val_size too large for the current fold given lookback.")

        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_all[:val_start])
        X_val_part = scaler.transform(X_all[val_start:train_end])
        X_test_part = scaler.transform(X_all[train_end:test_end])

        X_scaled_concat = np.vstack([X_train_fit, X_val_part, X_test_part]).astype(np.float32)
        y_concat = np.concatenate([y_all[:val_start], y_all[val_start:train_end], y_all[train_end:test_end]])

        X_seq, y_seq = make_sequences(X_scaled_concat, y_concat, args.lookback)
        seq_val_start = val_start - args.lookback
        seq_test_start = train_end - args.lookback

        X_train, y_train = X_seq[:seq_val_start], y_seq[:seq_val_start]
        X_val, y_val = X_seq[seq_val_start:seq_test_start], y_seq[seq_val_start:seq_test_start]
        X_test, y_test = X_seq[seq_test_start:], y_seq[seq_test_start:]

        print(
            f"\n[fold {fold:02d}] "
            f"train_end={train_end} ({df['date'].iloc[train_end - 1].date()}) | "
            f"test={df['date'].iloc[train_end].date()}→{df['date'].iloc[test_end - 1].date()} | "
            f"shapes train={X_train.shape} val={X_val.shape} test={X_test.shape}",
            flush=True,
        )

        model = build_model(
            tf=tf,
            layers=layers,
            models=models,
            lookback=args.lookback,
            n_features=X_train.shape[-1],
            lr=args.lr,
            lstm_units=args.lstm_units,
            dropout=args.dropout,
            rec_dropout=args.recurrent_dropout,
            dense_units=args.dense_units,
        )

        cb = [
            callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=3, factor=0.5, min_lr=1e-5),
        ]

        t0 = time.time()
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch,
            verbose=1,
            callbacks=cb,
            class_weight=compute_class_weight(y_train),
        )
        train_seconds = time.time() - t0

        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        val_auc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan")

        if args.auto_threshold:
            thr, j = best_threshold_youden(y_val, val_prob)
        else:
            thr, j = float(args.threshold), None

        val_pred = (val_prob >= thr).astype(int)
        val_acc = float(accuracy_score(y_val, val_pred))

        test_prob = model.predict(X_test, verbose=0).reshape(-1)
        test_auc = float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else float("nan")
        test_pred = (test_prob >= thr).astype(int)
        test_acc = float(accuracy_score(y_test, test_pred))
        test_cm = confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist()

        model.save(fold_dir / "model.keras")
        joblib.dump(scaler, fold_dir / "scaler.joblib")

        meta = {
            "fold": fold,
            "train_end_date": str(df["date"].iloc[train_end - 1].date()),
            "test_start_date": str(df["date"].iloc[train_end].date()),
            "test_end_date": str(df["date"].iloc[test_end - 1].date()),
            "val_auc": val_auc,
            "val_acc": val_acc,
            "threshold": thr,
            "youden_j": j,
            "test_auc": test_auc,
            "test_acc": test_acc,
            "test_cm": test_cm,
            "train_seconds": train_seconds,
            "params": {
                "lookback": args.lookback,
                "lstm_units": args.lstm_units,
                "dense_units": args.dense_units,
                "lr": args.lr,
                "dropout": args.dropout,
                "recurrent_dropout": args.recurrent_dropout,
                "batch": args.batch,
                "epochs": args.epochs,
            },
        }
        with open(fold_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        fold_summaries.append(meta)

        test_dates = df["date"].iloc[train_end:test_end].reset_index(drop=True)
        if len(test_dates) != len(y_test):
            test_dates = test_dates.iloc[-len(y_test):].reset_index(drop=True)

        out = pd.DataFrame(
            {
                "fold": fold,
                "date": test_dates,
                "y_true": y_test,
                "y_prob_up": test_prob,
                "y_pred": test_pred,
                "threshold": thr,
            }
        )

        if "target_next_return" in df.columns:
            next_ret = df["target_next_return"].iloc[train_end:test_end].reset_index(drop=True)
            if len(next_ret) != len(y_test):
                next_ret = next_ret.iloc[-len(y_test):].reset_index(drop=True)
            out["target_next_return"] = next_ret.values

        oos_records.append(out)

        print(
            f"[fold {fold:02d} DONE] VAL auc={val_auc:.4f} acc={val_acc:.4f} thr={thr:.2f} | "
            f"TEST auc={test_auc:.4f} acc={test_acc:.4f} cm={test_cm}",
            flush=True,
        )

        train_end += args.step
        tf.keras.backend.clear_session()

    if not oos_records:
        raise RuntimeError("No walk-forward folds were produced. Check initial_train/test_horizon/step settings.")

    oos = pd.concat(oos_records, ignore_index=True).sort_values("date").reset_index(drop=True)
    oos_path = args.outdir / "walk_forward_oos_predictions.csv"
    oos.to_csv(oos_path, index=False)

    overall_auc = float(roc_auc_score(oos["y_true"], oos["y_prob_up"])) if len(np.unique(oos["y_true"])) > 1 else float("nan")
    overall_acc = float(accuracy_score(oos["y_true"], oos["y_pred"]))
    overall_cm = confusion_matrix(oos["y_true"], oos["y_pred"], labels=[0, 1]).tolist()

    strat: dict[str, float] = {}
    if "target_next_return" in oos.columns:
        strategy_returns = oos["target_next_return"].values * oos["y_pred"].values
        strat = {
            "sharpe_long_only": sharpe_ratio(strategy_returns),
            "avg_daily_return_long_only": float(np.mean(strategy_returns)),
            "trade_rate_long_only": float(np.mean(oos["y_pred"].values)),
        }

    summary = {
        "overall": {"auc": overall_auc, "acc": overall_acc, "cm": overall_cm, "strategy": strat},
        "folds": fold_summaries,
        "saved_predictions": str(oos_path),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", None),
        "feature_count": int(X_df.shape[1]),
        "data": str(args.data),
    }
    with open(args.outdir / "walk_forward_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DONE ===", flush=True)
    print("OOS predictions:", oos_path, flush=True)
    print("Overall OOS AUC:", overall_auc, flush=True)
    print("Overall OOS ACC:", overall_acc, flush=True)
    if strat:
        print("OOS Sharpe (long-only):", strat["sharpe_long_only"], flush=True)
        print("Trade rate:", strat["trade_rate_long_only"], flush=True)


if __name__ == "__main__":
    main()
