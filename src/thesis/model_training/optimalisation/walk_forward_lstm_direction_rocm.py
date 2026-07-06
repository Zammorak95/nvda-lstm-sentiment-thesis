#!/usr/bin/env python3
"""
Walk-forward backtest for LSTM direction classifier (ROCm-safe).

If you run this and see *no output*, it usually means:
- you're not running this file (wrong path), OR
- the file on disk doesn't include the __main__ block, OR
- your stdout is buffered (use: python -u script.py)

This version prints immediately at startup and at each fold.

Usage:
  source /home/zammorak/thesis/.venv/bin/activate
  HIP_VISIBLE_DEVICES=0 python -u walk_forward_lstm_direction_rocm.py \
    --data /home/zammorak/thesis/data/model_feed/model_dataset_clean.csv \
    --outdir /home/zammorak/thesis/models/walk_forward_direction \
    --lookback 60 --initial_train 700 --val_size 126 --test_horizon 63 --step 63 --auto_threshold
"""

import os
import json
import time
import argparse
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

def _set_gpu(gpu_index: Optional[str]) -> None:
    if gpu_index is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_index)

# TensorFlow import after HIP_VISIBLE_DEVICES is set in main()
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import joblib


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences; label is y[t] at the end of each sequence."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


def build_model(
    lookback: int,
    n_features: int,
    lr: float,
    lstm_units: int,
    dropout: float,
    rec_dropout: float,
    dense_units: int,
) -> tf.keras.Model:
    """ROCm-safe LSTM (non-fused path)."""
    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(
            lstm_units,
            dropout=dropout,
            recurrent_dropout=rec_dropout,
            implementation=1,  # avoid fused CudnnRNNV3 path
        ),
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def compute_class_weight(y_train: np.ndarray) -> Optional[Dict[int, float]]:
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if pos == 0 or neg == 0:
        return None
    total = pos + neg
    return {0: float(total / (2 * neg)), 1: float(total / (2 * pos))}


def best_threshold_youden(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """Pick threshold using Youden's J (TPR - FPR)."""
    best_t, best_j = 0.5, -1e9
    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/zammorak/thesis/data/model_feed/model_dataset_clean.csv")
    ap.add_argument("--outdir", default="/home/zammorak/thesis/models/walk_forward_direction")

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

    args = ap.parse_args()

    _set_gpu(args.gpu)

    os.makedirs(args.outdir, exist_ok=True)

    print("=== WALK-FORWARD LSTM (ROCm-safe) ===", flush=True)
    print("Data   :", args.data, flush=True)
    print("Outdir :", args.outdir, flush=True)
    print("HIP_VISIBLE_DEVICES:", os.environ.get("HIP_VISIBLE_DEVICES", None), flush=True)
    print("TF GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

    # Load data
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

    X_all = X_df.to_numpy()
    y_all = df["target_direction"].astype(int).to_numpy()

    n = len(df)
    print(f"Rows={n} | Features={X_df.shape[1]} | Range={df['date'].iloc[0].date()}→{df['date'].iloc[-1].date()}",
          flush=True)

    if args.initial_train + args.test_horizon + args.lookback >= n:
        raise ValueError(
            f"Not enough rows for settings: initial_train({args.initial_train}) + "
            f"test_horizon({args.test_horizon}) + lookback({args.lookback}) >= n({n})."
        )

    oos_records: List[pd.DataFrame] = []
    fold_summaries: List[Dict[str, Any]] = []

    fold = 0
    train_end = args.initial_train

    while True:
        test_end = train_end + args.test_horizon
        if test_end > n:
            break

        fold += 1
        fold_dir = os.path.join(args.outdir, f"fold_{fold:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        val_start = train_end - args.val_size
        if val_start <= args.lookback:
            raise ValueError("val_size too large for the current fold given lookback.")

        # Scale with TRAIN-FIT ONLY (exclude val+test)
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_all[:val_start])
        X_val_part = scaler.transform(X_all[val_start:train_end])
        X_test_part = scaler.transform(X_all[train_end:test_end])

        X_scaled_concat = np.vstack([X_train_fit, X_val_part, X_test_part]).astype(np.float32)
        y_concat = np.concatenate([y_all[:val_start], y_all[val_start:train_end], y_all[train_end:test_end]])

        # Build sequences for this fold chunk
        X_seq, y_seq = make_sequences(X_scaled_concat, y_concat, args.lookback)

        seq_val_start = val_start - args.lookback
        seq_test_start = train_end - args.lookback

        X_train, y_train = X_seq[:seq_val_start], y_seq[:seq_val_start]
        X_val, y_val = X_seq[seq_val_start:seq_test_start], y_seq[seq_val_start:seq_test_start]
        X_test, y_test = X_seq[seq_test_start:], y_seq[seq_test_start:]

        print(
            f"\n[fold {fold:02d}] "
            f"train_end={train_end} ({df['date'].iloc[train_end-1].date()}) | "
            f"test={df['date'].iloc[train_end].date()}→{df['date'].iloc[test_end-1].date()} | "
            f"shapes train={X_train.shape} val={X_val.shape} test={X_test.shape}",
            flush=True
        )

        model = build_model(
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
        class_weight = compute_class_weight(y_train)

        t0 = time.time()
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch,
            verbose=1,  # <-- SHOW PROGRESS
            callbacks=cb,
            class_weight=class_weight
        )
        train_seconds = time.time() - t0

        # validation threshold
        val_prob = model(X_val, training=False).numpy().reshape(-1)
        val_auc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan")

        if args.auto_threshold:
            thr, j = best_threshold_youden(y_val, val_prob)
        else:
            thr, j = float(args.threshold), None

        val_pred = (val_prob >= thr).astype(int)
        val_acc = float(accuracy_score(y_val, val_pred))

        # OOS test
        test_prob = model(X_test, training=False).numpy().reshape(-1)
        test_auc = float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else float("nan")
        test_pred = (test_prob >= thr).astype(int)
        test_acc = float(accuracy_score(y_test, test_pred))
        test_cm = confusion_matrix(y_test, test_pred).tolist()

        # save artifacts
        model.save(os.path.join(fold_dir, "model.keras"))
        joblib.dump(scaler, os.path.join(fold_dir, "scaler.joblib"))

        meta = {
            "fold": fold,
            "train_end_date": str(df["date"].iloc[train_end-1].date()),
            "test_start_date": str(df["date"].iloc[train_end].date()),
            "test_end_date": str(df["date"].iloc[test_end-1].date()),
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
            }
        }
        with open(os.path.join(fold_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        fold_summaries.append(meta)

        # align dates with y_test length
        test_dates = df["date"].iloc[train_end:test_end].reset_index(drop=True)
        if len(test_dates) != len(y_test):
            test_dates = test_dates.iloc[-len(y_test):].reset_index(drop=True)

        out = pd.DataFrame({
            "fold": fold,
            "date": test_dates,
            "y_true": y_test,
            "y_prob_up": test_prob,
            "y_pred": test_pred,
            "threshold": thr,
        })

        if "target_next_return" in df.columns:
            next_ret = df["target_next_return"].iloc[train_end:test_end].reset_index(drop=True)
            if len(next_ret) != len(y_test):
                next_ret = next_ret.iloc[-len(y_test):].reset_index(drop=True)
            out["target_next_return"] = next_ret.values

        oos_records.append(out)

        print(
            f"[fold {fold:02d} DONE] VAL auc={val_auc:.4f} acc={val_acc:.4f} thr={thr:.2f} | "
            f"TEST auc={test_auc:.4f} acc={test_acc:.4f} cm={test_cm}",
            flush=True
        )

        # move forward
        train_end += args.step
        tf.keras.backend.clear_session()

    # aggregate OOS
    oos = pd.concat(oos_records, ignore_index=True).sort_values("date").reset_index(drop=True)
    oos_path = os.path.join(args.outdir, "walk_forward_oos_predictions.csv")
    oos.to_csv(oos_path, index=False)

    overall_auc = float(roc_auc_score(oos["y_true"], oos["y_prob_up"])) if len(np.unique(oos["y_true"])) > 1 else float("nan")
    overall_acc = float(accuracy_score(oos["y_true"], oos["y_pred"]))
    overall_cm = confusion_matrix(oos["y_true"], oos["y_pred"]).tolist()

    strat = {}
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
        "saved_predictions": oos_path,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", None),
        "feature_count": int(X_df.shape[1]),
    }
    with open(os.path.join(args.outdir, "walk_forward_summary.json"), "w") as f:
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
