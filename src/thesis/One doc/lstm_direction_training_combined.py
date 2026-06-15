#!/usr/bin/env python3
"""
lstm_direction_training_combined.py

Unified ROCm-safe LSTM direction-classifier utility.

Combines:
- random_search_lstm_direction.py
- random_search_lstm_direction_v2.py
- walk_forward_lstm_direction_rocm.py

Subcommands
-----------
1) random-search
   Chronological train/validation/test split, train-only scaling, random
   hyperparameter search, incremental CSV logging, best model/scaler/meta save,
   and final test evaluation of the best trial.

2) walk-forward
   Expanding-window walk-forward backtest with validation threshold selection,
   per-fold artifacts, out-of-sample prediction CSV, and summary JSON.

ROCm note
---------
The model uses LSTM(implementation=1, recurrent_dropout>0) to avoid the fused
MIOpen/CuDNN path that can fail on ROCm with packed-input errors.

Examples
--------
Random search:
  HIP_VISIBLE_DEVICES=0 python -u lstm_direction_training_combined.py random-search \
    --data /home/zammorak/thesis/data/model_feed/model_dataset_clean.csv \
    --outdir /home/zammorak/thesis/models/random_search_direction_v2 \
    --trials 50 --auto_threshold

Walk-forward:
  HIP_VISIBLE_DEVICES=0 python -u lstm_direction_training_combined.py walk-forward \
    --data /home/zammorak/thesis/data/model_feed/model_dataset_clean.csv \
    --outdir /home/zammorak/thesis/models/walk_forward_direction \
    --lookback 60 --initial_train 700 --val_size 126 --test_horizon 63 --step 63 \
    --auto_threshold
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


def _set_gpu(gpu_index: Optional[str]) -> None:
    """Set ROCm GPU visibility before TensorFlow does meaningful GPU work."""
    if gpu_index is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_index)


# TensorFlow is imported at module load for normal CLI usage. If --gpu is used,
# HIP_VISIBLE_DEVICES is still set before GPU listing/training in each command.
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA = "/home/zammorak/thesis/data/model_feed/model_dataset_clean.csv"


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create lookback sequences ending at each time t with label y[t]."""
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
    dense_units: int = 32,
) -> tf.keras.Model:
    """Build a ROCm-safe LSTM binary classifier."""
    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(
            lstm_units,
            dropout=dropout,
            recurrent_dropout=rec_dropout,
            implementation=1,
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
    """Balanced class weights, or None if one class is absent."""
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if pos == 0 or neg == 0:
        return None
    total = pos + neg
    return {0: float(total / (2 * neg)), 1: float(total / (2 * pos))}


def best_threshold_youden(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """Choose threshold by maximizing Youden's J = TPR - FPR."""
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


def eval_auc_acc(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Tuple[float, float, List[List[int]]]:
    pred = (prob >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    acc = float(accuracy_score(y_true, pred))
    cm = confusion_matrix(y_true, pred).tolist()
    return auc, acc, cm


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    daily_returns = np.asarray(daily_returns, dtype=float)
    std = daily_returns.std()
    if std == 0:
        return float("nan")
    return float(daily_returns.mean() / std * np.sqrt(252))


def load_direction_dataset(path: str, require_date: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    elif require_date:
        raise ValueError("Expected a 'date' column in dataset.")

    if "target_direction" not in df.columns:
        raise ValueError("target_direction missing in dataset.")

    drop_cols = {"date", "target_next_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].select_dtypes(include=[np.number]).copy()
    if X_df.shape[1] == 0:
        raise ValueError("No numeric features found after dropping meta/targets.")

    X_all = X_df.to_numpy()
    y_all = df["target_direction"].astype(int).to_numpy()
    return df, X_df, X_all, y_all


@dataclass
class TrialResult:
    trial: int
    val_auc: float
    val_acc: float
    threshold: float
    youden_j: Optional[float]
    best_epoch: int
    best_val_auc_in_training: float
    best_val_loss_in_training: float
    seconds: float
    lookback: int
    lstm_units: int
    dropout: float
    recurrent_dropout: float
    lr: float
    batch: int
    dense_units: int


def sample_params(rng: random.Random, space: str = "v2") -> Dict[str, Any]:
    """Sample hyperparameters. `v1` matches the earlier script; `v2` adds dense_units."""
    params: Dict[str, Any] = {
        "lookback": rng.choice([45, 60, 75, 90, 105]),
        "lstm_units": rng.choice([16, 32, 48, 64]),
        "dropout": rng.choice([0.02, 0.05, 0.08, 0.10]),
        "recurrent_dropout": rng.choice([0.15, 0.20, 0.25]),
        "lr": rng.choice([1e-4, 2e-4, 3e-4]),
        "batch": rng.choice([16, 32, 64]),
    }
    params["dense_units"] = rng.choice([16, 32, 64]) if space == "v2" else 32
    return params


def run_random_search(args: argparse.Namespace) -> None:
    _set_gpu(args.gpu)
    os.makedirs(args.outdir, exist_ok=True)

    results_csv = os.path.join(args.outdir, "random_search_results.csv")
    best_dir = os.path.join(args.outdir, "best")
    os.makedirs(best_dir, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df, X_df, X_all, y_all = load_direction_dataset(args.data)
    n = len(df)
    if args.test_size + args.val_size + 10 >= n:
        raise ValueError("Not enough rows for chosen val/test sizes.")

    test_start = n - args.test_size
    val_start = test_start - args.val_size

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_all[:val_start])
    X_val_scaled = scaler.transform(X_all[val_start:test_start])
    X_test_scaled = scaler.transform(X_all[test_start:])
    X_scaled_full = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled]).astype(np.float32)

    best_val_auc = -1.0
    best_payload: Optional[Dict[str, Any]] = None
    trial_rows: List[Dict[str, Any]] = []

    print("=== RANDOM SEARCH LSTM (ROCm-safe) ===", flush=True)
    print("TF GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    print("Dataset rows:", n, "| features:", X_df.shape[1], flush=True)
    print("Split sizes -> train:", val_start, "val:", args.val_size, "test:", args.test_size, flush=True)

    for trial in range(1, args.trials + 1):
        params = sample_params(rng, args.search_space)
        lookback = int(params["lookback"])

        if args.test_size + args.val_size + lookback >= n:
            continue

        X_seq, y_seq = make_sequences(X_scaled_full, y_all, lookback)
        seq_val_start = val_start - lookback
        seq_test_start = test_start - lookback

        X_train, y_train = X_seq[:seq_val_start], y_seq[:seq_val_start]
        X_val, y_val = X_seq[seq_val_start:seq_test_start], y_seq[seq_val_start:seq_test_start]

        model = build_model(
            lookback=lookback,
            n_features=X_train.shape[-1],
            lr=float(params["lr"]),
            lstm_units=int(params["lstm_units"]),
            dropout=float(params["dropout"]),
            rec_dropout=float(params["recurrent_dropout"]),
            dense_units=int(params["dense_units"]),
        )

        cb = [
            callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=args.patience, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", patience=max(2, args.patience // 2), factor=0.5, min_lr=1e-5
            ),
        ]

        t0 = time.time()
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.max_epochs,
            batch_size=int(params["batch"]),
            verbose=args.verbose,
            callbacks=cb,
            class_weight=compute_class_weight(y_train),
        )
        seconds = time.time() - t0

        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        val_auc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan")

        if args.auto_threshold:
            thr, j = best_threshold_youden(y_val, val_prob)
        else:
            thr, j = 0.5, None

        _, val_acc, val_cm = eval_auc_acc(y_val, val_prob, thr)
        best_val_auc_in_training = float(np.max(hist.history.get("val_auc", [np.nan])))
        best_val_loss_in_training = float(np.min(hist.history.get("val_loss", [np.nan])))
        best_epoch = int(np.argmax(hist.history.get("val_auc", [val_auc])) + 1)

        row = TrialResult(
            trial=trial,
            val_auc=val_auc,
            val_acc=val_acc,
            threshold=float(thr),
            youden_j=(float(j) if j is not None else None),
            best_epoch=best_epoch,
            best_val_auc_in_training=best_val_auc_in_training,
            best_val_loss_in_training=best_val_loss_in_training,
            seconds=float(seconds),
            lookback=lookback,
            lstm_units=int(params["lstm_units"]),
            dropout=float(params["dropout"]),
            recurrent_dropout=float(params["recurrent_dropout"]),
            lr=float(params["lr"]),
            batch=int(params["batch"]),
            dense_units=int(params["dense_units"]),
        ).__dict__
        row["val_cm_tn_fp_fn_tp"] = val_cm
        trial_rows.append(row)
        pd.DataFrame(trial_rows).to_csv(results_csv, index=False)

        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_payload = {
                "best_trial": trial,
                "params": params,
                "val_auc": float(val_auc),
                "val_acc": float(val_acc),
                "threshold": float(thr),
                "youden_j": (float(j) if j is not None else None),
                "feature_cols": X_df.columns.tolist(),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", None),
            }
            model.save(os.path.join(best_dir, "model.keras"))
            joblib.dump(scaler, os.path.join(best_dir, "scaler.joblib"))
            with open(os.path.join(best_dir, "meta.json"), "w") as f:
                json.dump(best_payload, f, indent=2)

        if args.save_within > 0 and np.isfinite(val_auc) and best_val_auc > 0:
            if val_auc >= (best_val_auc - float(args.save_within)):
                close_dir = os.path.join(args.outdir, f"top_trial_{trial:03d}_auc_{val_auc:.4f}")
                os.makedirs(close_dir, exist_ok=True)
                model.save(os.path.join(close_dir, "model.keras"))

        print(
            f"[{trial:03d}/{args.trials}] val_auc={val_auc:.4f} val_acc={val_acc:.4f} "
            f"thr={thr:.2f} lb={lookback} units={params['lstm_units']} dense={params['dense_units']} "
            f"lr={params['lr']} drop={params['dropout']} rdrop={params['recurrent_dropout']} batch={params['batch']}",
            flush=True,
        )
        tf.keras.backend.clear_session()

    print("\nRandom search complete.", flush=True)
    print("Results CSV:", results_csv, flush=True)

    if best_payload is None:
        print("No valid trials produced a best model.", flush=True)
        return

    best_params = best_payload["params"]
    lb = int(best_params["lookback"])
    X_seq, y_seq = make_sequences(X_scaled_full, y_all, lb)
    seq_test_start = test_start - lb
    X_test = X_seq[seq_test_start:]
    y_test = y_seq[seq_test_start:]

    best_model = tf.keras.models.load_model(os.path.join(best_dir, "model.keras"))
    test_prob = best_model.predict(X_test, verbose=0).reshape(-1)
    thr = float(best_payload.get("threshold", 0.5))
    test_auc, test_acc, test_cm = eval_auc_acc(y_test, test_prob, thr)

    best_payload.update({
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
        "test_cm_tn_fp_fn_tp": test_cm,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(os.path.join(best_dir, "meta.json"), "w") as f:
        json.dump(best_payload, f, indent=2)

    print("\nBEST MODEL", flush=True)
    print("  val_auc:", best_payload["val_auc"], "| val_acc:", best_payload["val_acc"], "| thr:", thr, flush=True)
    print("  test_auc:", test_auc, "| test_acc:", test_acc, "| test_cm:", test_cm, flush=True)
    print("Best artifacts saved in:", best_dir, flush=True)


def run_walk_forward(args: argparse.Namespace) -> None:
    _set_gpu(args.gpu)
    os.makedirs(args.outdir, exist_ok=True)

    print("=== WALK-FORWARD LSTM (ROCm-safe) ===", flush=True)
    print("Data   :", args.data, flush=True)
    print("Outdir :", args.outdir, flush=True)
    print("HIP_VISIBLE_DEVICES:", os.environ.get("HIP_VISIBLE_DEVICES", None), flush=True)
    print("TF GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

    df, X_df, X_all, y_all = load_direction_dataset(args.data, require_date=True)
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
            f"\n[fold {fold:02d}] train_end={train_end} ({df['date'].iloc[train_end-1].date()}) | "
            f"test={df['date'].iloc[train_end].date()}→{df['date'].iloc[test_end-1].date()} | "
            f"shapes train={X_train.shape} val={X_val.shape} test={X_test.shape}",
            flush=True,
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
            callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=args.patience, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=max(2, args.patience // 2), factor=0.5, min_lr=1e-5),
        ]

        t0 = time.time()
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch,
            verbose=args.verbose,
            callbacks=cb,
            class_weight=compute_class_weight(y_train),
        )
        train_seconds = time.time() - t0

        val_prob = model(X_val, training=False).numpy().reshape(-1)
        val_auc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan")
        if args.auto_threshold:
            thr, j = best_threshold_youden(y_val, val_prob)
        else:
            thr, j = float(args.threshold), None
        val_pred = (val_prob >= thr).astype(int)
        val_acc = float(accuracy_score(y_val, val_pred))

        test_prob = model(X_test, training=False).numpy().reshape(-1)
        test_auc = float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else float("nan")
        test_pred = (test_prob >= thr).astype(int)
        test_acc = float(accuracy_score(y_test, test_pred))
        test_cm = confusion_matrix(y_test, test_pred).tolist()

        model.save(os.path.join(fold_dir, "model.keras"))
        joblib.dump(scaler, os.path.join(fold_dir, "scaler.joblib"))

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
        with open(os.path.join(fold_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        fold_summaries.append(meta)

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
            flush=True,
        )

        train_end += args.step
        tf.keras.backend.clear_session()

    if not oos_records:
        raise RuntimeError("No walk-forward folds were produced. Check initial_train/test_horizon/step settings.")

    oos = pd.concat(oos_records, ignore_index=True).sort_values("date").reset_index(drop=True)
    oos_path = os.path.join(args.outdir, "walk_forward_oos_predictions.csv")
    oos.to_csv(oos_path, index=False)

    overall_auc = float(roc_auc_score(oos["y_true"], oos["y_prob_up"])) if len(np.unique(oos["y_true"])) > 1 else float("nan")
    overall_acc = float(accuracy_score(oos["y_true"], oos["y_pred"]))
    overall_cm = confusion_matrix(oos["y_true"], oos["y_pred"]).tolist()

    strat: Dict[str, float] = {}
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified ROCm-safe LSTM direction training/backtesting CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    rs = sub.add_parser("random-search", help="Run chronological random hyperparameter search.")
    rs.add_argument("--data", default=DEFAULT_DATA, help="Clean dataset path.")
    rs.add_argument("--outdir", default="/home/zammorak/thesis/models/random_search_direction_v2")
    rs.add_argument("--val_size", type=int, default=126)
    rs.add_argument("--test_size", type=int, default=252)
    rs.add_argument("--trials", type=int, default=50)
    rs.add_argument("--max_epochs", type=int, default=50)
    rs.add_argument("--patience", type=int, default=6)
    rs.add_argument("--seed", type=int, default=1337)
    rs.add_argument("--gpu", default=None, help="AMD GPU index for HIP_VISIBLE_DEVICES, e.g. 0")
    rs.add_argument("--auto_threshold", action="store_true", help="Select threshold per trial on validation via Youden J.")
    rs.add_argument("--save_within", type=float, default=0.01, help="Save trial models within this AUC of best; 0 disables.")
    rs.add_argument("--search_space", choices=["v1", "v2"], default="v2", help="v1 uses fixed dense=32; v2 samples dense units.")
    rs.add_argument("--verbose", type=int, default=0, help="Keras fit verbosity.")
    rs.set_defaults(func=run_random_search)

    wf = sub.add_parser("walk-forward", help="Run expanding-window walk-forward backtest.")
    wf.add_argument("--data", default=DEFAULT_DATA)
    wf.add_argument("--outdir", default="/home/zammorak/thesis/models/walk_forward_direction")
    wf.add_argument("--lookback", type=int, default=60)
    wf.add_argument("--initial_train", type=int, default=700)
    wf.add_argument("--val_size", type=int, default=126)
    wf.add_argument("--test_horizon", type=int, default=63)
    wf.add_argument("--step", type=int, default=63)
    wf.add_argument("--epochs", type=int, default=30)
    wf.add_argument("--batch", type=int, default=32)
    wf.add_argument("--lr", type=float, default=2e-4)
    wf.add_argument("--lstm_units", type=int, default=32)
    wf.add_argument("--dense_units", type=int, default=64)
    wf.add_argument("--dropout", type=float, default=0.05)
    wf.add_argument("--recurrent_dropout", type=float, default=0.2)
    wf.add_argument("--patience", type=int, default=6)
    wf.add_argument("--auto_threshold", action="store_true")
    wf.add_argument("--threshold", type=float, default=0.5)
    wf.add_argument("--gpu", default=None)
    wf.add_argument("--verbose", type=int, default=1, help="Keras fit verbosity.")
    wf.set_defaults(func=run_walk_forward)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
