#!/usr/bin/env python3
"""Random search hyperparameter optimization for a ROCm-safe LSTM direction classifier.

What it does
- Loads the clean thesis model dataset.
- Uses a chronological train | validation | test split.
- Scales using TRAIN ONLY to avoid look-ahead leakage.
- Randomly samples LSTM hyperparameters from the thesis search space.
- Trains with EarlyStopping on validation AUC.
- Optionally selects a decision threshold on validation using Youden's J.
- Saves every trial to CSV and the best model/scaler/meta to outdir/best/.
- Evaluates the best validation model on the untouched TEST split.

The search space intentionally includes 96 LSTM units because the historical NVDA
thesis run that produced an OOS AUC of approximately 0.5506 used 96 units. This
keeps the generic pipeline capable of rediscovering or approximating that earlier
specification when the same dataset and environment are used.
"""

import os
import json
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

def _set_gpu(gpu_index: Optional[str]) -> None:
    if gpu_index is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_index)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import joblib


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create lookback sequences ending at each time t with label y[t]."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback : i])
        ys.append(int(y[i]))
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
    """ROCm-safe LSTM classifier."""
    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(
            lstm_units,
            dropout=dropout,
            recurrent_dropout=rec_dropout,
            implementation=1,   # ROCm-safe non-fused path
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
    """Balanced class weights."""
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if pos == 0 or neg == 0:
        return None
    total = pos + neg
    return {0: float(total / (2 * neg)), 1: float(total / (2 * pos))}


def best_threshold_from_val(y_val: np.ndarray, val_prob: np.ndarray) -> Tuple[float, float]:
    """Pick threshold using Youden's J (TPR - FPR)."""
    best_t, best_j = 0.5, -1e9
    for t in np.linspace(0.05, 0.95, 19):
        pred = (val_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
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


def sample_params(rng: random.Random) -> Dict[str, Any]:
    """
    Narrowed search space based on your early best trials.
    You can widen if you want, but this is a good 'round 2' space.
    """
    return {
        "lookback": rng.choice([45, 60, 75, 90, 105]),
        "lstm_units": rng.choice([16, 32, 48, 64, 96]),
        "dense_units": rng.choice([16, 32, 64]),
        "dropout": rng.choice([0.02, 0.05, 0.08, 0.10]),
        "recurrent_dropout": rng.choice([0.15, 0.20, 0.25]),
        "lr": rng.choice([1e-4, 2e-4, 3e-4]),
        "batch": rng.choice([16, 32, 64]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/zammorak/thesis/data/model_feed/model_dataset_clean.csv",
                    help="Clean dataset path (recommended).")
    ap.add_argument("--outdir", default="/home/zammorak/thesis/models/random_search_direction_v2",
                    help="Where to write results + best artifacts.")
    ap.add_argument("--val_size", type=int, default=126)
    ap.add_argument("--test_size", type=int, default=252)

    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--gpu", default=None, help="AMD GPU index for HIP_VISIBLE_DEVICES, e.g. 0")
    ap.add_argument("--auto_threshold", action="store_true", help="Select threshold per trial on validation (Youden J).")
    ap.add_argument("--save_within", type=float, default=0.01,
                    help="Also save trial models within this AUC of best (top-k-ish). Set 0 to disable.")
    args = ap.parse_args()

    _set_gpu(args.gpu)
    os.makedirs(args.outdir, exist_ok=True)

    results_csv = os.path.join(args.outdir, "random_search_results.csv")
    best_dir = os.path.join(args.outdir, "best")
    os.makedirs(best_dir, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # --- Load data ---
    df = pd.read_csv(args.data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    if "target_direction" not in df.columns:
        raise ValueError("target_direction missing in dataset.")

    drop_cols = {"date", "target_next_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].select_dtypes(include=[np.number]).copy()
    if X_df.shape[1] == 0:
        raise ValueError("No numeric features found after dropping meta/targets.")

    y_all = df["target_direction"].astype(int).to_numpy()
    X_all = X_df.to_numpy()

    # --- Split points ---
    n = len(df)
    if args.test_size + args.val_size + 10 >= n:
        raise ValueError("Not enough rows for chosen val/test sizes.")

    test_start = n - args.test_size
    val_start = test_start - args.val_size

    # --- Scale with TRAIN ONLY (no leakage) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_all[:val_start])
    X_val_scaled = scaler.transform(X_all[val_start:test_start])
    X_test_scaled = scaler.transform(X_all[test_start:])
    X_scaled_full = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])

    best_val_auc = -1.0
    best_payload: Optional[Dict[str, Any]] = None

    trial_rows: List[Dict[str, Any]] = []

    print("TF GPUs:", tf.config.list_physical_devices("GPU"))
    print("Dataset rows:", n, "| features:", X_df.shape[1])
    print("Split sizes -> train:", val_start, "val:", args.val_size, "test:", args.test_size)

    for trial in range(1, args.trials + 1):
        params = sample_params(rng)
        lookback = int(params["lookback"])

        # Ensure the chosen lookback allows sequence splits
        if args.test_size + args.val_size + lookback >= n:
            # skip impossible lookback
            continue

        # Build sequences for this lookback
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
            callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=args.patience, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", patience=max(2, args.patience // 2),
                factor=0.5, min_lr=1e-5
            ),
        ]

        class_weight = compute_class_weight(y_train)

        t0 = time.time()
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.max_epochs,
            batch_size=int(params["batch"]),
            verbose=0,
            callbacks=cb,
            class_weight=class_weight,
        )
        seconds = time.time() - t0

        # Val predictions (post-training)
        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        val_auc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan")

        if args.auto_threshold:
            thr, j = best_threshold_from_val(y_val, val_prob)
        else:
            thr, j = 0.5, None

        val_auc2, val_acc, val_cm = eval_auc_acc(y_val, val_prob, thr)
        # val_auc and val_auc2 should match; val_auc2 just reuses helper.

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
        # add confusion matrix for convenience
        row["val_cm_tn_fp_fn_tp"] = val_cm

        trial_rows.append(row)
        pd.DataFrame(trial_rows).to_csv(results_csv, index=False)

        is_best = np.isfinite(val_auc) and (val_auc > best_val_auc)

        # Save best
        if is_best:
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

        # Optionally save models close to best (helps avoid flukes)
        if args.save_within > 0 and np.isfinite(val_auc) and best_val_auc > 0:
            if val_auc >= (best_val_auc - float(args.save_within)):
                close_dir = os.path.join(args.outdir, f"top_trial_{trial:03d}_auc_{val_auc:.4f}")
                os.makedirs(close_dir, exist_ok=True)
                model.save(os.path.join(close_dir, "model.keras"))

        print(
            f"[{trial:03d}/{args.trials}] val_auc={val_auc:.4f} val_acc={val_acc:.4f} "
            f"thr={thr:.2f} lb={lookback} units={params['lstm_units']} dense={params['dense_units']} "
            f"lr={params['lr']} drop={params['dropout']} rdrop={params['recurrent_dropout']} batch={params['batch']}"
        )

    print("\nRandom search complete.")
    print("Results CSV:", results_csv)

    # --- Evaluate best model on TEST ---
    if best_payload is None:
        print("No valid trials produced a best model.")
        return

    lb = int(best_payload["params"]["lookback"])
    X_seq, y_seq = make_sequences(X_scaled_full, y_all, lb)
    X_test = X_seq[test_start - lb :]
    y_test = y_seq[test_start - lb :]

    X_seq, y_seq = make_sequences(X_scaled_full, y_all, lb)
    seq_test_start = test_start - lb
    X_test = X_seq[seq_test_start:]
    y_test = y_seq[seq_test_start:]

    best_model = tf.keras.models.load_model(os.path.join(best_dir, "model.keras"))
    test_prob = best_model.predict(X_test, verbose=0).reshape(-1)

    thr = float(best_payload.get("threshold", 0.5))
    test_auc, test_acc, test_cm = eval_auc_acc(y_test, test_prob, thr)

    print("\nBEST MODEL")
    print("  val_auc:", best_payload["val_auc"], "| val_acc:", best_payload["val_acc"], "| thr:", thr)
    print("  test_auc:", test_auc, "| test_acc:", test_acc, "| test_cm:", test_cm)

    # append to best meta.json
    best_payload["test_auc"] = float(test_auc)
    best_payload["test_acc"] = float(test_acc)
    best_payload["test_cm_tn_fp_fn_tp"] = test_cm
    best_payload["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(best_dir, "meta.json"), "w") as f:
        json.dump(best_payload, f, indent=2)

    print("\nBest artifacts saved in:", best_dir)
    print("  model  :", os.path.join(best_dir, "model.keras"))
    print("  scaler :", os.path.join(best_dir, "scaler.joblib"))
    print("  meta   :", os.path.join(best_dir, "meta.json"))


if __name__ == "__main__":
    main()
