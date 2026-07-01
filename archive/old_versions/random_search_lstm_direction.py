#!/usr/bin/env python3
import os
import json
import time
import argparse
import random
import numpy as np
import pandas as pd

def _set_gpu(gpu_index):
    if gpu_index is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_index)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import joblib


def make_sequences(X, y, lookback: int):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def build_model(lookback, n_features, lr, lstm_units, dropout, rec_dropout):
    # ROCm-safe: implementation=1 and recurrent_dropout>0
    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(
            lstm_units,
            dropout=dropout,
            recurrent_dropout=rec_dropout,
            implementation=1
        ),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]
    )
    return model


def compute_class_weight(y_train: np.ndarray):
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if pos == 0 or neg == 0:
        return None
    return {0: float((pos + neg) / (2 * neg)), 1: float((pos + neg) / (2 * pos))}


def best_threshold_from_val(y_val: np.ndarray, val_prob: np.ndarray):
    # Youden J
    best_t, best_j = 0.5, -1e9
    for t in np.linspace(0.05, 0.95, 19):
        pred = (val_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = float(j), float(t)
    return best_t, best_j


def sample_params(rng: random.Random):
    return {
        "lookback": rng.choice([45, 60, 75, 90, 105]),
        "lstm_units": rng.choice([16, 32, 48, 64]),
        "dropout": rng.choice([0.02, 0.05, 0.08, 0.10]),
        "recurrent_dropout": rng.choice([0.15, 0.20, 0.25]),
        "lr": rng.choice([1e-4, 2e-4, 3e-4]),
        "batch": rng.choice([16, 32, 64]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/zammorak/thesis/data/model_feed/model_dataset_clean.csv")
    ap.add_argument("--outdir", default="/home/zammorak/thesis/models/random_search_direction")
    ap.add_argument("--val_size", type=int, default=126)
    ap.add_argument("--test_size", type=int, default=252)

    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--max_epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--gpu", default=None)
    ap.add_argument("--auto_threshold", action="store_true", help="Select threshold on validation each trial")
    args = ap.parse_args()

    _set_gpu(args.gpu)
    os.makedirs(args.outdir, exist_ok=True)

    results_csv = os.path.join(args.outdir, "random_search_results.csv")
    best_dir = os.path.join(args.outdir, "best")
    os.makedirs(best_dir, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "target_direction" not in df.columns:
        raise ValueError("target_direction missing in dataset.")

    # Features
    drop_cols = {"date", "target_next_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y_all = df["target_direction"].astype(int).to_numpy()

    n = len(df)
    if args.test_size + args.val_size + 10 >= n:
        raise ValueError("Not enough rows for chosen val/test sizes.")

    test_start = n - args.test_size
    val_start = test_start - args.val_size

    # Scale train-only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_df.to_numpy()[:val_start])
    X_val_scaled = scaler.transform(X_df.to_numpy()[val_start:test_start])
    X_test_scaled = scaler.transform(X_df.to_numpy()[test_start:])
    X_scaled_full = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])

    best_val_auc = -1.0
    best_payload = None

    rows = []
    for trial in range(1, args.trials + 1):
        params = sample_params(rng)
        lookback = params["lookback"]

        if args.test_size + args.val_size + lookback >= n:
            # skip impossible lookback
            continue

        # sequences depend on lookback, so rebuild splits per trial
        X_seq, y_seq = make_sequences(X_scaled_full, y_all, lookback)
        seq_val_start = val_start - lookback
        seq_test_start = test_start - lookback

        X_train, y_train = X_seq[:seq_val_start], y_seq[:seq_val_start]
        X_val, y_val = X_seq[seq_val_start:seq_test_start], y_seq[seq_val_start:seq_test_start]

        model = build_model(
            lookback=lookback,
            n_features=X_train.shape[-1],
            lr=params["lr"],
            lstm_units=params["lstm_units"],
            dropout=params["dropout"],
            rec_dropout=params["recurrent_dropout"]
        )

        cb = [
            callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=args.patience, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=max(2, args.patience//2), factor=0.5, min_lr=1e-5),
        ]

        class_weight = compute_class_weight(y_train)

        t0 = time.time()
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.max_epochs,
            batch_size=params["batch"],
            verbose=0,
            callbacks=cb,
            class_weight=class_weight
        )
        seconds = time.time() - t0

        # Validation scoring
        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        val_auc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan")

        if args.auto_threshold:
            thr, j = best_threshold_from_val(y_val, val_prob)
        else:
            thr, j = 0.5, None

        val_pred = (val_prob >= thr).astype(int)
        val_acc = float(accuracy_score(y_val, val_pred))
        best_epoch = int(np.argmax(hist.history.get("val_auc", [val_auc])) + 1)

        row = {
            "trial": trial,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "threshold": float(thr),
            "youden_j": (float(j) if j is not None else None),
            "best_epoch": best_epoch,
            "seconds": seconds,
            **params
        }
        rows.append(row)

        # Keep best model
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_payload = {
                "params": params,
                "val_auc": val_auc,
                "val_acc": val_acc,
                "threshold": float(thr),
                "feature_cols": X_df.columns.tolist(),
            }
            # Save best artifacts
            model.save(os.path.join(best_dir, "model.keras"))
            joblib.dump(scaler, os.path.join(best_dir, "scaler.joblib"))
            with open(os.path.join(best_dir, "meta.json"), "w") as f:
                json.dump(best_payload, f, indent=2)

        print(f"[{trial:03d}/{args.trials}] val_auc={val_auc:.4f} val_acc={val_acc:.4f} "
              f"lb={lookback} units={params['lstm_units']} lr={params['lr']} "
              f"drop={params['dropout']} rdrop={params['recurrent_dropout']} batch={params['batch']}")

        # Write incremental results
        pd.DataFrame(rows).to_csv(results_csv, index=False)

    print("\nDONE")
    print("Results saved:", results_csv)
    if best_payload:
        print("Best val_auc:", best_val_auc)
        print("Best model saved in:", best_dir)


if __name__ == "__main__":
    main()
