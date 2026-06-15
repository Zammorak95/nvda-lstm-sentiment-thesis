#!/usr/bin/env python3
"""LSTM binary classification: predict next-day direction (target_direction).

IMPORTANT (ROCm / MIOpen):
- This script forces the non-fused Keras LSTM implementation to avoid the ROCm MIOpen error:
  "ROCm MIOpen only supports packed input output." from CudnnRNNV3.
- We do this by setting: implementation=1 and recurrent_dropout>0, which prevents the fused path.

GPU selection:
- Use HIP_VISIBLE_DEVICES to pick your dedicated AMD GPU.
  Example: HIP_VISIBLE_DEVICES=0 python script.py ...
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

def _set_gpu(gpu_index):
    if gpu_index is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_index)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib


def make_sequences(X, y, lookback: int):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def build_model(lookback, n_features, lr, lstm_units, dropout, rec_dropout):
    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(
            lstm_units,
            dropout=dropout,
            recurrent_dropout=rec_dropout,
            implementation=1,  # ROCm-safe (non-fused)
        ),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def sharpe_ratio(returns: np.ndarray):
    std = np.std(returns)
    if std == 0:
        return float("nan")
    return float(np.mean(returns) / std * np.sqrt(252))


def compute_class_weight(y_train: np.ndarray):
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if pos == 0 or neg == 0:
        return None
    return {0: float((pos + neg) / (2 * neg)), 1: float((pos + neg) / (2 * pos))}


def best_threshold_from_val(y_val: np.ndarray, val_prob: np.ndarray):
    """Pick threshold using Youden's J (TPR - FPR) on the validation set."""
    best_t, best_j = 0.5, -1e9
    for t in np.linspace(0.05, 0.95, 19):
        val_pred = (val_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, val_pred).ravel()
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = float(j), float(t)
    return best_t, best_j


def eval_split(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"\\n{name} Accuracy: {acc}")
    print(f"{name} AUC: {auc}")
    print(f"{name} Confusion matrix [[TN,FP],[FN,TP]]: {cm}")
    print(f"{name} prob stats: min={float(np.min(y_prob)):.4f} mean={float(np.mean(y_prob)):.4f} max={float(np.max(y_prob)):.4f}")
    return {"accuracy": acc, "auc": auc, "confusion_matrix": cm}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default="/home/zammorak/thesis/data/model_feed/model_dataset_clean.csv",
        help="Path to model_dataset.csv",
    )
    ap.add_argument(
        "--outdir",
        default="/home/zammorak/thesis/models",
        help="Output directory for model + artifacts",
    )

    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--val_size", type=int, default=126)
    ap.add_argument("--test_size", type=int, default=252)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--lstm_units", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--recurrent_dropout", type=float, default=0.2)

    ap.add_argument("--gpu", default=None, help="AMD GPU index for HIP_VISIBLE_DEVICES (e.g., 0)")

    ap.add_argument("--threshold", type=float, default=0.45, help="Decision threshold if --auto_threshold is not used")
    ap.add_argument("--auto_threshold", action="store_true", help="Choose threshold using validation set (Youden's J)")

    args = ap.parse_args()

    _set_gpu(args.gpu)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "target_direction" not in df.columns:
        raise ValueError("target_direction missing.")

    # --- Features / target ---
    drop_cols = {"date", "target_next_return", "target_direction"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].select_dtypes(include=[np.number]).copy()
    if X_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after filtering.")

    X_all = X_df.to_numpy()
    y_all = df["target_direction"].astype(int).to_numpy()

    # --- Time split (train | val | test) ---
    n = len(df)
    if args.test_size + args.val_size + args.lookback >= n:
        raise ValueError("Not enough rows for chosen lookback/val/test sizes.")

    test_start = n - args.test_size
    val_start = test_start - args.val_size

    # --- Scale with train only ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_all[:val_start])
    X_val_scaled = scaler.transform(X_all[val_start:test_start])
    X_test_scaled = scaler.transform(X_all[test_start:])
    X_scaled = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])

    # --- Make sequences ---
    X_seq, y_seq = make_sequences(X_scaled, y_all, args.lookback)
    seq_val_start = val_start - args.lookback
    seq_test_start = test_start - args.lookback

    X_train, y_train = X_seq[:seq_val_start], y_seq[:seq_val_start]
    X_val, y_val = X_seq[seq_val_start:seq_test_start], y_seq[seq_val_start:seq_test_start]
    X_test, y_test = X_seq[seq_test_start:], y_seq[seq_test_start:]

    print("TF GPUs:", tf.config.list_physical_devices("GPU"))
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    model = build_model(
        args.lookback, X_train.shape[-1], args.lr, args.lstm_units, args.dropout, args.recurrent_dropout
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
    ]

    class_weight = compute_class_weight(y_train)
    if class_weight is not None:
        print("class_weight:", class_weight)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=1,
        class_weight=class_weight
    )

    # --- Validation probabilities + threshold selection ---
    val_prob = model.predict(X_val).reshape(-1)
    if args.auto_threshold:
        t, j = best_threshold_from_val(y_val, val_prob)
        print(f"\\nAuto threshold from validation: t={t:.2f} (Youden J={j:.4f})")
        threshold = t
    else:
        threshold = float(args.threshold)

    # --- Evaluate on validation + test ---
    val_metrics = eval_split("VAL", y_val, val_prob, threshold)

    test_prob = model.predict(X_test).reshape(-1)
    test_metrics = eval_split("TEST", y_test, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(int)

    # --- Simple strategy metric if next return is available ---
    strat = {}
    if "target_next_return" in df.columns:
        test_returns = df["target_next_return"].iloc[test_start:].to_numpy()
        if len(test_returns) != len(y_test):
            test_returns = test_returns[-len(y_test):]
        strategy_returns = test_returns * test_pred
        strat = {
            "sharpe_long_only": sharpe_ratio(strategy_returns),
            "avg_daily_return_long_only": float(np.mean(strategy_returns)),
        }
        print("\\nStrategy Sharpe (long when up):", strat["sharpe_long_only"])

    # --- Save artifacts ---
    model_path = os.path.join(args.outdir, "lstm_direction_model.keras")
    scaler_path = os.path.join(args.outdir, "lstm_direction_scaler.joblib")
    meta_path = os.path.join(args.outdir, "lstm_direction_meta.json")
    pred_path = os.path.join(args.outdir, "lstm_direction_predictions.csv")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    test_dates = df["date"].iloc[test_start:].reset_index(drop=True)
    if len(test_dates) != len(y_test):
        test_dates = test_dates.iloc[-len(y_test):].reset_index(drop=True)

    pd.DataFrame({
        "date": test_dates,
        "y_true": y_test,
        "y_prob_up": test_prob,
        "y_pred": test_pred
    }).to_csv(pred_path, index=False)

    meta = {
        "task": "direction_classification",
        "target": "target_direction",
        "lookback": args.lookback,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "threshold_used": threshold,
        "auto_threshold": bool(args.auto_threshold),
        "feature_cols": X_df.columns.tolist(),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "strategy": strat,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", None),
        "lstm": {"implementation": 1, "dropout": args.dropout, "recurrent_dropout": args.recurrent_dropout},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\\nSaved:")
    print("  model:", model_path)
    print("  scaler:", scaler_path)
    print("  meta :", meta_path)
    print("  preds:", pred_path)


if __name__ == "__main__":
    main()
