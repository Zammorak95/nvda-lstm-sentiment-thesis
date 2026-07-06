#!/usr/bin/env python3
"""LSTM regression: predict future realized volatility (default horizon=5).

Target:
  future_vol = std(log_return over next H trading days), shifted -H.

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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


def make_sequences(X, y, lookback):
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
            implementation=1,
        ),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to model_dataset.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for model + artifacts")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--val_size", type=int, default=126)
    ap.add_argument("--test_size", type=int, default=252)
    ap.add_argument("--horizon", type=int, default=5, help="Vol horizon in trading days")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lstm_units", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--recurrent_dropout", type=float, default=0.2)
    ap.add_argument("--gpu", default=None, help="AMD GPU index for HIP_VISIBLE_DEVICES (e.g., 0)")
    args = ap.parse_args()

    _set_gpu(args.gpu)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "log_return" not in df.columns:
        raise ValueError("log_return missing; needed to compute volatility target.")

    H = int(args.horizon)
    df["future_vol"] = df["log_return"].rolling(H).std().shift(-H)
    df = df.dropna(subset=["future_vol"]).reset_index(drop=True)

    drop_cols = {"date", "target_next_return", "target_direction", "future_vol"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].select_dtypes(include=[np.number]).copy()

    X_all = X_df.to_numpy()
    y_all = df["future_vol"].astype(float).to_numpy()

    n = len(df)
    if args.test_size + args.val_size + args.lookback >= n:
        raise ValueError("Not enough rows for chosen lookback/val/test sizes.")

    test_start = n - args.test_size
    val_start = test_start - args.val_size

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_all[:val_start])
    X_val_scaled = scaler.transform(X_all[val_start:test_start])
    X_test_scaled = scaler.transform(X_all[test_start:])
    X_scaled = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])

    X_seq, y_seq = make_sequences(X_scaled, y_all, args.lookback)
    seq_val_start = val_start - args.lookback
    seq_test_start = test_start - args.lookback

    X_train, y_train = X_seq[:seq_val_start], y_seq[:seq_val_start]
    X_val, y_val = X_seq[seq_val_start:seq_test_start], y_seq[seq_val_start:seq_test_start]
    X_test, y_test = X_seq[seq_test_start:], y_seq[seq_test_start:]

    print("TF GPUs:", tf.config.list_physical_devices("GPU"))
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    model = build_model(args.lookback, X_train.shape[-1], args.lr, args.lstm_units, args.dropout, args.recurrent_dropout)

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=1
    )

    y_pred = model.predict(X_test).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    print("\nTest RMSE:", rmse)
    print("Test MAE :", mae)

    model_path = os.path.join(args.outdir, "lstm_vol_model.keras")
    scaler_path = os.path.join(args.outdir, "lstm_vol_scaler.joblib")
    meta_path = os.path.join(args.outdir, "lstm_vol_meta.json")
    pred_path = os.path.join(args.outdir, "lstm_vol_test_predictions.csv")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    test_dates = df["date"].iloc[test_start:].reset_index(drop=True)
    if len(test_dates) != len(y_test):
        test_dates = test_dates.iloc[-len(y_test):].reset_index(drop=True)

    pd.DataFrame({
        "date": test_dates,
        "y_true_future_vol": y_test,
        "y_pred_future_vol": y_pred
    }).to_csv(pred_path, index=False)

    meta = {
        "task": "volatility_regression",
        "target": f"future_vol_{H}d",
        "lookback": args.lookback,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "feature_cols": X_df.columns.tolist(),
        "metrics": {"rmse": rmse, "mae": mae},
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", None),
        "lstm": {"implementation": 1, "dropout": args.dropout, "recurrent_dropout": args.recurrent_dropout},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print("  model:", model_path)
    print("  scaler:", scaler_path)
    print("  meta :", meta_path)
    print("  preds:", pred_path)


if __name__ == "__main__":
    main()
