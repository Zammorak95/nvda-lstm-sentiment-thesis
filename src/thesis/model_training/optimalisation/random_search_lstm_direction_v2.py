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

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
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
DEFAULT_OUTDIR = Path(os.getenv("THESIS_MODELS_DIR", MODELS)) / "random_search_direction_v2"


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
        Xs.append(X[i - lookback : i])
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
    dense_units: int = 32,
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


def best_threshold_from_val(y_val: np.ndarray, val_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_j = 0.5, -1e9
    for t in np.linspace(0.05, 0.95, 19):
        pred = (val_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        j = float(tpr - fpr)
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, best_j


def eval_auc_acc(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> tuple[float, float, list[list[int]]]:
    pred = (prob >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    acc = float(accuracy_score(y_true, pred))
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    return auc, acc, cm


@dataclass
class TrialResult:
    trial: int
    val_auc: float
    val_acc: float
    threshold: float
    youden_j: float | None
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
    val_cm_tn_fp_fn_tp: list[list[int]]


def sample_params(rng: random.Random) -> dict[str, Any]:
    return {
        "lookback": rng.choice([45, 60, 75, 90, 105]),
        "lstm_units": rng.choice([16, 32, 48, 64, 96]),
        "dense_units": rng.choice([16, 32, 64]),
        "dropout": rng.choice([0.02, 0.05, 0.08, 0.10]),
        "recurrent_dropout": rng.choice([0.15, 0.20, 0.25]),
        "lr": rng.choice([1e-4, 2e-4, 3e-4]),
        "batch": rng.choice([16, 32, 64]),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Clean dataset path.")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    ap.add_argument("--val_size", type=int, default=126)
    ap.add_argument("--test_size", type=int, default=252)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--gpu", default=None, help="AMD GPU index for HIP_VISIBLE_DEVICES, e.g. 0")
    ap.add_argument("--auto_threshold", action="store_true")
    ap.add_argument("--save_within", type=float, default=0.01)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tf, layers, models, callbacks = _import_tensorflow(args.gpu)

    args.outdir.mkdir(parents=True, exist_ok=True)
    results_csv = args.outdir / "random_search_results.csv"
    best_dir = args.outdir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

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
    X_all = X_df.to_numpy(dtype=np.float32)

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
    best_payload: dict[str, Any] | None = None
    trial_rows: list[dict[str, Any]] = []

    print("TF GPUs:", tf.config.list_physical_devices("GPU"))
    print("Dataset:", args.data)
    print("Output :", args.outdir)
    print("Dataset rows:", n, "| features:", X_df.shape[1])
    print("Split sizes -> train:", val_start, "val:", args.val_size, "test:", args.test_size)

    for trial in range(1, args.trials + 1):
        params = sample_params(rng)
        lookback = int(params["lookback"])
        if args.test_size + args.val_size + lookback >= n:
            continue

        X_seq, y_seq = make_sequences(X_scaled_full, y_all, lookback)
        seq_val_start = val_start - lookback
        seq_test_start = test_start - lookback
        X_train, y_train = X_seq[:seq_val_start], y_seq[:seq_val_start]
        X_val, y_val = X_seq[seq_val_start:seq_test_start], y_seq[seq_val_start:seq_test_start]

        model = build_model(
            tf=tf,
            layers=layers,
            models=models,
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
            verbose=0,
            callbacks=cb,
            class_weight=compute_class_weight(y_train),
        )
        seconds = time.time() - t0

        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        val_auc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else float("nan")

        if args.auto_threshold:
            thr, j = best_threshold_from_val(y_val, val_prob)
        else:
            thr, j = 0.5, None

        _, val_acc, val_cm = eval_auc_acc(y_val, val_prob, thr)
        row = TrialResult(
            trial=trial,
            val_auc=val_auc,
            val_acc=val_acc,
            threshold=float(thr),
            youden_j=(float(j) if j is not None else None),
            best_epoch=int(np.argmax(hist.history.get("val_auc", [val_auc])) + 1),
            best_val_auc_in_training=float(np.max(hist.history.get("val_auc", [np.nan]))),
            best_val_loss_in_training=float(np.min(hist.history.get("val_loss", [np.nan]))),
            seconds=float(seconds),
            lookback=lookback,
            lstm_units=int(params["lstm_units"]),
            dropout=float(params["dropout"]),
            recurrent_dropout=float(params["recurrent_dropout"]),
            lr=float(params["lr"]),
            batch=int(params["batch"]),
            dense_units=int(params["dense_units"]),
            val_cm_tn_fp_fn_tp=val_cm,
        )
        trial_rows.append(asdict(row))
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
                "data": str(args.data),
            }
            model.save(best_dir / "model.keras")
            joblib.dump(scaler, best_dir / "scaler.joblib")
            with open(best_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2)

        if args.save_within > 0 and np.isfinite(val_auc) and best_val_auc > 0:
            if val_auc >= (best_val_auc - float(args.save_within)):
                close_dir = args.outdir / f"top_trial_{trial:03d}_auc_{val_auc:.4f}"
                close_dir.mkdir(parents=True, exist_ok=True)
                model.save(close_dir / "model.keras")

        print(
            f"[{trial:03d}/{args.trials}] val_auc={val_auc:.4f} val_acc={val_acc:.4f} "
            f"thr={thr:.2f} lb={lookback} units={params['lstm_units']} dense={params['dense_units']} "
            f"lr={params['lr']} drop={params['dropout']} rdrop={params['recurrent_dropout']} batch={params['batch']}"
        )

    print("\nRandom search complete.")
    print("Results CSV:", results_csv)

    if best_payload is None:
        print("No valid trials produced a best model.")
        return

    lb = int(best_payload["params"]["lookback"])
    X_seq, y_seq = make_sequences(X_scaled_full, y_all, lb)
    X_test = X_seq[test_start - lb :]
    y_test = y_seq[test_start - lb :]

    best_model = tf.keras.models.load_model(best_dir / "model.keras")
    test_prob = best_model.predict(X_test, verbose=0).reshape(-1)
    thr = float(best_payload.get("threshold", 0.5))
    test_auc, test_acc, test_cm = eval_auc_acc(y_test, test_prob, thr)

    print("\nBEST MODEL")
    print("  val_auc:", best_payload["val_auc"], "| val_acc:", best_payload["val_acc"], "| thr:", thr)
    print("  test_auc:", test_auc, "| test_acc:", test_acc, "| test_cm:", test_cm)

    best_payload["test_auc"] = float(test_auc)
    best_payload["test_acc"] = float(test_acc)
    best_payload["test_cm_tn_fp_fn_tp"] = test_cm
    best_payload["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(best_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    print("\nBest artifacts saved in:", best_dir)
    print("  model  :", best_dir / "model.keras")
    print("  scaler :", best_dir / "scaler.joblib")
    print("  meta   :", best_dir / "meta.json")


if __name__ == "__main__":
    main()
