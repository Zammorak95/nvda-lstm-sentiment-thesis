#!/usr/bin/env python3
"""Run simple thesis baseline models under chronological walk-forward validation.

The goal is not to replace the LSTM, but to create a defensible comparison set:
- Majority-class classifier
- Logistic regression
- Random Forest
- SVM with RBF kernel

The script uses the same high-level walk-forward idea as the LSTM evaluation:
training data comes first, the validation window is used only for threshold
selection, and the test horizon is kept out-of-sample.

Default command:
    python -m thesis.eval.run_baseline_models

Extended feature-group ablation:
    python -m thesis.eval.run_baseline_models --run-ablations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy import stats  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _ensure_src_on_path() -> None:
    root = _find_project_root()
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

try:
    from thesis.paths import ARTIFACTS_DIR, DATA_DIR
except Exception:
    _ROOT = _find_project_root()
    DATA_DIR = Path(os.getenv("THESIS_DATA_DIR", _ROOT / "data")).resolve()
    ARTIFACTS_DIR = Path(os.getenv("THESIS_ARTIFACTS_DIR", _ROOT / "artifacts")).resolve()


DEFAULT_DATASET = DATA_DIR / "model_feed" / "model_dataset_clean.csv"
DEFAULT_OUTDIR = ARTIFACTS_DIR / "reports" / "baseline_models"

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


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    predictions: Path


def make_output_dirs(outdir: Path) -> OutputPaths:
    paths = OutputPaths(
        root=outdir,
        figures=outdir / "figures",
        tables=outdir / "tables",
        predictions=outdir / "predictions",
    )
    for path in (paths.root, paths.figures, paths.tables, paths.predictions):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    if "target_direction" not in df.columns:
        raise ValueError("target_direction missing from dataset.")
    return df


def clean_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def available_feature_sets(df: pd.DataFrame, run_ablations: bool) -> dict[str, list[str]]:
    all_numeric = [
        c
        for c in df.columns
        if c not in {"date", "target_direction", "target_next_return"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    sets = {"all_features": all_numeric}
    if run_ablations:
        groups = {
            "market_only": MARKET_FEATURES,
            "market_macro": MARKET_FEATURES + MACRO_FEATURES,
            "alternative_only": SENTIMENT_FEATURES + ATTENTION_FEATURES,
            "no_alternative_data": MARKET_FEATURES + MACRO_FEATURES,
            "sentiment_only": SENTIMENT_FEATURES,
            "attention_only": ATTENTION_FEATURES,
        }
        for name, cols in groups.items():
            existing = [c for c in cols if c in df.columns]
            if existing:
                sets[name] = existing
    return sets


def build_models(random_state: int) -> dict[str, Any]:
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def model_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.repeat(float(proba[0, 0]), len(X))
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def choose_threshold(y_val: np.ndarray, val_score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(val_score)) <= 2:
        candidates = np.unique(val_score)
    else:
        candidates = np.quantile(val_score, np.linspace(0.05, 0.95, 19))
        candidates = np.unique(candidates)

    best_t, best_j = float(np.median(val_score)), -1e9
    for t in candidates:
        pred = (val_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        j = float(tpr - fpr)
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t, best_j


def annualized_sharpe(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0 or returns.std(ddof=0) == 0:
        return float("nan")
    return float(returns.mean() / returns.std(ddof=0) * np.sqrt(252))


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(np.min(drawdown))


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return float("nan")


def binomial_accuracy_pvalue(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = int(np.sum(y_true == y_pred))
    n = int(len(y_true))
    if n == 0:
        return float("nan")
    return float(stats.binomtest(correct, n=n, p=0.5, alternative="greater").pvalue)


def one_sample_ttest_pvalue(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 3 or returns.std(ddof=1) == 0:
        return float("nan")
    return float(stats.ttest_1samp(returns, popmean=0.0, alternative="greater").pvalue)


def permutation_mean_return_pvalue(
    strategy_returns: np.ndarray,
    base_returns: np.ndarray,
    trade_flag: np.ndarray,
    n_perm: int,
    seed: int,
) -> float:
    if n_perm <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    strategy_returns = np.asarray(strategy_returns, dtype=float)
    base_returns = np.asarray(base_returns, dtype=float)
    trade_flag = np.asarray(trade_flag, dtype=int)
    obs = float(np.nanmean(strategy_returns))
    perm_means = np.empty(n_perm)
    for i in range(n_perm):
        perm_means[i] = float(np.nanmean(base_returns * rng.permutation(trade_flag)))
    return float((np.sum(perm_means >= obs) + 1) / (n_perm + 1))


def evaluate_predictions(df_pred: pd.DataFrame, n_perm: int, seed: int) -> dict[str, float | int | list[list[int]]]:
    y_true = df_pred["y_true"].astype(int).to_numpy()
    y_pred = df_pred["y_pred"].astype(int).to_numpy()
    score = df_pred["score"].astype(float).to_numpy()
    strategy_returns = df_pred.get("strategy_return", pd.Series(np.zeros(len(df_pred)))).astype(float).to_numpy()
    base_returns = df_pred.get("target_next_return", pd.Series(np.zeros(len(df_pred)))).astype(float).to_numpy()
    trade_flag = y_pred

    brier = float("nan")
    if np.nanmin(score) >= 0 and np.nanmax(score) <= 1:
        brier = float(brier_score_loss(y_true, score))

    return {
        "observations": int(len(df_pred)),
        "auc": safe_auc(y_true, score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": brier,
        "trade_rate": float(np.mean(trade_flag)),
        "mean_strategy_return": float(np.nanmean(strategy_returns)),
        "strategy_sharpe": annualized_sharpe(strategy_returns),
        "max_drawdown": max_drawdown_from_returns(strategy_returns),
        "accuracy_binom_pvalue_vs_0p50": binomial_accuracy_pvalue(y_true, y_pred),
        "strategy_mean_ttest_pvalue_gt_0": one_sample_ttest_pvalue(strategy_returns),
        "strategy_mean_permutation_pvalue": permutation_mean_return_pvalue(
            strategy_returns=strategy_returns,
            base_returns=base_returns,
            trade_flag=trade_flag,
            n_perm=n_perm,
            seed=seed,
        ),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def run_walk_forward(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_cols: list[str],
    model_names: list[str],
    args: argparse.Namespace,
    outputs: OutputPaths,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    X_raw = clean_feature_frame(df, feature_cols)
    y = df["target_direction"].astype(int).to_numpy()
    n = len(df)
    if args.initial_train + args.test_horizon >= n:
        raise ValueError("Not enough rows for the requested walk-forward settings.")

    all_models = build_models(args.seed)
    selected_models = {name: all_models[name] for name in model_names if name in all_models}
    if not selected_models:
        raise ValueError(f"No valid models selected. Available: {sorted(all_models)}")

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    rf_importances: list[dict[str, Any]] = []

    fold = 0
    train_end = args.initial_train
    while True:
        val_start = train_end - args.val_size
        test_end = train_end + args.test_horizon
        if test_end > n:
            break
        if val_start <= 0:
            raise ValueError("val_size is too large relative to initial_train.")

        fold += 1
        X_train = X_raw.iloc[:val_start]
        y_train = y[:val_start]
        X_val = X_raw.iloc[val_start:train_end]
        y_val = y[val_start:train_end]
        X_test = X_raw.iloc[train_end:test_end]
        y_test = y[train_end:test_end]

        # Median imputation is fit on training only.
        train_median = X_train.median(numeric_only=True)
        X_train = X_train.fillna(train_median)
        X_val = X_val.fillna(train_median)
        X_test = X_test.fillna(train_median)

        for model_name, model in build_models(args.seed + fold).items():
            if model_name not in selected_models:
                continue
            fitted = model.fit(X_train, y_train)
            val_score = model_score(fitted, X_val)
            test_score = model_score(fitted, X_test)

            if model_name == "majority_class":
                val_pred = fitted.predict(X_val).astype(int)
                test_pred = fitted.predict(X_test).astype(int)
                threshold, youden_j = float("nan"), float("nan")
            else:
                threshold, youden_j = choose_threshold(y_val, val_score)
                val_pred = (val_score >= threshold).astype(int)
                test_pred = (test_score >= threshold).astype(int)

            test_dates = df["date"].iloc[train_end:test_end].reset_index(drop=True) if "date" in df.columns else pd.RangeIndex(len(y_test))
            pred = pd.DataFrame(
                {
                    "date": test_dates,
                    "fold": fold,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "y_true": y_test,
                    "score": test_score,
                    "y_pred": test_pred,
                    "threshold": threshold,
                    "youden_j": youden_j,
                }
            )
            if "target_next_return" in df.columns:
                returns = df["target_next_return"].iloc[train_end:test_end].reset_index(drop=True).astype(float)
                pred["target_next_return"] = returns
                pred["strategy_return"] = returns.to_numpy() * test_pred

            prediction_frames.append(pred)

            fold_metric = evaluate_predictions(pred, n_perm=0, seed=args.seed)
            fold_metric.update(
                {
                    "fold": fold,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "feature_count": len(feature_cols),
                    "val_auc": safe_auc(y_val, val_score),
                    "val_accuracy": float(accuracy_score(y_val, val_pred)),
                    "threshold": threshold,
                    "youden_j": youden_j,
                }
            )
            fold_rows.append(fold_metric)

            if model_name == "random_forest" and feature_set_name == "all_features":
                importances = fitted.feature_importances_
                for feature, value in zip(feature_cols, importances, strict=False):
                    rf_importances.append({"fold": fold, "feature": feature, "importance": float(value)})

        train_end += args.step

    return fold_rows, prediction_frames, rf_importances


def summarize_predictions(predictions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (feature_set, model), group in predictions.groupby(["feature_set", "model"], sort=True):
        metrics = evaluate_predictions(group, n_perm=args.permutations, seed=args.seed)
        metrics.update({"feature_set": feature_set, "model": model})
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["feature_set", "auc", "strategy_sharpe"], ascending=[True, False, False])


def plot_model_comparison(metrics: pd.DataFrame, outputs: OutputPaths) -> list[Path]:
    created: list[Path] = []
    all_features = metrics[metrics["feature_set"] == "all_features"].copy()
    if all_features.empty:
        return created

    for metric, ylabel, filename in [
        ("auc", "OOS AUC", "baseline_model_auc.png"),
        ("accuracy", "OOS accuracy", "baseline_model_accuracy.png"),
        ("strategy_sharpe", "Strategy Sharpe", "baseline_model_sharpe.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        plot_df = all_features.sort_values(metric, ascending=False)
        ax.bar(plot_df["model"], plot_df[metric])
        if metric == "auc":
            ax.axhline(0.5, linestyle="--", linewidth=1, label="Random AUC")
            ax.legend(frameon=False)
        ax.set_title(f"Baseline comparison: {ylabel}")
        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
        path = outputs.figures / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    return created


def plot_cumulative_returns(predictions: pd.DataFrame, outputs: OutputPaths) -> Path | None:
    if "strategy_return" not in predictions.columns or "date" not in predictions.columns:
        return None
    data = predictions[predictions["feature_set"] == "all_features"].copy()
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for model, group in data.groupby("model"):
        g = group.sort_values("date")
        equity = (1.0 + g["strategy_return"].fillna(0.0)).cumprod() - 1.0
        ax.plot(g["date"], equity, linewidth=1.3, label=model)
    ax.set_title("Baseline cumulative out-of-sample returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path = outputs.figures / "baseline_cumulative_returns.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_set_ablation(metrics: pd.DataFrame, outputs: OutputPaths) -> Path | None:
    if metrics["feature_set"].nunique() <= 1:
        return None
    keep = metrics[metrics["model"].isin(["logistic_regression", "random_forest", "svm_rbf"])].copy()
    if keep.empty:
        return None

    pivot = keep.pivot_table(index="feature_set", columns="model", values="auc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.5, linestyle="--", linewidth=1, label="Random AUC")
    ax.set_title("Feature-set ablation: OOS AUC")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("OOS AUC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path = outputs.figures / "feature_set_ablation_auc.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rf_importance(rf_importances: pd.DataFrame, outputs: OutputPaths) -> Path | None:
    if rf_importances.empty:
        return None
    imp = (
        rf_importances.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ordered = imp.iloc[::-1]
    ax.barh(ordered["feature"], ordered["importance"])
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("Mean impurity-based importance across folds")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    path = outputs.figures / "random_forest_feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        df.to_markdown(path.with_suffix(".md"), index=False)
    except Exception:
        path.with_suffix(".md").write_text(df.to_string(index=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis baseline models with walk-forward validation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--initial-train", type=int, default=700)
    parser.add_argument("--val-size", type=int, default=126)
    parser.add_argument("--test-horizon", type=int, default=63)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=500, help="Permutation samples for mean-return timing test.")
    parser.add_argument("--run-ablations", action="store_true", help="Also run feature-group ablation tests.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["majority_class", "logistic_regression", "random_forest", "svm_rbf"],
        help="Models to run: majority_class logistic_regression random_forest svm_rbf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    outputs = make_output_dirs(args.outdir)
    df = load_dataset(args.dataset)

    feature_sets = available_feature_sets(df, run_ablations=args.run_ablations)
    all_fold_rows: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_rf_importances: list[dict[str, Any]] = []

    print("Dataset:", args.dataset)
    print("Output :", outputs.root)
    print("Rows   :", len(df))
    print("Feature sets:", ", ".join(feature_sets.keys()))
    print("Models :", ", ".join(args.models))

    for feature_set_name, feature_cols in feature_sets.items():
        print(f"\nRunning feature set: {feature_set_name} ({len(feature_cols)} features)")
        fold_rows, prediction_frames, rf_importances = run_walk_forward(
            df=df,
            feature_set_name=feature_set_name,
            feature_cols=feature_cols,
            model_names=args.models,
            args=args,
            outputs=outputs,
        )
        all_fold_rows.extend(fold_rows)
        all_predictions.extend(prediction_frames)
        all_rf_importances.extend(rf_importances)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = summarize_predictions(predictions, args=args)
    fold_metrics = pd.DataFrame(all_fold_rows)
    rf_importances_df = pd.DataFrame(all_rf_importances)

    predictions.to_csv(outputs.predictions / "baseline_oos_predictions.csv", index=False)
    write_table(metrics, outputs.tables / "baseline_model_metrics")
    write_table(fold_metrics, outputs.tables / "baseline_fold_metrics")
    if not rf_importances_df.empty:
        write_table(rf_importances_df, outputs.tables / "random_forest_feature_importance_by_fold")
        rf_mean = rf_importances_df.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
        write_table(rf_mean, outputs.tables / "random_forest_feature_importance_mean")

    created_figures: list[Path] = []
    created_figures += plot_model_comparison(metrics, outputs)
    cumret = plot_cumulative_returns(predictions, outputs)
    if cumret is not None:
        created_figures.append(cumret)
    ablation = plot_feature_set_ablation(metrics, outputs)
    if ablation is not None:
        created_figures.append(ablation)
    rf_fig = plot_rf_importance(rf_importances_df, outputs)
    if rf_fig is not None:
        created_figures.append(rf_fig)

    summary = {
        "dataset": str(args.dataset),
        "outdir": str(outputs.root),
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "models": args.models,
        "walk_forward": {
            "initial_train": args.initial_train,
            "val_size": args.val_size,
            "test_horizon": args.test_horizon,
            "step": args.step,
        },
        "best_by_auc": metrics.sort_values("auc", ascending=False).head(5).to_dict(orient="records"),
        "figures": [str(p) for p in created_figures],
    }
    (outputs.root / "baseline_report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBaseline model report complete.")
    print("Metrics:", outputs.tables / "baseline_model_metrics.csv")
    print("Predictions:", outputs.predictions / "baseline_oos_predictions.csv")
    print("Figures:", outputs.figures)


if __name__ == "__main__":
    main()
