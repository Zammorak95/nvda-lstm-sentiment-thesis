#!/usr/bin/env python3
"""Run thesis baseline models with a linear SVM instead of an RBF-kernel SVM.

This wrapper reuses the baseline-model reporting pipeline but replaces the
`svm_rbf` benchmark with `svm_linear`. This is useful when the thesis literature
supports SVM as a classical benchmark, but the methodology does not motivate a
specific non-linear RBF kernel.

Default command:
    python -m thesis.eval.run_baseline_models_linear_svm

With feature-group ablation:
    python -m thesis.eval.run_baseline_models_linear_svm --run-ablations
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from thesis.eval import run_baseline_models as base


DEFAULT_OUTDIR = base.ARTIFACTS_DIR / "reports" / "baseline_models_linear_svm"


def build_models(random_state: int) -> dict[str, Any]:
    """Return the thesis baseline set using a standard linear SVM."""
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
        "svm_linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run thesis baseline models with walk-forward validation and a linear SVM."
    )
    parser.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--initial-train", type=int, default=700)
    parser.add_argument("--val-size", type=int, default=126)
    parser.add_argument("--test-horizon", type=int, default=63)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--permutations",
        type=int,
        default=500,
        help="Permutation samples for mean-return timing test.",
    )
    parser.add_argument("--run-ablations", action="store_true", help="Also run feature-group ablation tests.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["majority_class", "logistic_regression", "random_forest", "svm_linear"],
        help="Models to run: majority_class logistic_regression random_forest svm_linear",
    )
    return parser.parse_args()


def plot_feature_set_ablation(metrics, outputs):
    """Plot feature ablations for the non-RBF benchmark set."""
    if metrics["feature_set"].nunique() <= 1:
        return None
    keep = metrics[
        metrics["model"].isin(["logistic_regression", "random_forest", "svm_linear"])
    ].copy()
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


def main() -> None:
    # Patch the reusable baseline pipeline with the linear-SVM choices.
    base.build_models = build_models
    base.parse_args = parse_args
    base.plot_feature_set_ablation = plot_feature_set_ablation
    base.main()


if __name__ == "__main__":
    main()
