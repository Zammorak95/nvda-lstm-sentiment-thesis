#!/usr/bin/env python3
"""Create thesis-ready combined model-comparison tables.

This script combines the classical benchmark results with the final LSTM
walk-forward result and exports:
- CSV table
- Markdown table
- LaTeX table
- PNG image of the table
- AUC bar chart

Typical command:
    python -m thesis.eval.make_model_comparison_table \
      --baseline-metrics artifacts/reports/baseline_models_linear_svm_ablations/tables/baseline_model_metrics.csv \
      --lstm-summary artifacts/models/walk_forward_direction_bestparams/walk_forward_summary_best_features.json \
      --outdir artifacts/reports/model_comparison

If the LSTM summary file is unavailable, pass the LSTM values manually:
    python -m thesis.eval.make_model_comparison_table \
      --baseline-metrics artifacts/reports/baseline_models_linear_svm_ablations/tables/baseline_model_metrics.csv \
      --lstm-auc 0.550643920654932 \
      --lstm-accuracy 0.5178571428571429 \
      --lstm-sharpe 0.9957887190041333 \
      --lstm-trade-rate 0.5396825396825397
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


MODEL_ORDER = [
    "majority_class",
    "logistic_regression",
    "svm_linear",
    "random_forest",
    "lstm_bestparams",
]

MODEL_LABELS = {
    "majority_class": "Majority class",
    "logistic_regression": "Logistic regression",
    "svm_linear": "Linear SVM",
    "random_forest": "Random Forest",
    "lstm_bestparams": "LSTM (best specification)",
}

MODEL_ROLES = {
    "majority_class": "Naïve benchmark",
    "logistic_regression": "Linear benchmark",
    "svm_linear": "Classical ML benchmark",
    "random_forest": "Non-linear robustness check",
    "lstm_bestparams": "Main model",
}

OUTPUT_COLUMNS = [
    "Model",
    "Role",
    "OOS AUC",
    "OOS accuracy",
    "Balanced accuracy",
    "Strategy Sharpe",
    "Trade rate",
    "Max drawdown",
]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = PROJECT_ROOT / "artifacts" / "reports" / "baseline_models_linear_svm_ablations" / "tables" / "baseline_model_metrics.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "reports" / "model_comparison"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def parse_lstm_metrics(args: argparse.Namespace) -> dict[str, float | None]:
    """Read LSTM metrics from a summary JSON or manual CLI values."""
    values: dict[str, float | None] = {
        "auc": args.lstm_auc,
        "accuracy": args.lstm_accuracy,
        "balanced_accuracy": args.lstm_balanced_accuracy,
        "strategy_sharpe": args.lstm_sharpe,
        "trade_rate": args.lstm_trade_rate,
        "max_drawdown": args.lstm_max_drawdown,
    }

    summary_path = args.lstm_summary
    if summary_path is None:
        summary_path = first_existing(
            [
                PROJECT_ROOT / "artifacts" / "models" / "walk_forward_direction_bestparams" / "walk_forward_summary_best_features.json",
                PROJECT_ROOT / "artifacts" / "models" / "walk_forward_direction_bestparams" / "walk_forward_summary.json",
                PROJECT_ROOT / "models" / "walk_forward_direction_bestparams" / "walk_forward_summary_best_features.json",
                PROJECT_ROOT / "models" / "walk_forward_direction_bestparams" / "walk_forward_summary.json",
            ]
        )

    if summary_path is not None and summary_path.exists():
        summary = read_json(summary_path)
        # Accept several naming variants because older thesis runs used slightly
        # different JSON keys.
        key_map = {
            "auc": ["overall_oos_auc", "overall_auc", "oos_auc", "auc"],
            "accuracy": ["overall_oos_acc", "overall_oos_accuracy", "overall_acc", "oos_acc", "accuracy"],
            "balanced_accuracy": ["balanced_accuracy", "overall_balanced_accuracy"],
            "strategy_sharpe": ["strategy_sharpe", "oos_sharpe_long_only", "sharpe", "sharpe_long_only"],
            "trade_rate": ["trade_rate", "oos_trade_rate"],
            "max_drawdown": ["max_drawdown", "strategy_max_drawdown"],
        }
        for out_key, candidates in key_map.items():
            if values[out_key] is not None:
                continue
            for key in candidates:
                if key in summary and summary[key] is not None:
                    try:
                        values[out_key] = float(summary[key])
                        break
                    except (TypeError, ValueError):
                        continue

        # Some summaries nest strategy metrics.
        strategy = summary.get("strategy") or summary.get("strategy_metrics") or {}
        if isinstance(strategy, dict):
            if values["strategy_sharpe"] is None:
                for key in ["sharpe", "strategy_sharpe", "oos_sharpe_long_only"]:
                    if key in strategy and strategy[key] is not None:
                        values["strategy_sharpe"] = float(strategy[key])
                        break
            if values["trade_rate"] is None:
                for key in ["trade_rate", "oos_trade_rate"]:
                    if key in strategy and strategy[key] is not None:
                        values["trade_rate"] = float(strategy[key])
                        break
            if values["max_drawdown"] is None:
                for key in ["max_drawdown", "strategy_max_drawdown"]:
                    if key in strategy and strategy[key] is not None:
                        values["max_drawdown"] = float(strategy[key])
                        break

    missing_core = [k for k in ["auc", "accuracy"] if values[k] is None]
    if missing_core:
        raise ValueError(
            "Could not determine required LSTM metrics: "
            + ", ".join(missing_core)
            + ". Pass --lstm-auc and --lstm-accuracy manually, or provide --lstm-summary."
        )
    return values


def load_baseline_metrics(path: Path, feature_set: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {path}")
    df = pd.read_csv(path)
    required = {"model", "feature_set", "auc", "accuracy", "strategy_sharpe", "trade_rate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Baseline metrics missing required columns: {sorted(missing)}")
    df = df[df["feature_set"] == feature_set].copy()
    keep = ["majority_class", "logistic_regression", "svm_linear", "random_forest"]
    df = df[df["model"].isin(keep)].copy()
    if df.empty:
        raise ValueError(f"No baseline rows found for feature_set={feature_set!r}")
    return df


def build_comparison_table(baselines: pd.DataFrame, lstm: dict[str, float | None]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER[:-1]:
        row = baselines[baselines["model"] == model]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append(
            {
                "model_key": model,
                "Model": MODEL_LABELS[model],
                "Role": MODEL_ROLES[model],
                "OOS AUC": r.get("auc"),
                "OOS accuracy": r.get("accuracy"),
                "Balanced accuracy": r.get("balanced_accuracy"),
                "Strategy Sharpe": r.get("strategy_sharpe"),
                "Trade rate": r.get("trade_rate"),
                "Max drawdown": r.get("max_drawdown"),
            }
        )

    rows.append(
        {
            "model_key": "lstm_bestparams",
            "Model": MODEL_LABELS["lstm_bestparams"],
            "Role": MODEL_ROLES["lstm_bestparams"],
            "OOS AUC": lstm["auc"],
            "OOS accuracy": lstm["accuracy"],
            "Balanced accuracy": lstm["balanced_accuracy"],
            "Strategy Sharpe": lstm["strategy_sharpe"],
            "Trade rate": lstm["trade_rate"],
            "Max drawdown": lstm["max_drawdown"],
        }
    )

    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    out["_order"] = out["model_key"].map(order)
    out = out.sort_values("_order").drop(columns=["_order", "model_key"])
    return out[OUTPUT_COLUMNS]


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ["OOS AUC", "OOS accuracy", "Balanced accuracy", "Strategy Sharpe", "Trade rate", "Max drawdown"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: "—" if pd.isna(x) else f"{float(x):.4f}")
    return display


def write_outputs(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    display = format_for_display(df)

    df.to_csv(outdir / "model_comparison_table_raw.csv", index=False)
    display.to_csv(outdir / "model_comparison_table.csv", index=False)
    display.to_markdown(outdir / "model_comparison_table.md", index=False)
    try:
        display.to_latex(
            outdir / "model_comparison_table.tex",
            index=False,
            escape=True,
            caption="Out-of-sample model comparison under walk-forward validation.",
            label="tab:model_comparison",
        )
    except Exception as exc:
        (outdir / "model_comparison_table_latex_warning.txt").write_text(str(exc), encoding="utf-8")

    make_table_png(display, outdir / "model_comparison_table.png")
    make_auc_plot(df, outdir / "model_comparison_auc.png")


def make_table_png(display: pd.DataFrame, path: Path) -> None:
    n_rows, n_cols = display.shape
    fig_width = 12.0
    fig_height = max(2.8, 0.55 * n_rows + 1.25)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    # Column widths tuned for thesis-style output.
    widths = {
        0: 0.18,
        1: 0.23,
        2: 0.09,
        3: 0.10,
        4: 0.11,
        5: 0.11,
        6: 0.08,
        7: 0.10,
    }
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.85")
        cell.set_linewidth(0.6)
        if col in widths:
            cell.set_width(widths[col])
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.92")
        elif row % 2 == 0:
            cell.set_facecolor("0.97")
        if col in [0, 1] and row > 0:
            cell.get_text().set_ha("left")

    ax.set_title(
        "Out-of-sample model comparison under walk-forward validation",
        fontsize=13,
        weight="bold",
        pad=16,
    )
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_auc_plot(df: pd.DataFrame, path: Path) -> None:
    plot_df = df[["Model", "OOS AUC"]].copy()
    plot_df["OOS AUC"] = pd.to_numeric(plot_df["OOS AUC"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(plot_df["Model"], plot_df["OOS AUC"])
    ax.axhline(0.5, linestyle="--", linewidth=1, label="Random AUC")
    ax.set_title("Out-of-sample AUC comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("OOS AUC")
    ax.set_ylim(0.45, max(0.58, float(plot_df["OOS AUC"].max()) + 0.02))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create combined LSTM and baseline comparison tables.")
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--feature-set", default="all_features")
    parser.add_argument("--lstm-summary", type=Path, default=None)
    parser.add_argument("--lstm-auc", type=float, default=None)
    parser.add_argument("--lstm-accuracy", type=float, default=None)
    parser.add_argument("--lstm-balanced-accuracy", type=float, default=None)
    parser.add_argument("--lstm-sharpe", type=float, default=None)
    parser.add_argument("--lstm-trade-rate", type=float, default=None)
    parser.add_argument("--lstm-max-drawdown", type=float, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = load_baseline_metrics(args.baseline_metrics, feature_set=args.feature_set)
    lstm = parse_lstm_metrics(args)
    table = build_comparison_table(baselines, lstm)
    write_outputs(table, args.outdir)

    print("Model comparison table complete.")
    print("CSV :", args.outdir / "model_comparison_table.csv")
    print("PNG :", args.outdir / "model_comparison_table.png")
    print("AUC :", args.outdir / "model_comparison_auc.png")


if __name__ == "__main__":
    main()
