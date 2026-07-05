#!/usr/bin/env python3
"""Create thesis-ready combined model-comparison tables.

Combines classical benchmark metrics with an LSTM walk-forward summary and exports
CSV, Markdown, LaTeX and PNG tables plus an AUC bar chart.

The LSTM summary parser accepts both historical walk-forward summaries with an
`overall` block and gross thesis report summaries with `classification` and
`trading` blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import fill
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

COMPACT_HEADERS = {
    "Model": "Model",
    "Role": "Role",
    "OOS AUC": "OOS\nAUC",
    "OOS accuracy": "OOS\nAcc.",
    "Balanced accuracy": "Bal.\nAcc.",
    "Strategy Sharpe": "Strategy\nSharpe",
    "Trade rate": "Trade\nRate",
    "Max drawdown": "Max\nDD",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "baseline_models_linear_svm_ablations"
    / "tables"
    / "baseline_model_metrics.csv"
)
DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "reports" / "model_comparison"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_value(mapping: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in mapping:
            val = as_float(mapping.get(key))
            if val is not None:
                return val
    return None


def update_if_missing(values: dict[str, float | None], key: str, candidate: float | None) -> None:
    if values.get(key) is None and candidate is not None:
        values[key] = candidate


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
                PROJECT_ROOT
                / "artifacts"
                / "models"
                / "walk_forward_direction_bestparams"
                / "walk_forward_summary_best_features.json",
                PROJECT_ROOT
                / "artifacts"
                / "models"
                / "walk_forward_direction_bestparams"
                / "walk_forward_summary.json",
                PROJECT_ROOT
                / "models"
                / "walk_forward_direction_bestparams"
                / "walk_forward_summary_best_features.json",
                PROJECT_ROOT
                / "models"
                / "walk_forward_direction_bestparams"
                / "walk_forward_summary.json",
            ]
        )

    if summary_path is not None and summary_path.exists():
        summary = read_json(summary_path)

        update_if_missing(values, "auc", first_value(summary, ["overall_oos_auc", "overall_auc", "oos_auc", "auc"]))
        update_if_missing(
            values,
            "accuracy",
            first_value(summary, ["overall_oos_acc", "overall_oos_accuracy", "overall_acc", "oos_acc", "accuracy", "acc"]),
        )
        update_if_missing(values, "balanced_accuracy", first_value(summary, ["balanced_accuracy", "overall_balanced_accuracy"]))
        update_if_missing(values, "strategy_sharpe", first_value(summary, ["strategy_sharpe", "oos_sharpe_long_only", "sharpe", "sharpe_long_only"]))
        update_if_missing(values, "trade_rate", first_value(summary, ["trade_rate", "oos_trade_rate"]))
        update_if_missing(values, "max_drawdown", first_value(summary, ["max_drawdown", "strategy_max_drawdown"]))

        # Historical walk-forward summary shape: {"overall": {"auc", "acc", "strategy": {...}}}
        overall = summary.get("overall", {})
        if isinstance(overall, dict):
            update_if_missing(values, "auc", first_value(overall, ["auc", "oos_auc", "overall_auc"]))
            update_if_missing(values, "accuracy", first_value(overall, ["acc", "accuracy", "oos_acc"]))
            update_if_missing(values, "balanced_accuracy", first_value(overall, ["balanced_accuracy"]))
            strategy = overall.get("strategy", {})
            if isinstance(strategy, dict):
                update_if_missing(values, "strategy_sharpe", first_value(strategy, ["sharpe_long_only", "sharpe", "strategy_sharpe"]))
                update_if_missing(values, "trade_rate", first_value(strategy, ["trade_rate_long_only", "trade_rate"]))
                update_if_missing(values, "max_drawdown", first_value(strategy, ["max_drawdown_long_only", "max_drawdown"]))

        # Gross thesis report shape: {"classification": {...}, "trading": {...}}
        classification = summary.get("classification", {})
        if isinstance(classification, dict):
            update_if_missing(values, "auc", first_value(classification, ["oos_auc", "auc"]))
            update_if_missing(values, "accuracy", first_value(classification, ["oos_acc", "accuracy", "acc"]))
            update_if_missing(values, "trade_rate", first_value(classification, ["trade_rate"]))
            update_if_missing(values, "balanced_accuracy", first_value(classification, ["balanced_accuracy"]))

        trading = summary.get("trading", {})
        if isinstance(trading, dict):
            update_if_missing(values, "strategy_sharpe", first_value(trading, ["annualized_sharpe", "sharpe", "strategy_sharpe"]))
            update_if_missing(values, "max_drawdown", first_value(trading, ["max_drawdown"]))

        strategy = summary.get("strategy") or summary.get("strategy_metrics") or {}
        if isinstance(strategy, dict):
            update_if_missing(values, "strategy_sharpe", first_value(strategy, ["sharpe", "strategy_sharpe", "oos_sharpe_long_only", "sharpe_long_only"]))
            update_if_missing(values, "trade_rate", first_value(strategy, ["trade_rate", "oos_trade_rate", "trade_rate_long_only"]))
            update_if_missing(values, "max_drawdown", first_value(strategy, ["max_drawdown", "strategy_max_drawdown"]))

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
    for col in [
        "OOS AUC",
        "OOS accuracy",
        "Balanced accuracy",
        "Strategy Sharpe",
        "Trade rate",
        "Max drawdown",
    ]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: "—" if pd.isna(x) else f"{float(x):.4f}")
    return display


def wrap_text_columns(display: pd.DataFrame) -> pd.DataFrame:
    wrapped = display.copy()
    if "Model" in wrapped.columns:
        wrapped["Model"] = wrapped["Model"].replace({"LSTM (best specification)": "LSTM\n(best spec.)"})
    if "Role" in wrapped.columns:
        wrapped["Role"] = wrapped["Role"].apply(lambda x: fill(str(x), width=18))
    return wrapped


def compact_headers(display: pd.DataFrame) -> pd.DataFrame:
    return display.rename(columns=COMPACT_HEADERS)


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

    compact = compact_headers(wrap_text_columns(display))
    make_table_png(
        compact,
        outdir / "model_comparison_table.png",
        title="Out-of-sample model comparison under walk-forward validation",
        widths=[0.18, 0.20, 0.09, 0.09, 0.10, 0.11, 0.09, 0.09],
        fig_width=11.2,
        fig_height=3.4,
        fontsize=8.5,
        header_fontsize=8.8,
    )

    classification = compact_headers(wrap_text_columns(display[["Model", "Role", "OOS AUC", "OOS accuracy", "Balanced accuracy"]]))
    make_table_png(
        classification,
        outdir / "model_comparison_classification_table.png",
        title="Classification performance under walk-forward validation",
        widths=[0.24, 0.28, 0.14, 0.14, 0.14],
        fig_width=8.8,
        fig_height=3.2,
        fontsize=9.0,
        header_fontsize=9.2,
    )

    trading = compact_headers(wrap_text_columns(display[["Model", "Strategy Sharpe", "Trade rate", "Max drawdown"]]))
    make_table_png(
        trading,
        outdir / "model_comparison_trading_table.png",
        title="Trading-oriented performance metrics",
        widths=[0.34, 0.19, 0.18, 0.18],
        fig_width=7.4,
        fig_height=3.0,
        fontsize=9.0,
        header_fontsize=9.2,
    )

    make_auc_plot(df, outdir / "model_comparison_auc.png")


def make_table_png(
    display: pd.DataFrame,
    path: Path,
    title: str,
    widths: list[float],
    fig_width: float,
    fig_height: float,
    fontsize: float,
    header_fontsize: float,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.02, 0.03, 0.96, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        cell.set_linewidth(0.55)
        if col < len(widths):
            cell.set_width(widths[col])
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=header_fontsize)
            cell.set_facecolor("0.90")
        elif row % 2 == 0:
            cell.set_facecolor("0.97")
        else:
            cell.set_facecolor("1.00")
        if col in [0, 1] and row > 0:
            cell.get_text().set_ha("left")

    ax.set_title(title, fontsize=11.5, weight="bold", pad=8)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_auc_plot(df: pd.DataFrame, path: Path) -> None:
    plot_df = df[["Model", "OOS AUC"]].copy()
    plot_df["Model"] = plot_df["Model"].replace({"LSTM (best specification)": "LSTM"})
    plot_df["OOS AUC"] = pd.to_numeric(plot_df["OOS AUC"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(plot_df["Model"], plot_df["OOS AUC"])
    ax.axhline(0.5, linestyle="--", linewidth=1, label="Random AUC")
    ax.set_title("Out-of-sample AUC comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("OOS AUC")
    ax.set_ylim(0.45, max(0.58, float(plot_df["OOS AUC"].max()) + 0.02))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
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
    print("CLS :", args.outdir / "model_comparison_classification_table.png")
    print("TRD :", args.outdir / "model_comparison_trading_table.png")
    print("AUC :", args.outdir / "model_comparison_auc.png")


if __name__ == "__main__":
    main()
