#!/usr/bin/env python3
"""Prepare final thesis outputs for committing to GitHub.

This script does three things:
1. writes compact dataset-statistics reports, correlations and target correlations;
2. rebuilds the LSTM feature-ablation summary from the thesis report JSON files;
3. creates a final manifest and optionally moves non-core analyses to artifacts/legacy.

It intentionally does not commit raw datasets or secrets. The generated reports are small
CSV/Markdown/PNG files that are suitable as thesis evidence in the private repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


FEATURE_GROUPS = {
    "market": [
        "log_return",
        "overnight_return",
        "momentum_5d",
        "momentum_20d",
        "volatility_20d",
        "volume_change",
        "volume_20d_avg",
    ],
    "macro_sector_bond": ["spy_return", "soxx_return", "ief_return"],
    "sentiment": ["avg_sentiment", "sentiment_std", "news_count"],
    "attention": ["trends_zscore_30d", "trends_momentum_7d", "trends_spike"],
}

LSTM_ABLATION_FEATURE_COUNTS = {
    "market_only": 10,
    "market_sentiment": 13,
    "market_attention": 13,
    "full_model": 16,
}

CORE_RESULT_PATTERNS = [
    "artifacts/reports/{symbol}_full_pipeline_summary.csv",
    "artifacts/reports/{symbol}_model_comparison/model_comparison_table.csv",
    "artifacts/reports/{symbol}_model_comparison/model_comparison_auc.png",
    "artifacts/reports/{symbol}_model_comparison/model_comparison_classification_table.png",
    "artifacts/reports/{symbol}_model_comparison/model_comparison_trading_table.png",
    "artifacts/reports/{symbol}_baseline_models_linear_svm_ablations/tables/baseline_model_metrics.csv",
    "artifacts/reports/{symbol}_baseline_models_linear_svm_ablations/tables/baseline_fold_metrics.csv",
    "artifacts/reports/{symbol}_baseline_models_linear_svm_ablations/figures/feature_set_ablation_auc.png",
    "artifacts/models/{symbol}_walk_forward_random_search_bestparams/walk_forward_summary.json",
    "artifacts/models/{symbol}_walk_forward_random_search_bestparams/thesis_report_gross/report_summary.json",
    "artifacts/models/{symbol}_walk_forward_random_search_bestparams/thesis_report_gross/figures/roc_curve.png",
    "artifacts/models/{symbol}_walk_forward_random_search_bestparams/thesis_report_gross/figures/equity_drawdown.png",
    "artifacts/models/{symbol}_lstm_feature_ablation/lstm_feature_ablation_summary.csv",
    "artifacts/models/{symbol}_lstm_feature_ablation/lstm_feature_ablation_summary_fixed.csv",
]

LEGACY_PATTERNS = [
    "artifacts/models/{symbol}_walk_forward_legacy_05506_params",
    "artifacts/models/{symbol}_walk_forward_reduced_fixedparams",
    "artifacts/models/{symbol}_walk_forward_full_fixedparams",
    "artifacts/reports/{symbol}_baseline_models",
    "artifacts/reports/{symbol}_old*",
    "artifacts/reports/{symbol}_debug*",
    "artifacts/models/{symbol}_old*",
    "artifacts/models/{symbol}_debug*",
]


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[1]


def write_table(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    try:
        df.to_markdown(stem.with_suffix(".md"), index=False)
    except Exception:
        stem.with_suffix(".md").write_text(df.to_string(index=False), encoding="utf-8")


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}")
        return None


def dataset_path_for_symbol(root: Path, symbol: str) -> Path:
    symbol = symbol.lower()
    candidates = [
        root / f"data/model_feed/{symbol}_model_dataset_clean.csv",
        root / "data/model_feed/model_dataset_clean.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"date", "target_direction", "target_next_return"}
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def plot_correlation_heatmap(corr: pd.DataFrame, out_png: Path) -> None:
    if plt is None or corr.empty:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Feature correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_dataset_statistics(root: Path, symbol: str, high_corr_threshold: float) -> dict[str, Any]:
    symbol = symbol.lower()
    dataset_path = dataset_path_for_symbol(root, symbol)
    outdir = root / f"artifacts/reports/{symbol}_dataset_statistics"
    outdir.mkdir(parents=True, exist_ok=True)

    df = safe_read_csv(dataset_path)
    if df is None:
        return {"symbol": symbol.upper(), "dataset": str(dataset_path), "status": "missing"}

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    feature_cols = numeric_feature_columns(df)
    num = df[feature_cols].apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame()

    overview_rows: list[dict[str, Any]] = [
        {"metric": "dataset_path", "value": str(dataset_path)},
        {"metric": "rows", "value": len(df)},
        {"metric": "columns", "value": len(df.columns)},
        {"metric": "numeric_features", "value": len(feature_cols)},
        {"metric": "duplicate_dates", "value": int(df["date"].duplicated().sum()) if "date" in df.columns else "n/a"},
        {"metric": "total_missing_values", "value": int(df.isna().sum().sum())},
    ]
    if "date" in df.columns:
        overview_rows.extend(
            [
                {"metric": "date_min", "value": str(df["date"].min().date()) if df["date"].notna().any() else "n/a"},
                {"metric": "date_max", "value": str(df["date"].max().date()) if df["date"].notna().any() else "n/a"},
            ]
        )
    if "target_direction" in df.columns:
        counts = df["target_direction"].value_counts(dropna=False).sort_index()
        pcts = df["target_direction"].value_counts(normalize=True, dropna=False).sort_index()
        for cls in counts.index:
            overview_rows.append({"metric": f"target_count_{cls}", "value": int(counts.loc[cls])})
            overview_rows.append({"metric": f"target_pct_{cls}", "value": float(pcts.loc[cls])})
    overview = pd.DataFrame(overview_rows)
    write_table(overview, outdir / "dataset_overview")

    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing["missing_pct"] = missing["missing_count"] / max(len(df), 1)
    missing = missing.sort_values(["missing_count", "column"], ascending=[False, True])
    write_table(missing, outdir / "missing_values")

    if not num.empty:
        desc = num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.reset_index()
        desc = desc.rename(columns={"index": "feature"})
        write_table(desc, outdir / "descriptive_statistics")

        corr = num.corr()
        corr.to_csv(outdir / "correlation_matrix.csv")
        try:
            corr.to_markdown(outdir / "correlation_matrix.md")
        except Exception:
            (outdir / "correlation_matrix.md").write_text(corr.to_string(), encoding="utf-8")
        plot_correlation_heatmap(corr, outdir / "correlation_heatmap.png")

        pairs = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                value = corr.loc[a, b]
                if pd.notna(value) and abs(value) >= high_corr_threshold:
                    pairs.append({"feature_1": a, "feature_2": b, "correlation": float(value), "abs_correlation": float(abs(value))})
        high_corr = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False) if pairs else pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation"])
        write_table(high_corr, outdir / "high_correlations")

        target_corr_rows = []
        for target in ["target_direction", "target_next_return"]:
            if target in df.columns:
                y = pd.to_numeric(df[target], errors="coerce")
                for feature in feature_cols:
                    x = pd.to_numeric(df[feature], errors="coerce")
                    c = x.corr(y)
                    target_corr_rows.append({"target": target, "feature": feature, "correlation": c, "abs_correlation": abs(c) if pd.notna(c) else np.nan})
        target_corr = pd.DataFrame(target_corr_rows)
        if not target_corr.empty:
            target_corr = target_corr.sort_values(["target", "abs_correlation"], ascending=[True, False])
        write_table(target_corr, outdir / "target_correlations")

    group_rows = []
    for group, cols in FEATURE_GROUPS.items():
        existing = [c for c in cols if c in df.columns]
        group_rows.append({"group": group, "expected_features": len(cols), "present_features": len(existing), "features": ", ".join(existing)})
    write_table(pd.DataFrame(group_rows), outdir / "feature_group_summary")

    return {"symbol": symbol.upper(), "dataset": str(dataset_path), "status": "ok", "outdir": str(outdir)}


def parse_lstm_report(report: Path) -> dict[str, Any] | None:
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    classification = data.get("classification", {})
    trading = data.get("trading", {})
    overall = data.get("overall", {})
    strategy = overall.get("strategy", {}) if isinstance(overall, dict) else {}
    return {
        "auc": classification.get("oos_auc", classification.get("auc", overall.get("auc"))),
        "accuracy": classification.get("oos_acc", classification.get("accuracy", classification.get("acc", overall.get("acc")))),
        "sharpe": trading.get("annualized_sharpe", trading.get("sharpe", trading.get("strategy_sharpe", strategy.get("sharpe_long_only")))),
        "trade_rate": classification.get("trade_rate", trading.get("trade_rate", strategy.get("trade_rate_long_only"))),
        "max_drawdown": trading.get("max_drawdown"),
        "mean_daily_return": trading.get("mean_daily_return", strategy.get("avg_daily_return_long_only")),
        "num_trades": trading.get("num_trades"),
    }


def rebuild_lstm_ablation_summary(root: Path, symbol: str) -> dict[str, Any]:
    symbol = symbol.lower()
    base = root / f"artifacts/models/{symbol}_lstm_feature_ablation"
    if not base.exists():
        return {"symbol": symbol.upper(), "status": "missing", "path": str(base)}
    rows = []
    for feature_set, feature_count in LSTM_ABLATION_FEATURE_COUNTS.items():
        report = base / feature_set / "thesis_report_gross" / "report_summary.json"
        parsed = parse_lstm_report(report)
        if parsed is None:
            rows.append({"feature_set": feature_set, "feature_count": feature_count, "status": "missing", "report": str(report)})
        else:
            rows.append({"feature_set": feature_set, "feature_count": feature_count, "status": "ok", **parsed, "report": str(report)})
    df = pd.DataFrame(rows)
    out = base / "lstm_feature_ablation_summary.csv"
    df.to_csv(out, index=False)
    try:
        df.to_markdown(base / "lstm_feature_ablation_summary.md", index=False)
    except Exception:
        (base / "lstm_feature_ablation_summary.md").write_text(df.to_string(index=False), encoding="utf-8")
    return {"symbol": symbol.upper(), "status": "ok", "path": str(out)}


def existing_core_results(root: Path, symbol: str) -> list[Path]:
    symbol = symbol.lower()
    paths = []
    for pattern in CORE_RESULT_PATTERNS:
        path = root / pattern.format(symbol=symbol)
        if path.exists():
            paths.append(path)
    return paths


def move_legacy(root: Path, symbols: list[str], dry_run: bool) -> list[dict[str, str]]:
    legacy_root = root / "artifacts/legacy"
    moves: list[dict[str, str]] = []
    for symbol in [s.lower() for s in symbols]:
        for pattern in LEGACY_PATTERNS:
            for src in root.glob(pattern.format(symbol=symbol)):
                if not src.exists():
                    continue
                if legacy_root in src.parents:
                    continue
                rel = src.relative_to(root)
                dst = legacy_root / rel
                moves.append({"source": str(rel), "destination": str(dst.relative_to(root)), "action": "dry_run" if dry_run else "moved"})
                if not dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dst = dst.with_name(f"{dst.name}_{stamp}")
                    shutil.move(str(src), str(dst))
    if moves and not dry_run:
        legacy_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(moves).to_csv(legacy_root / "legacy_moves.csv", index=False)
        lines = ["# Legacy artifacts", "", "Generated by `scripts/prepare_results_for_github.py --move-legacy`.", ""]
        lines += [f"- `{m['source']}` -> `{m['destination']}`" for m in moves]
        (legacy_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return moves


def write_manifest(root: Path, symbols: list[str], dataset_reports: list[dict[str, Any]], ablation_reports: list[dict[str, Any]], legacy_moves: list[dict[str, str]]) -> Path:
    out = root / "artifacts/reports/final_results_manifest.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final thesis results manifest",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This file lists the compact results and statistical reports that should be kept in GitHub. Raw data, API credentials and trained model binaries should remain local.",
        "",
    ]
    for symbol in [s.lower() for s in symbols]:
        lines.append(f"## {symbol.upper()}")
        lines.append("")
        lines.append("### Core model results")
        core = existing_core_results(root, symbol)
        if core:
            for path in core:
                lines.append(f"- `{path.relative_to(root)}`")
        else:
            lines.append("- No core result files found yet.")
        lines.append("")
        lines.append("### Dataset statistics")
        stat_dir = root / f"artifacts/reports/{symbol}_dataset_statistics"
        for name in [
            "dataset_overview.csv",
            "descriptive_statistics.csv",
            "correlation_matrix.csv",
            "correlation_heatmap.png",
            "target_correlations.csv",
            "high_correlations.csv",
            "feature_group_summary.csv",
        ]:
            path = stat_dir / name
            if path.exists():
                lines.append(f"- `{path.relative_to(root)}`")
        lines.append("")
    lines.append("## Preparation status")
    lines.append("")
    lines.append("### Dataset reports")
    for item in dataset_reports:
        lines.append(f"- {item['symbol']}: {item['status']} — `{item.get('outdir', item.get('dataset', ''))}`")
    lines.append("")
    lines.append("### LSTM ablation summaries")
    for item in ablation_reports:
        lines.append(f"- {item['symbol']}: {item['status']} — `{item.get('path', '')}`")
    lines.append("")
    if legacy_moves:
        lines.append("### Legacy moves")
        for m in legacy_moves:
            lines.append(f"- {m['action']}: `{m['source']}` -> `{m['destination']}`")
        lines.append("")
    lines.append("## Suggested Git commands")
    lines.append("")
    lines.append("```bash")
    lines.append("git status --short")
    lines.append("git add artifacts/reports artifacts/models artifacts/legacy docs scripts src tests")
    lines.append("git commit -m \"Add final thesis results and dataset statistics\"")
    lines.append("git push origin main_v2")
    lines.append("```")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare final thesis result artifacts and dataset reports for GitHub.")
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "AMD"], help="Ticker symbols to process.")
    parser.add_argument("--high-corr-threshold", type=float, default=0.80)
    parser.add_argument("--move-legacy", action="store_true", help="Move known old/debug/fixed/legacy analysis folders to artifacts/legacy.")
    parser.add_argument("--dry-run", action="store_true", help="Show legacy moves without moving files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    symbols = [s.upper() for s in args.symbols]
    print("Root:", root)
    print("Symbols:", ", ".join(symbols))

    dataset_reports = [write_dataset_statistics(root, s, args.high_corr_threshold) for s in symbols]
    ablation_reports = [rebuild_lstm_ablation_summary(root, s) for s in symbols]
    legacy_moves = move_legacy(root, symbols, dry_run=args.dry_run) if args.move_legacy or args.dry_run else []
    manifest = write_manifest(root, symbols, dataset_reports, ablation_reports, legacy_moves)

    print("\n=== PREPARE RESULTS DONE ===")
    print("Manifest:", manifest.relative_to(root))
    for report in dataset_reports:
        print(f"Dataset stats {report['symbol']}: {report['status']} -> {report.get('outdir', report.get('dataset'))}")
    for report in ablation_reports:
        print(f"LSTM ablation {report['symbol']}: {report['status']} -> {report.get('path')}")
    if legacy_moves:
        print("Legacy moves:", len(legacy_moves))


if __name__ == "__main__":
    main()
