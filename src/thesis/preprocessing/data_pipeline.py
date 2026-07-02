#!/usr/bin/env python3
"""Canonical preprocessing pipeline for the thesis dataset.

This module turns raw/intermediate source files into the modelling dataset used
by the LSTM and benchmark evaluation scripts.

It consolidates the earlier preprocessing scripts into one documented CLI:
- raw news CSV combination
- trading-day news alignment
- daily VADER sentiment aggregation
- EOD stock and ETF feature engineering
- Google Trends daily-series reconstruction and feature engineering
- final model-dataset merge
- validation and Excel audit output

The cleaned model dataset expected by the evaluation scripts is:
    data/model_feed/model_dataset_clean.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_DIR = PROJECT_ROOT / "data"


def paths(base_dir: Path = DEFAULT_BASE_DIR) -> dict[str, Path]:
    """Return standard project paths for a given data directory."""
    base_dir = Path(base_dir)
    return {
        "raw_news_dir": base_dir / "raw" / "news_headlines",
        "news_master": base_dir / "interim" / "news_headlines_master.csv",
        "news_clean": base_dir / "processed" / "news_headlines_clean.csv",
        "news_sentiment": base_dir / "processed" / "news_daily_sentiment.csv",
        "trends_monthly": base_dir / "raw" / "trends" / "multiTimeline.csv",
        "trends_daily_glob": base_dir / "raw" / "trends" / "multiTimeline*.csv",
        "trends_consistent": base_dir / "interim" / "nvidia_trends_daily_consistent.csv",
        "trends_processed": base_dir / "processed" / "nvidia_trends_processed.csv",
        "model_dataset": base_dir / "model_feed" / "model_dataset.csv",
        "model_dataset_clean": base_dir / "model_feed" / "model_dataset_clean.csv",
        "model_audit": base_dir / "model_feed" / "model_dataset_audit.xlsx",
        "nvda_raw": base_dir
        / "raw"
        / "stock_data"
        / "NVDA"
        / "NVDA_eod_chunked_2019-03-01_to_2026-03-01.csv",
        "spy_raw": base_dir
        / "raw"
        / "macro_stock_data"
        / "SPY"
        / "SPY_eod_chunked_2019-03-01_to_2026-03-01.csv",
        "soxx_raw": base_dir
        / "raw"
        / "macro_stock_data"
        / "SOXX"
        / "SOXX_eod_chunked_2019-03-01_to_2026-03-01.csv",
        "ief_raw": base_dir
        / "raw"
        / "macro_stock_data"
        / "IEF"
        / "IEF_eod_chunked_2019-03-01_to_2026-03-01.csv",
        "nvda_processed": base_dir / "processed" / "NVDA_eod_processed.csv",
        "spy_processed": base_dir / "processed" / "SPY_eod_processed.csv",
        "soxx_processed": base_dir / "processed" / "SOXX_eod_processed.csv",
        "ief_processed": base_dir / "processed" / "IEF_eod_processed.csv",
    }


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def standardize_columns(cols: Iterable[object]) -> pd.Index:
    return (
        pd.Index(cols)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )


def print_done(message: str, path: str | Path | None = None) -> None:
    print(f"OK - {message}")
    if path is not None:
        print(f"Saved to: {path}")


# ---------------------------------------------------------------------------
# News preprocessing
# ---------------------------------------------------------------------------
def combine_all_news_csvs(
    folder_path: str | Path,
    output_path: str | Path,
    recursive: bool = False,
) -> pd.DataFrame:
    """Combine monthly/raw news CSV files into one master CSV."""
    folder_path = Path(folder_path)
    output_path = Path(output_path)
    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(glob.glob(str(folder_path / pattern), recursive=recursive))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    all_cols: list[pd.Index] = []
    for fp in files:
        header_df = pd.read_csv(fp, nrows=0)
        all_cols.append(standardize_columns(header_df.columns))

    union_cols = pd.Index([])
    for cols in all_cols:
        union_cols = union_cols.union(cols)

    preferred_order = [
        "symbol_target",
        "uuid",
        "published_at",
        "date",
        "title",
        "description",
        "keywords",
        "snippet",
        "url",
        "source",
        "image_url",
        "entities_json",
        "raw_json",
    ]
    preferred_present = [c for c in preferred_order if c in union_cols]
    remaining = sorted([c for c in union_cols if c not in preferred_present])
    canonical_cols = preferred_present + remaining + ["source_file"]

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        df.columns = standardize_columns(df.columns)
        df = df.reindex(columns=canonical_cols)
        df["source_file"] = os.path.basename(fp)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    if "published_at" in combined.columns:
        combined["published_at"] = pd.to_datetime(combined["published_at"], errors="coerce", utc=True)

    if "uuid" in combined.columns:
        combined = combined.drop_duplicates(subset=["uuid"], keep="first")
    else:
        combined = combined.drop_duplicates()

    combined = combined.reset_index(drop=True)
    ensure_parent(output_path)
    combined.to_csv(output_path, index=False)
    return combined


def clean_news_dataset(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Clean news records and align publication timing to US trading days."""
    df = pd.read_csv(input_path)
    df.columns = standardize_columns(df.columns)

    required_cols = ["symbol_target", "published_at", "title", "description", "keywords", "source"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy()
    for col in ["title", "description", "keywords", "source"]:
        df[col] = df[col].where(df[col].notna(), None)
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": None, "nan": None, "None": None})

    df = df[df["title"].notna()].copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df[df["published_at"].notna()].copy()

    # Economic timing: news after the US close is assigned to the next trading day.
    df["published_et"] = df["published_at"].dt.tz_convert("America/New_York")
    df = df[df["published_et"].dt.weekday < 5].copy()
    df["trading_date"] = df["published_et"].dt.floor("D")
    after_close_mask = df["published_et"].dt.hour >= 16
    df.loc[after_close_mask, "trading_date"] = df.loc[after_close_mask, "trading_date"] + pd.Timedelta(days=1)

    saturday_mask = df["trading_date"].dt.weekday == 5
    sunday_mask = df["trading_date"].dt.weekday == 6
    df.loc[saturday_mask, "trading_date"] = df.loc[saturday_mask, "trading_date"] + pd.Timedelta(days=2)
    df.loc[sunday_mask, "trading_date"] = df.loc[sunday_mask, "trading_date"] + pd.Timedelta(days=1)
    df["trading_date"] = df["trading_date"].dt.date

    df["date"] = df["published_at"].dt.date
    df["year"] = df["published_at"].dt.year
    df["month"] = df["published_at"].dt.month
    df["day"] = df["published_at"].dt.day
    df["hour"] = df["published_at"].dt.hour
    df["weekday"] = df["published_at"].dt.weekday
    df["is_weekend"] = df["weekday"] >= 5

    df = df.sort_values("published_at").reset_index(drop=True)
    ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df


def build_daily_sentiment(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Aggregate article-level VADER sentiment to trading-day features."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise ImportError("Install nltk first: python -m pip install nltk") from exc

    try:
        sid = SentimentIntensityAnalyzer()
    except LookupError:
        import nltk

        nltk.download("vader_lexicon")
        sid = SentimentIntensityAnalyzer()

    df = pd.read_csv(input_path)
    if "trading_date" not in df.columns:
        raise ValueError("trading_date column missing. Run news-clean first.")

    df["text_for_sentiment"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    df["sentiment"] = df["text_for_sentiment"].apply(lambda x: sid.polarity_scores(str(x))["compound"])

    daily = (
        df.groupby("trading_date")
        .agg(
            avg_sentiment=("sentiment", "mean"),
            median_sentiment=("sentiment", "median"),
            sentiment_std=("sentiment", "std"),
            news_count=("sentiment", "count"),
            positive_ratio=("sentiment", lambda x: np.mean(x > 0)),
            negative_ratio=("sentiment", lambda x: np.mean(x < 0)),
        )
        .reset_index()
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    daily = daily.sort_values("trading_date").reset_index(drop=True)

    ensure_parent(output_path)
    daily.to_csv(output_path, index=False)
    return daily


# ---------------------------------------------------------------------------
# Stock preprocessing
# ---------------------------------------------------------------------------
def clean_and_process_eod(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Clean one EOD OHLCV file and engineer return/volume features."""
    df = pd.read_csv(input_path)
    df.columns = standardize_columns(df.columns)

    required_price_cols = ["date", "open", "high", "low", "close"]
    missing = [c for c in required_price_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
    df["overnight_return"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    df["volatility_20d"] = df["log_return"].rolling(20).std()
    df["momentum_5d"] = df["log_return"].rolling(5).mean()
    df["momentum_20d"] = df["log_return"].rolling(20).mean()

    if "volume" in df.columns:
        df["volume_change"] = df["volume"].pct_change()
        df["volume_20d_avg"] = df["volume"].rolling(20).mean()

    df["target_next_return"] = df["log_return"].shift(-1)
    df["target_direction"] = (df["target_next_return"] > 0).astype(int)

    df = df.dropna().reset_index(drop=True)
    ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Google Trends preprocessing
# ---------------------------------------------------------------------------
def read_google_trends_csv(path: str | Path) -> pd.DataFrame:
    """Read a Google Trends export with variable Dutch/English header rows."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Dag,") or line.startswith("Maand,") or line.startswith("Day,") or line.startswith("Month,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find Google Trends header row in {path}")

    df = pd.read_csv(path, skiprows=header_idx)
    date_col, value_col = df.columns[:2]
    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = df["value"].replace({"<1": 0}).astype(float)
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df[["date", "value"]]


def reconstruct_google_trends(
    monthly_file: str | Path,
    daily_glob: str,
    output_path: str | Path,
) -> pd.DataFrame:
    """Reconstruct a consistent daily Google Trends series from monthly and daily exports."""
    monthly = read_google_trends_csv(monthly_file)
    monthly["month"] = monthly["date"].values.astype("datetime64[M]")
    monthly_ref = (
        monthly.groupby("month", as_index=False)["value"].mean().rename(columns={"value": "ref_value"})
    )

    daily_files = sorted([p for p in glob.glob(daily_glob) if Path(p).name != Path(monthly_file).name])
    if not daily_files:
        raise FileNotFoundError(f"No daily Google Trends chunk files found: {daily_glob}")

    daily_chunks = [read_google_trends_csv(p) for p in daily_files]

    def compute_monthly_scale(daily_df: pd.DataFrame) -> float:
        tmp = daily_df.copy()
        tmp["month"] = tmp["date"].values.astype("datetime64[M]")
        chunk_monthly = tmp.groupby("month", as_index=False)["value"].mean()
        merged = chunk_monthly.merge(monthly_ref, on="month", how="inner")
        if merged.empty:
            return 1.0
        ratios = merged["ref_value"] / merged["value"].replace(0, np.nan)
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
        return float(ratios.median()) if len(ratios) else 1.0

    scaled_chunks = []
    for chunk in daily_chunks:
        out = chunk.copy()
        out["value"] = out["value"] * compute_monthly_scale(chunk)
        scaled_chunks.append(out)

    # Fine chain-linking at chunk boundaries to reduce artificial jumps.
    for i in range(1, len(scaled_chunks)):
        prev = scaled_chunks[i - 1]
        curr = scaled_chunks[i]
        v_prev = float(prev.loc[prev["date"] == prev["date"].max(), "value"].iloc[0])
        v_curr = float(curr.loc[curr["date"] == curr["date"].min(), "value"].iloc[0])
        if abs(v_curr) > 1e-9:
            scaled_chunks[i]["value"] = scaled_chunks[i]["value"] * (v_prev / v_curr)

    daily_scaled = pd.concat(scaled_chunks, ignore_index=True).sort_values("date")
    daily_scaled = daily_scaled.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    daily_series = daily_scaled.set_index("date").asfreq("D")
    daily_series["value"] = daily_series["value"].interpolate(limit_direction="both")
    daily_series = daily_series.reset_index()

    ensure_parent(output_path)
    daily_series.to_csv(output_path, index=False)
    return daily_series


def clean_and_process_trends(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Engineer attention variables from the reconstructed daily Trends series."""
    df = pd.read_csv(input_path)
    df.columns = standardize_columns(df.columns)
    if "date" not in df.columns:
        raise ValueError("No 'date' column found in trends data.")

    value_cols = [c for c in df.columns if c != "date"]
    if len(value_cols) != 1:
        raise ValueError("Unexpected trends structure. Should contain one value column plus date.")

    trend_col = value_cols[0]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)
    df = df.rename(columns={trend_col: "trends_raw"})
    df["trends_raw"] = pd.to_numeric(df["trends_raw"], errors="coerce")

    df["trends_7d_ma"] = df["trends_raw"].rolling(7).mean()
    df["trends_momentum_7d"] = df["trends_raw"].pct_change(7)
    rolling_mean = df["trends_raw"].rolling(30).mean()
    rolling_std = df["trends_raw"].rolling(30).std()
    df["trends_zscore_30d"] = (df["trends_raw"] - rolling_mean) / rolling_std
    df["trends_spike"] = (df["trends_zscore_30d"] > 2).astype(int)

    df = df.dropna().reset_index(drop=True)
    ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Final dataset construction and validation
# ---------------------------------------------------------------------------
def build_model_dataset(
    nvda_path: str | Path,
    spy_path: str | Path,
    soxx_path: str | Path,
    ief_path: str | Path,
    sent_path: str | Path,
    trends_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Merge processed stock, macro, sentiment and attention features."""
    nvda = pd.read_csv(nvda_path)
    spy = pd.read_csv(spy_path)
    soxx = pd.read_csv(soxx_path)
    ief = pd.read_csv(ief_path)
    sent = pd.read_csv(sent_path)
    trends = pd.read_csv(trends_path)

    for frame in [nvda, spy, soxx, ief, trends]:
        frame["date"] = pd.to_datetime(frame["date"])
    sent["trading_date"] = pd.to_datetime(sent["trading_date"])

    spy = spy[["date", "log_return"]].rename(columns={"log_return": "spy_return"})
    soxx = soxx[["date", "log_return"]].rename(columns={"log_return": "soxx_return"})
    ief = ief[["date", "log_return"]].rename(columns={"log_return": "ief_return"})

    df = nvda.merge(spy, on="date", how="left")
    df = df.merge(soxx, on="date", how="left")
    df = df.merge(ief, on="date", how="left")
    df = df.merge(sent, left_on="date", right_on="trading_date", how="left")
    df = df.drop(columns=["trading_date"], errors="ignore")
    df = df.merge(trends, on="date", how="left")
    df = df.sort_values("date")

    fill_cols = [
        "avg_sentiment",
        "median_sentiment",
        "sentiment_std",
        "news_count",
        "positive_ratio",
        "negative_ratio",
        "trends_raw",
        "trends_7d_ma",
        "trends_momentum_7d",
        "trends_zscore_30d",
        "trends_spike",
    ]
    existing_fill_cols = [c for c in fill_cols if c in df.columns]
    df[existing_fill_cols] = df[existing_fill_cols].ffill()

    df = df.dropna().reset_index(drop=True)
    ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df


def write_clean_model_dataset(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Create the compact modelling dataset used by the final evaluation scripts."""
    df = pd.read_csv(input_path)
    keep_cols = [
        "date",
        "target_next_return",
        "target_direction",
        "log_return",
        "overnight_return",
        "momentum_5d",
        "momentum_20d",
        "volatility_20d",
        "volume_change",
        "volume_20d_avg",
        "spy_return",
        "soxx_return",
        "ief_return",
        "avg_sentiment",
        "sentiment_std",
        "news_count",
        "trends_zscore_30d",
        "trends_momentum_7d",
        "trends_spike",
    ]
    keep = [c for c in keep_cols if c in df.columns]
    clean = df[keep].dropna().reset_index(drop=True)
    ensure_parent(output_path)
    clean.to_csv(output_path, index=False)
    return clean


def dataset_validation_tables(data_path: str | Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    basic_info = pd.DataFrame(
        {
            "metric": ["rows", "columns", "date_min", "date_max"],
            "value": [len(df), len(df.columns), df["date"].min(), df["date"].max()],
        }
    )
    dup_summary = pd.DataFrame(
        {
            "duplicate_dates_total": [int(df["date"].duplicated().sum())],
            "unique_dates_duplicated": [int(df.loc[df["date"].duplicated(), "date"].nunique())],
        }
    )
    date_diffs = df["date"].diff().dt.days
    large_gaps = pd.DataFrame({"date": df["date"], "gap_days_since_prev": date_diffs})
    large_gaps = large_gaps[large_gaps["gap_days_since_prev"] > 3].copy()
    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum().sort_values(ascending=False),
            "missing_pct": (df.isna().mean() * 100).round(2).sort_values(ascending=False),
        }
    )
    missing = missing[missing["missing_count"] > 0]
    if missing.empty:
        missing = pd.DataFrame({"note": ["No missing values detected"]})

    zero_var = df.nunique(dropna=False)
    zero_var = zero_var[zero_var <= 1].sort_values().to_frame("n_unique_values")
    if zero_var.empty:
        zero_var = pd.DataFrame({"note": ["No zero-variance columns detected"]})

    corr_target = pd.DataFrame({"note": ["Column target_next_return not found"]})
    if "target_next_return" in df.columns:
        corr = df.corr(numeric_only=True)
        corr_target = corr["target_next_return"].sort_values(ascending=False).to_frame(
            "corr_with_target_next_return"
        )

    return {
        "basic_info": basic_info,
        "dup_summary": dup_summary,
        "large_gaps": large_gaps,
        "missing": missing,
        "zero_variance": zero_var,
        "corr_target": corr_target,
    }


def validate_dataset(data_path: str | Path) -> None:
    tables = dataset_validation_tables(data_path)
    for name, table in tables.items():
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)
        print(table.to_string(index=True))


def write_audit_workbook(data_path: str | Path, out_path: str | Path) -> None:
    tables = dataset_validation_tables(data_path)
    ensure_parent(out_path)
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
            for name, table in tables.items():
                table.to_excel(writer, sheet_name=name[:31], index=True)
    except ImportError as exc:
        raise ImportError("Install xlsxwriter first: python -m pip install xlsxwriter") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thesis preprocessing pipeline.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("news-combine", help="Combine raw news CSV files into one master file.")
    p.add_argument("--folder", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--recursive", action="store_true")

    p = sub.add_parser("news-clean", help="Clean and trading-align combined news headlines.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)

    p = sub.add_parser("news-sentiment", help="Build daily VADER sentiment factors.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)

    p = sub.add_parser("stock-clean", help="Clean/process one EOD stock CSV.")
    p.add_argument("--symbol", choices=["NVDA", "SPY", "SOXX", "IEF"], default="NVDA")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)

    sub.add_parser("stock-clean-all", help="Clean/process NVDA, SPY, SOXX and IEF using default paths.")

    p = sub.add_parser("trends-reconstruct", help="Reconstruct daily Google Trends from monthly/daily exports.")
    p.add_argument("--monthly", default=None)
    p.add_argument("--daily-glob", default=None)
    p.add_argument("--output", default=None)

    p = sub.add_parser("trends-clean", help="Clean and feature-engineer Google Trends data.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)

    p = sub.add_parser("build-model", help="Merge processed stock, sentiment and trends data.")
    p.add_argument("--output", default=None)

    p = sub.add_parser("write-clean", help="Create compact model_dataset_clean.csv from model_dataset.csv.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)

    p = sub.add_parser("validate", help="Print final dataset validation checks.")
    p.add_argument("--input", default=None)

    p = sub.add_parser("audit", help="Write Excel audit workbook for final dataset.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)

    p = sub.add_parser("all", help="Run the default full preprocessing pipeline.")
    p.add_argument("--recursive-news", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    p = paths(args.base_dir)
    stock_inputs = {"NVDA": p["nvda_raw"], "SPY": p["spy_raw"], "SOXX": p["soxx_raw"], "IEF": p["ief_raw"]}
    stock_outputs = {
        "NVDA": p["nvda_processed"],
        "SPY": p["spy_processed"],
        "SOXX": p["soxx_processed"],
        "IEF": p["ief_processed"],
    }

    if args.command == "news-combine":
        out = combine_all_news_csvs(args.folder or p["raw_news_dir"], args.output or p["news_master"], args.recursive)
        print_done(f"Combined news CSVs. Rows: {len(out):,}", args.output or p["news_master"])

    elif args.command == "news-clean":
        out = clean_news_dataset(args.input or p["news_master"], args.output or p["news_clean"])
        print_done(f"Cleaned and trading-aligned news. Rows: {len(out):,}", args.output or p["news_clean"])

    elif args.command == "news-sentiment":
        out = build_daily_sentiment(args.input or p["news_clean"], args.output or p["news_sentiment"])
        print_done(f"Built daily sentiment. Rows: {len(out):,}", args.output or p["news_sentiment"])

    elif args.command == "stock-clean":
        inp = Path(args.input) if args.input else stock_inputs[args.symbol]
        outp = Path(args.output) if args.output else stock_outputs[args.symbol]
        out = clean_and_process_eod(inp, outp)
        print_done(f"Processed {args.symbol}. Rows: {len(out):,}", outp)

    elif args.command == "stock-clean-all":
        for symbol in ["NVDA", "SPY", "SOXX", "IEF"]:
            out = clean_and_process_eod(stock_inputs[symbol], stock_outputs[symbol])
            print_done(f"Processed {symbol}. Rows: {len(out):,}", stock_outputs[symbol])

    elif args.command == "trends-reconstruct":
        out = reconstruct_google_trends(
            args.monthly or p["trends_monthly"],
            args.daily_glob or str(p["trends_daily_glob"]),
            args.output or p["trends_consistent"],
        )
        print_done(f"Reconstructed daily Trends. Rows: {len(out):,}", args.output or p["trends_consistent"])

    elif args.command == "trends-clean":
        out = clean_and_process_trends(args.input or p["trends_consistent"], args.output or p["trends_processed"])
        print_done(f"Processed Trends. Rows: {len(out):,}", args.output or p["trends_processed"])

    elif args.command == "build-model":
        out = build_model_dataset(
            p["nvda_processed"],
            p["spy_processed"],
            p["soxx_processed"],
            p["ief_processed"],
            p["news_sentiment"],
            p["trends_processed"],
            args.output or p["model_dataset"],
        )
        print_done(f"Built model dataset. Rows: {len(out):,}; Columns: {len(out.columns):,}", args.output or p["model_dataset"])

    elif args.command == "write-clean":
        out = write_clean_model_dataset(args.input or p["model_dataset"], args.output or p["model_dataset_clean"])
        print_done(f"Wrote clean model dataset. Rows: {len(out):,}; Columns: {len(out.columns):,}", args.output or p["model_dataset_clean"])

    elif args.command == "validate":
        validate_dataset(args.input or p["model_dataset_clean"])

    elif args.command == "audit":
        write_audit_workbook(args.input or p["model_dataset_clean"], args.output or p["model_audit"])
        print_done("Wrote audit workbook", args.output or p["model_audit"])

    elif args.command == "all":
        out = combine_all_news_csvs(p["raw_news_dir"], p["news_master"], recursive=args.recursive_news)
        print_done(f"Combined news CSVs. Rows: {len(out):,}", p["news_master"])
        out = clean_news_dataset(p["news_master"], p["news_clean"])
        print_done(f"Cleaned news. Rows: {len(out):,}", p["news_clean"])
        out = build_daily_sentiment(p["news_clean"], p["news_sentiment"])
        print_done(f"Built sentiment. Rows: {len(out):,}", p["news_sentiment"])
        for symbol in ["NVDA", "SPY", "SOXX", "IEF"]:
            out = clean_and_process_eod(stock_inputs[symbol], stock_outputs[symbol])
            print_done(f"Processed {symbol}. Rows: {len(out):,}", stock_outputs[symbol])
        out = reconstruct_google_trends(p["trends_monthly"], str(p["trends_daily_glob"]), p["trends_consistent"])
        print_done(f"Reconstructed Trends. Rows: {len(out):,}", p["trends_consistent"])
        out = clean_and_process_trends(p["trends_consistent"], p["trends_processed"])
        print_done(f"Processed Trends. Rows: {len(out):,}", p["trends_processed"])
        out = build_model_dataset(
            p["nvda_processed"],
            p["spy_processed"],
            p["soxx_processed"],
            p["ief_processed"],
            p["news_sentiment"],
            p["trends_processed"],
            p["model_dataset"],
        )
        print_done(f"Built model dataset. Rows: {len(out):,}", p["model_dataset"])
        out = write_clean_model_dataset(p["model_dataset"], p["model_dataset_clean"])
        print_done(f"Wrote clean model dataset. Rows: {len(out):,}", p["model_dataset_clean"])
        validate_dataset(p["model_dataset_clean"])
        write_audit_workbook(p["model_dataset_clean"], p["model_audit"])
        print_done("Wrote audit workbook", p["model_audit"])


if __name__ == "__main__":
    main()
