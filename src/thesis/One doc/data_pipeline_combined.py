#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path("/home/zammorak/thesis/data")

RAW_NEWS_DIR = BASE_DIR / "raw/news_headlines"
NEWS_MASTER_PATH = BASE_DIR / "interim/news_headlines_master.csv"
NEWS_CLEAN_PATH = BASE_DIR / "processed/news_headlines_clean.csv"
NEWS_SENTIMENT_PATH = BASE_DIR / "processed/news_daily_sentiment.csv"

TRENDS_RAW_PATH = BASE_DIR / "interim/nvidia_trends_daily_consistent.csv"
TRENDS_PROCESSED_PATH = BASE_DIR / "processed/nvidia_trends_processed.csv"

MODEL_DIR = BASE_DIR / "model_feed"
MODEL_DATASET_PATH = MODEL_DIR / "model_dataset.csv"
MODEL_AUDIT_PATH = MODEL_DIR / "model_dataset_audit.xlsx"

DEFAULT_STOCK_INPUTS = {
    "NVDA": BASE_DIR / "raw/stock_data/NVDA/NVDA_eod_chunked_2019-03-01_to_2026-03-01.csv",
    "SPY": BASE_DIR / "raw/macro_stock_data/SPY/SPY_eod_chunked_2019-03-01_to_2026-03-01.csv",
    "SOXX": BASE_DIR / "raw/macro_stock_data/SOXX/SOXX_eod_chunked_2019-03-01_to_2026-03-01.csv",
    "IEF": BASE_DIR / "raw/macro_stock_data/IEF/IEF_eod_chunked_2019-03-01_to_2026-03-01.csv",
}
DEFAULT_STOCK_OUTPUTS = {
    symbol: BASE_DIR / f"processed/{symbol}_eod_processed.csv"
    for symbol in DEFAULT_STOCK_INPUTS
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
    print(f"Done: {message}")
    if path is not None:
        print(f"Saved to: {path}")


def combine_all_news_csvs(
    folder_path: str | Path = RAW_NEWS_DIR,
    output_path: str | Path = NEWS_MASTER_PATH,
    recursive: bool = False,
) -> pd.DataFrame:
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
        "symbol_target", "uuid", "published_at", "date", "title", "description",
        "keywords", "snippet", "url", "source", "image_url", "entities_json", "raw_json",
    ]
    preferred_present = [c for c in preferred_order if c in union_cols]
    remaining = sorted([c for c in union_cols if c not in preferred_present])
    canonical_cols = preferred_present + remaining

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        df.columns = standardize_columns(df.columns)
        if "source_file" not in canonical_cols:
            canonical_cols.append("source_file")
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


def clean_news_dataset(
    input_path: str | Path = NEWS_MASTER_PATH,
    output_path: str | Path = NEWS_CLEAN_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = standardize_columns(df.columns)

    required_cols = ["symbol_target", "published_at", "title", "description", "keywords", "source"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy()
    text_cols = ["title", "description", "keywords", "source"]
    for col in text_cols:
        df[col] = df[col].where(df[col].notna(), None)
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": None, "nan": None, "None": None})

    df = df[df["title"].notna()]
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df[df["published_at"].notna()]

    df["published_et"] = df["published_at"].dt.tz_convert("America/New_York")
    df["weekday_et"] = df["published_et"].dt.weekday

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
    df = df.drop(columns=["weekday_et"], errors="ignore")

    ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df


def build_daily_sentiment(
    input_path: str | Path = NEWS_CLEAN_PATH,
    output_path: str | Path = NEWS_SENTIMENT_PATH,
) -> pd.DataFrame:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    df = pd.read_csv(input_path)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    if "trading_date" not in df.columns:
        raise ValueError("trading_date column missing. Run news-clean first.")

    sid = SentimentIntensityAnalyzer()
    df["text_for_sentiment"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    df["sentiment"] = df["text_for_sentiment"].apply(lambda x: sid.polarity_scores(x)["compound"])

    daily = df.groupby("trading_date").agg(
        avg_sentiment=("sentiment", "mean"),
        median_sentiment=("sentiment", "median"),
        sentiment_std=("sentiment", "std"),
        news_count=("sentiment", "count"),
        positive_ratio=("sentiment", lambda x: np.mean(x > 0)),
        negative_ratio=("sentiment", lambda x: np.mean(x < 0)),
    ).reset_index()
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    daily = daily.sort_values("trading_date").reset_index(drop=True)

    ensure_parent(output_path)
    daily.to_csv(output_path, index=False)
    return daily


def clean_and_process_eod(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = standardize_columns(df.columns)

    required_price_cols = ["date", "open", "high", "low", "close"]
    missing = [c for c in required_price_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df.drop_duplicates(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

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


def clean_and_process_trends(
    input_path: str | Path = TRENDS_RAW_PATH,
    output_path: str | Path = TRENDS_PROCESSED_PATH,
) -> pd.DataFrame:
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


def build_model_dataset(
    nvda_path: str | Path = DEFAULT_STOCK_OUTPUTS["NVDA"],
    spy_path: str | Path = DEFAULT_STOCK_OUTPUTS["SPY"],
    soxx_path: str | Path = DEFAULT_STOCK_OUTPUTS["SOXX"],
    ief_path: str | Path = DEFAULT_STOCK_OUTPUTS["IEF"],
    sent_path: str | Path = NEWS_SENTIMENT_PATH,
    trends_path: str | Path = TRENDS_PROCESSED_PATH,
    output_path: str | Path = MODEL_DATASET_PATH,
) -> pd.DataFrame:
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
        "avg_sentiment", "median_sentiment", "sentiment_std", "news_count",
        "positive_ratio", "negative_ratio", "trends_raw", "trends_7d_ma",
        "trends_momentum_7d", "trends_zscore_30d", "trends_spike",
    ]
    existing_fill_cols = [c for c in fill_cols if c in df.columns]
    df[existing_fill_cols] = df[existing_fill_cols].ffill()

    df = df.dropna().reset_index(drop=True)
    ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df


def dataset_validation_tables(data_path: str | Path = MODEL_DATASET_PATH) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    basic_info = pd.DataFrame({
        "metric": ["rows", "columns", "date_min", "date_max"],
        "value": [len(df), len(df.columns), df["date"].min(), df["date"].max()],
    })

    dup_mask = df["date"].duplicated(keep=False)
    duplicate_dates = df.loc[dup_mask, ["date"]].copy()
    if not duplicate_dates.empty:
        duplicate_dates["count_for_date"] = duplicate_dates.groupby("date")["date"].transform("count")
        duplicate_dates = duplicate_dates.sort_values(["date"]).drop_duplicates()
    dup_summary = pd.DataFrame({
        "duplicate_dates_total": [int(df["date"].duplicated().sum())],
        "unique_dates_duplicated": [int(duplicate_dates["date"].nunique()) if not duplicate_dates.empty else 0],
    })

    date_diffs = df["date"].diff().dt.days
    gaps = pd.DataFrame({"date": df["date"], "gap_days_since_prev": date_diffs})
    large_gaps = gaps[gaps["gap_days_since_prev"] > 3].copy()
    max_gap = pd.DataFrame({"max_gap_days": [float(date_diffs.max())]})

    missing_table = pd.DataFrame({
        "missing_count": df.isna().sum().sort_values(ascending=False),
        "missing_pct": (df.isna().mean() * 100).round(2).sort_values(ascending=False),
    })
    missing_table = missing_table[missing_table["missing_count"] > 0]
    if missing_table.empty:
        missing_table = pd.DataFrame({"note": ["No missing values detected"]})

    zero_var_table = df.nunique(dropna=False)
    zero_var_table = zero_var_table[zero_var_table <= 1].sort_values().to_frame("n_unique_values")
    if zero_var_table.empty:
        zero_var_table = pd.DataFrame({"note": ["No zero-variance columns"]})

    sanity_rows = []
    for col, label in [
        ("log_return", "Return"),
        ("avg_sentiment", "Sentiment"),
        ("trends_zscore_30d", "Trends z-score (30d)"),
    ]:
        if col in df.columns:
            s = df[col].dropna()
            sanity_rows += [
                {"metric": f"{label} mean", "value": np.mean(s)},
                {"metric": f"{label} std", "value": np.std(s)},
                {"metric": f"{label} min", "value": np.min(s)},
                {"metric": f"{label} max", "value": np.max(s)},
            ]
        else:
            sanity_rows.append({"metric": f"{label} (missing column)", "value": "N/A"})
    sanity_checks = pd.DataFrame(sanity_rows)

    if "target_next_return" in df.columns:
        corr = df.corr(numeric_only=True)
        corr_target = corr["target_next_return"].sort_values(ascending=False).to_frame("corr_with_target_next_return")
    else:
        corr_target = pd.DataFrame({"note": ["Column target_next_return not found"]})

    if "trends_zscore_30d" in df.columns and "target_next_return" in df.columns:
        df_tmp = df.copy()
        df_tmp["trends_z_lag1"] = df_tmp["trends_zscore_30d"].shift(1)
        df_tmp["trends_z_lag2"] = df_tmp["trends_zscore_30d"].shift(2)
        lag_corrs = pd.DataFrame({
            "feature": ["trends_z_lag1", "trends_z_lag2"],
            "corr_with_target_next_return": [
                df_tmp["trends_z_lag1"].corr(df_tmp["target_next_return"]),
                df_tmp["trends_z_lag2"].corr(df_tmp["target_next_return"]),
            ],
        })
    else:
        lag_corrs = pd.DataFrame({"note": ["Required columns missing: trends_zscore_30d and/or target_next_return"]})

    if "trends_spike" in df.columns and "target_next_return" in df.columns:
        group_trends = df.groupby("trends_spike")["target_next_return"].mean().to_frame("mean_target_next_return")
    else:
        group_trends = pd.DataFrame({"note": ["Required columns missing: trends_spike and/or target_next_return"]})

    return {
        "basic_info": basic_info,
        "dup_summary": dup_summary,
        "duplicate_dates": duplicate_dates,
        "max_gap": max_gap,
        "large_gaps": large_gaps,
        "missing_table": missing_table,
        "zero_var_table": zero_var_table,
        "sanity_checks": sanity_checks,
        "corr_target": corr_target,
        "lag_corrs": lag_corrs,
        "group_trends": group_trends,
    }


def validate_dataset(data_path: str | Path = MODEL_DATASET_PATH) -> None:
    tables = dataset_validation_tables(data_path)
    print("===================================")
    print("BASIC INFO")
    print("===================================")
    print(tables["basic_info"].to_string(index=False))
    print("\nDuplicate summary:")
    print(tables["dup_summary"].to_string(index=False))
    print("\nMax gap:")
    print(tables["max_gap"].to_string(index=False))
    print("\nPotential abnormal gaps (>3 days):")
    print(tables["large_gaps"].to_string(index=False))
    print("\nMissing values:")
    print(tables["missing_table"].to_string())
    print("\nZero variance columns:")
    print(tables["zero_var_table"].to_string())
    print("\nSanity checks:")
    print(tables["sanity_checks"].to_string(index=False))
    print("\nCorrelation with target_next_return:")
    print(tables["corr_target"].to_string())
    print("\nLag correlations:")
    print(tables["lag_corrs"].to_string(index=False))
    print("\nGroup by trends_spike:")
    print(tables["group_trends"].to_string())


def write_audit_workbook(
    data_path: str | Path = MODEL_DATASET_PATH,
    out_path: str | Path = MODEL_AUDIT_PATH,
) -> None:
    tables = dataset_validation_tables(data_path)
    out_path = Path(out_path)
    ensure_parent(out_path)

    with pd.ExcelWriter(out_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
        cell_fmt = workbook.add_format({"border": 1})

        def write_sheet(df_sheet: pd.DataFrame, name: str, index: bool = True, freeze=(1, 0), autofilter=True):
            df_sheet.to_excel(writer, sheet_name=name, index=index)
            ws = writer.sheets[name]
            ws.freeze_panes(*freeze)
            ws.set_row(0, None, header_fmt)
            width_df = df_sheet.reset_index() if index else df_sheet
            for col_idx, col_name in enumerate(width_df.columns):
                series_as_str = width_df.iloc[:, col_idx].astype(str) if len(width_df) else pd.Series(dtype=str)
                max_len = max([len(str(col_name))] + series_as_str.map(len).tolist())
                ws.set_column(col_idx, col_idx, min(max_len + 2, 50), cell_fmt)
            if autofilter:
                rows, cols = width_df.shape
                ws.autofilter(0, 0, rows, max(cols - 1, 0))

        write_sheet(tables["basic_info"], "Basic_Info", index=False)
        write_sheet(tables["dup_summary"], "Duplicate_Dates", index=False)
        write_sheet(tables["duplicate_dates"].set_index("date") if not tables["duplicate_dates"].empty else pd.DataFrame({"note": ["No duplicate dates"]}), "Duplicate_Dates_List")
        write_sheet(tables["max_gap"], "Date_Gaps", index=False)
        write_sheet(tables["large_gaps"].set_index("date") if not tables["large_gaps"].empty else pd.DataFrame({"note": ["No large gaps >3 days"]}), "Large_Gaps_List")
        write_sheet(tables["missing_table"], "Missing_Values")
        write_sheet(tables["zero_var_table"], "Zero_Variance")
        write_sheet(tables["sanity_checks"], "Sanity_Checks", index=False)
        write_sheet(tables["corr_target"], "Corr_TargetNextReturn")
        write_sheet(tables["lag_corrs"].set_index("feature") if "feature" in tables["lag_corrs"].columns else tables["lag_corrs"], "Lag_Corrs")
        write_sheet(tables["group_trends"], "Group_TrendsSpike")

    print_done("Wrote audit workbook", out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combined thesis data pipeline utility.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("news-combine", help="Combine raw news CSV files into one master file.")
    p.add_argument("--folder", default=str(RAW_NEWS_DIR))
    p.add_argument("--output", default=str(NEWS_MASTER_PATH))
    p.add_argument("--recursive", action="store_true")

    p = sub.add_parser("news-clean", help="Clean and trading-align combined news headlines.")
    p.add_argument("--input", default=str(NEWS_MASTER_PATH))
    p.add_argument("--output", default=str(NEWS_CLEAN_PATH))

    p = sub.add_parser("news-sentiment", help="Build daily VADER sentiment factors.")
    p.add_argument("--input", default=str(NEWS_CLEAN_PATH))
    p.add_argument("--output", default=str(NEWS_SENTIMENT_PATH))

    p = sub.add_parser("stock-clean", help="Clean/process one EOD stock CSV.")
    p.add_argument("--symbol", choices=sorted(DEFAULT_STOCK_INPUTS), default="NVDA")
    p.add_argument("--input", default=None, help="Defaults to the configured path for --symbol.")
    p.add_argument("--output", default=None, help="Defaults to the configured processed path for --symbol.")

    sub.add_parser("stock-clean-all", help="Clean/process NVDA, SPY, SOXX, and IEF using default paths.")

    p = sub.add_parser("trends-clean", help="Clean and feature-engineer Google Trends data.")
    p.add_argument("--input", default=str(TRENDS_RAW_PATH))
    p.add_argument("--output", default=str(TRENDS_PROCESSED_PATH))

    p = sub.add_parser("build-model", help="Merge processed stock, sentiment, and trends data into model_dataset.csv.")
    p.add_argument("--output", default=str(MODEL_DATASET_PATH))
    p.add_argument("--nvda", default=str(DEFAULT_STOCK_OUTPUTS["NVDA"]))
    p.add_argument("--spy", default=str(DEFAULT_STOCK_OUTPUTS["SPY"]))
    p.add_argument("--soxx", default=str(DEFAULT_STOCK_OUTPUTS["SOXX"]))
    p.add_argument("--ief", default=str(DEFAULT_STOCK_OUTPUTS["IEF"]))
    p.add_argument("--sentiment", default=str(NEWS_SENTIMENT_PATH))
    p.add_argument("--trends", default=str(TRENDS_PROCESSED_PATH))

    p = sub.add_parser("validate", help="Print final dataset validation checks.")
    p.add_argument("--input", default=str(MODEL_DATASET_PATH))

    p = sub.add_parser("audit", help="Write Excel audit workbook for final dataset.")
    p.add_argument("--input", default=str(MODEL_DATASET_PATH))
    p.add_argument("--output", default=str(MODEL_AUDIT_PATH))

    p = sub.add_parser("all", help="Run the full default pipeline and write validation/audit outputs.")
    p.add_argument("--recursive-news", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "news-combine":
        df = combine_all_news_csvs(args.folder, args.output, args.recursive)
        print_done(f"Combined news CSVs. Rows: {len(df):,}; Columns: {len(df.columns):,}", args.output)
    elif args.command == "news-clean":
        df = clean_news_dataset(args.input, args.output)
        print_done(f"Cleaned and trading-aligned news. Rows: {len(df):,}; Columns: {len(df.columns):,}", args.output)
    elif args.command == "news-sentiment":
        df = build_daily_sentiment(args.input, args.output)
        print_done(f"Built daily sentiment. Rows: {len(df):,}; Columns: {len(df.columns):,}", args.output)
    elif args.command == "stock-clean":
        input_path = Path(args.input) if args.input else DEFAULT_STOCK_INPUTS[args.symbol]
        output_path = Path(args.output) if args.output else DEFAULT_STOCK_OUTPUTS[args.symbol]
        df = clean_and_process_eod(input_path, output_path)
        print_done(f"Processed {args.symbol} EOD. Rows: {len(df):,}; Columns: {len(df.columns):,}", output_path)
    elif args.command == "stock-clean-all":
        for symbol in ["NVDA", "SPY", "SOXX", "IEF"]:
            df = clean_and_process_eod(DEFAULT_STOCK_INPUTS[symbol], DEFAULT_STOCK_OUTPUTS[symbol])
            print_done(f"Processed {symbol} EOD. Rows: {len(df):,}; Columns: {len(df.columns):,}", DEFAULT_STOCK_OUTPUTS[symbol])
    elif args.command == "trends-clean":
        df = clean_and_process_trends(args.input, args.output)
        print_done(f"Processed trends. Rows: {len(df):,}; Columns: {len(df.columns):,}", args.output)
    elif args.command == "build-model":
        df = build_model_dataset(args.nvda, args.spy, args.soxx, args.ief, args.sentiment, args.trends, args.output)
        print_done(f"Built model dataset. Rows: {len(df):,}; Columns: {len(df.columns):,}", args.output)
    elif args.command == "validate":
        validate_dataset(args.input)
    elif args.command == "audit":
        write_audit_workbook(args.input, args.output)
    elif args.command == "all":
        df = combine_all_news_csvs(RAW_NEWS_DIR, NEWS_MASTER_PATH, recursive=args.recursive_news)
        print_done(f"Combined news CSVs. Rows: {len(df):,}", NEWS_MASTER_PATH)
        df = clean_news_dataset(NEWS_MASTER_PATH, NEWS_CLEAN_PATH)
        print_done(f"Cleaned news. Rows: {len(df):,}", NEWS_CLEAN_PATH)
        df = build_daily_sentiment(NEWS_CLEAN_PATH, NEWS_SENTIMENT_PATH)
        print_done(f"Built daily sentiment. Rows: {len(df):,}", NEWS_SENTIMENT_PATH)
        for symbol in ["NVDA", "SPY", "SOXX", "IEF"]:
            df = clean_and_process_eod(DEFAULT_STOCK_INPUTS[symbol], DEFAULT_STOCK_OUTPUTS[symbol])
            print_done(f"Processed {symbol} EOD. Rows: {len(df):,}", DEFAULT_STOCK_OUTPUTS[symbol])
        df = clean_and_process_trends(TRENDS_RAW_PATH, TRENDS_PROCESSED_PATH)
        print_done(f"Processed trends. Rows: {len(df):,}", TRENDS_PROCESSED_PATH)
        df = build_model_dataset(output_path=MODEL_DATASET_PATH)
        print_done(f"Built model dataset. Rows: {len(df):,}; Columns: {len(df.columns):,}", MODEL_DATASET_PATH)
        validate_dataset(MODEL_DATASET_PATH)
        write_audit_workbook(MODEL_DATASET_PATH, MODEL_AUDIT_PATH)


if __name__ == "__main__":
    main()
