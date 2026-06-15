import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("/home/zammorak/thesis/data/model_feed/model_dataset.csv")
OUT_PATH  = DATA_PATH.with_name("model_dataset_audit.xlsx")

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# -----------------------------
# 1) Basic info
# -----------------------------
basic_info = pd.DataFrame({
    "metric": ["rows", "columns", "date_min", "date_max"],
    "value": [len(df), len(df.columns), df["date"].min(), df["date"].max()]
})

# -----------------------------
# 2) Duplicate dates
# -----------------------------
dup_mask = df["date"].duplicated(keep=False)
duplicate_dates = df.loc[dup_mask, ["date"]].copy()
duplicate_dates["count_for_date"] = duplicate_dates.groupby("date")["date"].transform("count")
duplicate_dates = duplicate_dates.sort_values(["date"]).drop_duplicates()

dup_summary = pd.DataFrame({
    "duplicate_dates_total": [int(df["date"].duplicated().sum())],
    "unique_dates_duplicated": [int(duplicate_dates["date"].nunique())]
})

# -----------------------------
# 3) Missing dates (gaps)
# -----------------------------
date_diffs = df["date"].diff().dt.days
gaps = pd.DataFrame({
    "date": df["date"],
    "gap_days_since_prev": date_diffs
})
large_gaps = gaps[gaps["gap_days_since_prev"] > 3].copy()
max_gap = pd.DataFrame({"max_gap_days": [float(date_diffs.max())]})

# -----------------------------
# 4) Missing values
# -----------------------------
missing_counts = df.isna().sum().sort_values(ascending=False)
missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)

missing_table = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_pct": missing_pct
})
missing_table = missing_table[missing_table["missing_count"] > 0]

if missing_table.empty:
    missing_table = pd.DataFrame({"note": ["No missing values detected ✅"]})

# -----------------------------
# 5) Zero variance columns
# -----------------------------
nunique = df.nunique(dropna=False)
zero_var = nunique[nunique <= 1].sort_values()
zero_var_table = zero_var.to_frame(name="n_unique_values")

if zero_var_table.empty:
    zero_var_table = pd.DataFrame({"note": ["No zero-variance columns ✅"]})

# -----------------------------
# 6) Sanity checks
# -----------------------------
def safe_stat(series, fn):
    if series is None or series.empty:
        return np.nan
    return fn(series)

sanity_rows = []
for col, label in [
    ("log_return", "Return"),
    ("avg_sentiment", "Sentiment"),
    ("trends_zscore_30d", "Trends z-score (30d)")
]:
    if col in df.columns:
        s = df[col].dropna()
        sanity_rows += [
            {"metric": f"{label} mean", "value": safe_stat(s, np.mean)},
            {"metric": f"{label} std", "value": safe_stat(s, np.std)},
            {"metric": f"{label} min", "value": safe_stat(s, np.min)},
            {"metric": f"{label} max", "value": safe_stat(s, np.max)},
        ]
    else:
        sanity_rows.append({"metric": f"{label} (missing column)", "value": "N/A"})

sanity_checks = pd.DataFrame(sanity_rows)

# -----------------------------
# Correlation with target
# -----------------------------
corr_target = pd.DataFrame()
if "target_next_return" in df.columns:
    corr = df.corr(numeric_only=True)
    if "target_next_return" in corr.columns:
        corr_target = corr["target_next_return"].sort_values(ascending=False).to_frame("corr_with_target_next_return")
else:
    corr_target = pd.DataFrame({"note": ["Column target_next_return not found"]})

# -----------------------------
# Lag correlations (trend zscore lags vs target)
# -----------------------------
lag_corrs = pd.DataFrame()
if "trends_zscore_30d" in df.columns and "target_next_return" in df.columns:
    df_tmp = df.copy()
    df_tmp["trends_z_lag1"] = df_tmp["trends_zscore_30d"].shift(1)
    df_tmp["trends_z_lag2"] = df_tmp["trends_zscore_30d"].shift(2)

    lag_corrs = pd.DataFrame({
        "feature": ["trends_z_lag1", "trends_z_lag2"],
        "corr_with_target_next_return": [
            df_tmp["trends_z_lag1"].corr(df_tmp["target_next_return"]),
            df_tmp["trends_z_lag2"].corr(df_tmp["target_next_return"]),
        ]
    })
else:
    lag_corrs = pd.DataFrame({"note": ["Required columns missing: trends_zscore_30d and/or target_next_return"]})

# -----------------------------
# Groupby trends_spike mean target
# -----------------------------
group_trends = pd.DataFrame()
if "trends_spike" in df.columns and "target_next_return" in df.columns:
    group_trends = df.groupby("trends_spike")["target_next_return"].mean().to_frame("mean_target_next_return")
else:
    group_trends = pd.DataFrame({"note": ["Required columns missing: trends_spike and/or target_next_return"]})

# -----------------------------
# Write Excel with formatting
# -----------------------------
with pd.ExcelWriter(OUT_PATH, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
    workbook  = writer.book

    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#F2F2F2",
        "border": 1
    })
    cell_fmt = workbook.add_format({"border": 1})
    pct_fmt = workbook.add_format({"num_format": "0.00", "border": 1})
    date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd", "border": 1})

    def write_sheet(df_sheet, name, freeze=(1, 0), autofilter=True):
        df_sheet.to_excel(writer, sheet_name=name, index=True)
        ws = writer.sheets[name]
        ws.freeze_panes(*freeze)

        # Apply header format
        ws.set_row(0, None, header_fmt)

        # Autofit-ish: set column widths based on content length (reasonable cap)
        for col_idx, col_name in enumerate(df_sheet.reset_index().columns):
            # get max width from header and data
            series_as_str = df_sheet.reset_index().iloc[:, col_idx].astype(str)
            max_len = max([len(str(col_name))] + series_as_str.map(len).tolist())
            ws.set_column(col_idx, col_idx, min(max_len + 2, 50), cell_fmt)

        if autofilter:
            rows, cols = df_sheet.reset_index().shape
            ws.autofilter(0, 0, rows, cols - 1)

        return ws

    # Write all sheets
    basic_info.to_excel(writer, sheet_name="Basic_Info", index=False)
    writer.sheets["Basic_Info"].set_row(0, None, header_fmt)
    writer.sheets["Basic_Info"].set_column(0, 0, 22, cell_fmt)
    writer.sheets["Basic_Info"].set_column(1, 1, 30, cell_fmt)

    dup_summary.to_excel(writer, sheet_name="Duplicate_Dates", index=False)
    ws = writer.sheets["Duplicate_Dates"]
    ws.set_row(0, None, header_fmt)
    ws.set_column(0, 1, 28, cell_fmt)

    write_sheet(duplicate_dates.set_index("date"), "Duplicate_Dates_List")

    max_gap.to_excel(writer, sheet_name="Date_Gaps", index=False)
    ws = writer.sheets["Date_Gaps"]
    ws.set_row(0, None, header_fmt)
    ws.set_column(0, 0, 18, cell_fmt)

    write_sheet(large_gaps.set_index("date"), "Large_Gaps_List")

    write_sheet(missing_table, "Missing_Values")
    write_sheet(zero_var_table, "Zero_Variance")

    sanity_checks.to_excel(writer, sheet_name="Sanity_Checks", index=False)
    ws = writer.sheets["Sanity_Checks"]
    ws.set_row(0, None, header_fmt)
    ws.set_column(0, 0, 32, cell_fmt)
    ws.set_column(1, 1, 22, cell_fmt)

    write_sheet(corr_target, "Corr_TargetNextReturn")
    write_sheet(lag_corrs.set_index("feature"), "Lag_Corrs")
    write_sheet(group_trends, "Group_TrendsSpike")

print(f"✅ Wrote audit workbook to: {OUT_PATH}")