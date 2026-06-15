import glob
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
BASE = "/home/zammorak/Downloads/OLD_TREND"   # <-- change if needed
MONTHLY_FILE = f"{BASE}/multiTimeline.csv"
DAILY_GLOB = f"{BASE}/multiTimeline(??).csv"

# -----------------------------
# 1) Read Google Trends CSV (daily or monthly)
# -----------------------------
def read_trends_csv(path: str) -> pd.DataFrame:
    # Google Trends exports include a few header lines before the real CSV header.
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Dag,") or line.startswith("Maand,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find header row in {path}")

    df = pd.read_csv(path, skiprows=header_idx)

    # Expect first col = date, second col = series value
    date_col, value_col = df.columns[:2]
    df = df.rename(columns={date_col: "date", value_col: "value"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = df["value"].replace({"<1": 0}).astype(float)

    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df[["date", "value"]]

# -----------------------------
# 2) Monthly reference (global scale anchor)
# -----------------------------
monthly = read_trends_csv(MONTHLY_FILE)

# month key as month-start; works on older pandas too
monthly["month"] = monthly["date"].values.astype("datetime64[M]")

monthly_ref = (
    monthly.groupby("month", as_index=False)["value"]
    .mean()
    .rename(columns={"value": "ref_value"})
)

# -----------------------------
# 3) Read daily chunks
# -----------------------------
daily_files = sorted(glob.glob(DAILY_GLOB))
if not daily_files:
    raise FileNotFoundError(f"No daily chunk files found: {DAILY_GLOB}")

daily_chunks = [read_trends_csv(p) for p in daily_files]

# -----------------------------
# 4) Scale each daily chunk to monthly reference
#    (coarse scaling)
# -----------------------------
def compute_monthly_scale(daily_df: pd.DataFrame, monthly_ref: pd.DataFrame) -> float:
    tmp = daily_df.copy()
    tmp["month"] = tmp["date"].values.astype("datetime64[M]")

    chunk_monthly = tmp.groupby("month", as_index=False)["value"].mean()
    merged = chunk_monthly.merge(monthly_ref, on="month", how="inner")

    if merged.empty:
        return 1.0

    ratios = merged["ref_value"] / merged["value"].replace(0, np.nan)
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

    return float(ratios.median()) if len(ratios) else 1.0

def scale_daily_chunk_to_monthly(daily_df: pd.DataFrame, monthly_ref: pd.DataFrame) -> pd.DataFrame:
    scale = compute_monthly_scale(daily_df, monthly_ref)
    out = daily_df.copy()
    out["value"] = out["value"] * scale
    return out

scaled_chunks = [scale_daily_chunk_to_monthly(ch, monthly_ref) for ch in daily_chunks]

# -----------------------------
# 4b) Boundary chain-linking (fine scaling)
#     Force continuity at each chunk boundary:
#     multiply chunk i by (last value of chunk i-1) / (first value of chunk i)
# -----------------------------
for i in range(1, len(scaled_chunks)):
    prev = scaled_chunks[i - 1]
    curr = scaled_chunks[i]

    prev_last_date = prev["date"].max()
    curr_first_date = curr["date"].min()

    v_prev = float(prev.loc[prev["date"] == prev_last_date, "value"].iloc[0])
    v_curr = float(curr.loc[curr["date"] == curr_first_date, "value"].iloc[0])

    if abs(v_curr) > 1e-9:
        factor = v_prev / v_curr
        scaled_chunks[i]["value"] = scaled_chunks[i]["value"] * factor

# -----------------------------
# 5) Concatenate, resolve overlaps, make continuous daily index
# -----------------------------
daily_scaled = pd.concat(scaled_chunks, ignore_index=True).sort_values("date")

# If there are overlaps (you had one around 2024-06), pick ONE strategy:
# Strategy A (recommended): keep the earlier chunk's values for overlaps
daily_scaled = daily_scaled.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

# Strategy B (alternative): blend overlaps by averaging
# daily_scaled = (daily_scaled.groupby("date", as_index=False)["value"].mean()
#                .sort_values("date").reset_index(drop=True))

# Make a continuous daily series and fill any gaps
daily_series = daily_scaled.set_index("date").asfreq("D")

# Fill missing dates (if any). Interpolation is reasonable for Trends.
daily_series["value"] = daily_series["value"].interpolate(limit_direction="both")

daily_series = daily_series.reset_index()

# Optional: clip extreme outliers (usually not needed after chain-linking)
# lo, hi = daily_series["value"].quantile([0.001, 0.999])
# daily_series["value"] = daily_series["value"].clip(lo, hi)

# -----------------------------
# Diagnostics (prints)
# -----------------------------
print("Daily series:", daily_series["date"].min().date(), "→", daily_series["date"].max().date())
print(daily_series.head())
print(daily_series.tail())

# Confirm no missing days
print("Missing days:", daily_series["date"].diff().dt.days.value_counts().sort_index().head(10))

# Print per-file monthly scale (before chain-link) + date ranges
for p, ch in zip(daily_files, daily_chunks):
    s = compute_monthly_scale(ch, monthly_ref)
    print(f"{p.split('/')[-1]:>20}  monthly_scale={s:.4f}  range={ch['date'].min().date()}→{ch['date'].max().date()}")

# Boundary step check after ALL scaling (should be ~0% now)
# Build ranges from ORIGINAL files
ranges = []
for p in daily_files:
    df = read_trends_csv(p)
    ranges.append((p.split("/")[-1], df["date"].min(), df["date"].max()))

daily_scaled_idx = daily_scaled.set_index("date")

for (name_a, start_a, end_a), (name_b, start_b, end_b) in zip(ranges[:-1], ranges[1:]):
    prev_day = end_a
    next_day = start_b
    if prev_day in daily_scaled_idx.index and next_day in daily_scaled_idx.index:
        v_prev = float(daily_scaled_idx.loc[prev_day, "value"])
        v_next = float(daily_scaled_idx.loc[next_day, "value"])
        pct = (v_next - v_prev) / max(abs(v_prev), 1e-9) * 100
        print(f"{name_a}→{name_b}: {prev_day.date()}={v_prev:.2f}  {next_day.date()}={v_next:.2f}  step={pct:+.4f}%")


out_path = f"{BASE}/nvidia_trends_daily_consistent.csv"
daily_series.to_csv(out_path, index=False)
print("Saved:", out_path)


import matplotlib.pyplot as plt

# Filter only 2024
daily_2024 = daily_series[
    (daily_series["date"] >= "2024-01-01") &
    (daily_series["date"] <= "2024-12-31")
]

plt.figure(figsize=(14, 6))
plt.plot(daily_2024["date"], daily_2024["value"])
plt.title("Daily Google Trends – Nvidia (Worldwide) — Year 2024")
plt.xlabel("Date")
plt.ylabel("Search Interest (Scaled)")
plt.tight_layout()

plt.savefig("/home/zammorak/Downloads/OLD_TREND/nvidia_daily_2024.png", dpi=150)
plt.close()

print("Graph saved: /home/zammorak/Downloads/OLD_TREND/nvidia_daily_2024.png")

daily_2024 = daily_2024.copy()
daily_2024["ma7"] = daily_2024["value"].rolling(7).mean()

plt.figure(figsize=(14, 6))
plt.plot(daily_2024["date"], daily_2024["value"], alpha=0.4, label="Daily")
plt.plot(daily_2024["date"], daily_2024["ma7"], label="7-day average")
plt.title("Daily Google Trends – Nvidia (Worldwide) — Year 2024")
plt.xlabel("Date")
plt.ylabel("Search Interest (Scaled)")
plt.legend()
plt.tight_layout()

plt.savefig("/home/zammorak/Downloads/OLD_TREND/nvidia_daily_2024_smooth.png", dpi=150)
plt.close()

print("Graph saved: /home/zammorak/Downloads/OLD_TREND/nvidia_daily_2024_smooth.png")




