#!/usr/bin/env python3
"""
Download Google-Trends search-interest for NASDAQ:NVDA at 3 resolutions:

1. MONTHLY  2004-01-01 → today   → nvda_monthly_2004-YYYY.csv
2. DAILY    full history (≤270-day windows)  → one CSV per calendar month
3. HOURLY   last 7 days                      → nvda_last7d_hourly.csv

All requests:
  • random User-Agent
  • random 60-180 s pause between calls
  • each file saved immediately
"""

import time, random, datetime as dt, calendar, pathlib
import pandas as pd
from pytrends.request import TrendReq

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
KW            = ["NASDAQ:NVDA"]
DATA_DIR      = pathlib.Path("trends_data")
DATA_DIR.mkdir(exist_ok=True)
SLEEP_RANGE   = (60, 180)      # seconds

USER_AGENTS   = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

def new_session():
    ua = random.choice(USER_AGENTS)
    return TrendReq(
        hl="en-US", tz=0,
        retries=2, backoff_factor=0.2,
        requests_args={"headers": {"User-Agent": ua}},
    )

def fetch(kw, timeframe):
    session = new_session()
    session.build_payload(kw, timeframe=timeframe)
    df = session.interest_over_time().drop(columns="isPartial")
    sleep_for = random.uniform(*SLEEP_RANGE)
    print(f"Fetched {timeframe:<24} – sleeping {sleep_for:0.1f}s")
    time.sleep(sleep_for)
    return df

# ──────────────────────────────────────────────────────────────
# 1) MONTHLY (entire history)
# ──────────────────────────────────────────────────────────────
print("\n▣ MONTHLY SERIES")
start_all = "2004-01-01"
today     = dt.date.today().isoformat()
monthly_raw = fetch(KW, f"{start_all} {today}")

monthly = (
    monthly_raw
    .resample("M")
    .mean()
    .rename(columns={KW[0]: "nvda_trend"})
)

monthly_out = DATA_DIR / f"nvda_monthly_2004-{today[:4]}.csv"
monthly.to_csv(monthly_out)
print(f"✔ Saved monthly CSV → {monthly_out.name}")

# ──────────────────────────────────────────────────────────────
# 2) DAILY series – one CSV per calendar month
# ──────────────────────────────────────────────────────────────
print("\n▣ DAILY SERIES (last 6 years only)")
train_start = dt.date.today() - dt.timedelta(days=6 * 365)
last_day = dt.date.today()

for yr in range(train_start.year, last_day.year + 1):
    for mo in range(1, 13):
        win_start = dt.date(yr, mo, 1)
        if win_start < train_start or win_start > last_day:
            continue

        win_end = min(
            dt.date(yr, mo, calendar.monthrange(yr, mo)[1]),
            last_day
        )

        timeframe = f"{win_start} {win_end}"
        outname = DATA_DIR / f"nvda_daily_{win_start}.csv"

        if outname.exists():
            print(f"  ↳ Skipping {outname.name} (already exists)")
            continue

        try:
            daily_df = fetch(KW, timeframe)
            if daily_df.empty:
                print(f"  ↳ Skipped (no data for {win_start})")
                continue
            daily_df.to_csv(outname)
            print(f"  ↳ Saved {outname.name}")
        except Exception as e:
            print(f"  ⚠️ Failed {timeframe}: {e}")


# ──────────────────────────────────────────────────────────────
# 3) HOURLY – last 7 days
# ──────────────────────────────────────────────────────────────
print("\n▣ HOURLY SERIES (now 7-d)")
try:
    hourly = fetch(KW, "now 7-d")
    hourly_out = DATA_DIR / "nvda_last7d_hourly.csv"
    hourly.to_csv(hourly_out)
    print(f"✔ Saved hourly CSV → {hourly_out.name}")
except Exception as e:
    print(f"⚠️ Hourly fetch failed: {e}")

# ──────────────────────────────────────────────────────────────
print("\n✅ All available files saved to:", DATA_DIR.resolve())
