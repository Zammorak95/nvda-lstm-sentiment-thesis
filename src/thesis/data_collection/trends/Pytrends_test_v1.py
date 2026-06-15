#!/usr/bin/env python3
"""
Fetch NVDA Google Trends data at multiple zoom levels and save to CSV.

Outputs
-------
nvda_weekly_raw.csv   – weekly, 2004-01-01 .. today
nvda_daily_raw.csv    – daily,  2025-06-01 .. 2025-07-07
nvda_hourly_raw.csv   – hourly, 2025-07-01 .. 2025-07-07
nvda_hourly_scaled.csv– hourly, rescaled to the historical weekly baseline
"""

import time
import random
import datetime as dt
import pandas as pd
from pytrends.request import TrendReq

# ──────────────────────────────────────────────────────────────────────────────
# 1. Helpers
# ──────────────────────────────────────────────────────────────────────────────
USER_AGENTS = [
    # A small rotating pool; add more if you like
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

def new_pytrend():
    """Return a TrendReq object with a random User-Agent."""
    ua = random.choice(USER_AGENTS)
    return TrendReq(
        hl="en-US",
        tz=0,                 # UTC; adjust if you want local time
        retries=2,
        backoff_factor=0.2,
        requests_args={"headers": {"User-Agent": ua}},
    )

def fetch(kw_list, timeframe):
    """Fetch a Trends dataframe for a timeframe, handling UA + sleep."""
    pt = new_pytrend()
    pt.build_payload(kw_list, timeframe=timeframe)
    df = pt.interest_over_time().drop(columns="isPartial")
    # Random sleep 60-180 s
    nap = random.uniform(60, 180)
    print(f"Fetched {timeframe}, sleeping {nap:0.1f}s …")
    time.sleep(nap)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 2. Pull the three windows
# ──────────────────────────────────────────────────────────────────────────────
KW = ["NASDAQ:NVDA"]

# (a) Weekly history: 2004-01-01 ➜ today
history_tf = "2004-01-01 {}".format(dt.date.today().isoformat())
weekly = fetch(KW, history_tf)
weekly.to_csv("nvda_weekly_raw.csv")

# (b) Daily bridge: 2025-06-01 ➜ 2025-07-07
daily_tf = "2025-06-01 2025-07-07"
daily = fetch(KW, daily_tf)
daily.to_csv("nvda_daily_raw.csv")

# (c) Hourly window: 2025-07-01 ➜ 2025-07-07  (<= 7 days allowed for hourly)
hourly_tf = "2025-07-01 2025-07-07"
hourly = fetch(KW, hourly_tf)
hourly.to_csv("nvda_hourly_raw.csv")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Rescale hourly ➜ daily ➜ weekly
# ──────────────────────────────────────────────────────────────────────────────
col = KW[0]

# --- 3a. Hourly → Daily scale
hourly_avg_by_day = hourly[col].resample("D").mean()
overlap_days = hourly_avg_by_day.index.intersection(daily.index)
scale_hd = daily.loc[overlap_days, col].mean() / hourly_avg_by_day.loc[overlap_days].mean()

# --- 3b. Daily → Weekly scale
overlap = daily.index.intersection(weekly.index)
# Take means over the overlap range
scale_dw = weekly.loc[overlap, col].mean() / daily.loc[overlap, col].mean()

# Total factor brings raw hourly onto weekly base
total_scale = scale_hd * scale_dw
hourly["nvda_scaled"] = hourly[col] * total_scale

# ──────────────────────────────────────────────────────────────────────────────
# 4. Save the final scaled hourly series
# ──────────────────────────────────────────────────────────────────────────────
hourly[["nvda_scaled"]].to_csv("nvda_hourly_scaled.csv")

print("\nDone! CSV files written:\n"
      "  • nvda_weekly_raw.csv\n"
      "  • nvda_daily_raw.csv\n"
      "  • nvda_hourly_raw.csv\n"
      "  • nvda_hourly_scaled.csv")
