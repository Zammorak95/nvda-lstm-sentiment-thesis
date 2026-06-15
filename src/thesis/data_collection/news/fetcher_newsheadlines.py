#!/usr/bin/env python3

import os
import csv
import json
import time
from datetime import date, datetime, timedelta

import requests

BASE_URL = "https://api.stockdata.org/v1/news/all"
SYMBOL = "NVDA"
OUT_DIR = "/home/zammorak/News"
LIMIT_PER_DAY = 25

API_TOKEN = os.environ.get("STOCKDATA_API_TOKEN")
if not API_TOKEN:
    raise SystemExit("Missing STOCKDATA_API_TOKEN environment variable.")

session = requests.Session()
session.headers.update({"User-Agent": "nvda-news-2025-downloader/1.0"})


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def month_start_end(d: date):
    start = d.replace(day=1)
    if d.month == 12:
        next_month = d.replace(year=d.year + 1, month=1, day=1)
    else:
        next_month = d.replace(month=d.month + 1, day=1)
    end = next_month - timedelta(days=1)
    return start, end


def month_filename(year: int, month: int):
    return os.path.join(OUT_DIR, f"{SYMBOL}_news_{year:04d}_{month:02d}.csv")


def safe_get(params: dict, retries: int = 6) -> dict:
    backoff = 2
    last_err = None

    for attempt in range(retries):
        try:
            r = session.get(BASE_URL, params=params, timeout=30)

            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue

            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

            data = r.json()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"API error payload: {data['error']}")

            return data

        except (requests.RequestException, ValueError, RuntimeError) as e:
            last_err = e
            if attempt == retries - 1:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)

    raise RuntimeError(f"Request failed after retries: {last_err}")


def fetch_day(day: date):
    params = {
        "api_token": API_TOKEN,
        "symbols": SYMBOL,
        "filter_entities": "true",
        "language": "en",
        "published_on": day.isoformat(),
        "limit": str(LIMIT_PER_DAY),
        "page": "1",
    }
    data = safe_get(params)
    meta = data.get("meta", {})
    articles = data.get("data", [])
    return articles[:LIMIT_PER_DAY], meta


def flatten_article(a: dict):
    return {
        "symbol_target": SYMBOL,
        "uuid": a.get("uuid", ""),
        "published_at": a.get("published_at", ""),
        "title": a.get("title", ""),
        "description": a.get("description", ""),
        "keywords": a.get("keywords", ""),
        "snippet": a.get("snippet", ""),
        "url": a.get("url", ""),
        "source": (a.get("source") or a.get("domain") or a.get("source_name") or ""),
        "image_url": (a.get("image_url") or a.get("imageUrl") or ""),
        "entities_json": json.dumps(a.get("entities", []), ensure_ascii=False),
        "raw_json": json.dumps(a, ensure_ascii=False),
    }


def write_month_csv(path: str, rows: list[dict]):
    fieldnames = [
        "symbol_target",
        "uuid",
        "published_at",
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
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ensure_out_dir()

    start_date = date(2020, 1, 1)
    end_date = date(2020, 12, 31)

    cur = start_date.replace(day=1) 

    while cur <= end_date:
        m_start, m_end = month_start_end(cur)
        if m_start < start_date:
            m_start = start_date
        if m_end > end_date:
            m_end = end_date

        out_path = month_filename(cur.year, cur.month)

        if os.path.exists(out_path):
            print(f"[SKIP] {out_path} exists")
        else:
            print(f"[MONTH] {cur.year:04d}-{cur.month:02d} ({m_start} .. {m_end})")
            month_rows = []
            nonzero_days = 0

            for d in daterange(m_start, m_end):
                try:
                    articles, meta = fetch_day(d)
                    found = meta.get("found", 0)
                    print(f"  [DAY] {d} found={found} returned={len(articles)}")

                    if articles:
                        nonzero_days += 1
                        for a in articles:
                            month_rows.append(flatten_article(a))

                except Exception as e:
                    print(f"  [ERR] {d}: {e}")

                time.sleep(0.25)

            if month_rows:
                write_month_csv(out_path, month_rows)
                print(f"[SAVED] {out_path} rows={len(month_rows)} nonzero_days={nonzero_days}")
            else:
                print(f"[NO DATA] {cur.year:04d}-{cur.month:02d} -> not saving CSV (0 rows)")

        # next month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month + 1, day=1)


if __name__ == "__main__":
    main()
