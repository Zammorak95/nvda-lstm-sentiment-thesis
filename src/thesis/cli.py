from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from thesis.config import settings

app = typer.Typer(add_completion=False, help="Thesis utilities for data retrieval & checks.")
console = Console()

API_BASE = "https://api.stockdata.org/v1/data"


def _http_get(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """Wrapper with a short retry on status 429/5xx."""
    session = requests.Session()
    attempts = 0
    while True:
        attempts += 1
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504) and attempts < 3:
            wait = 1.5 * attempts
            rprint(f"[yellow]HTTP {resp.status_code} — retrying in {wait:.1f}s...[/yellow]")
            time.sleep(wait)
            continue
        return resp


def _save_csv(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    rprint(f"[green]Saved[/green] {len(df):,} rows → {out}")


@app.command("check-api")
def check_api(
    symbol: str = typer.Argument("NVDA", help="Ticker symbol to test."),
    intraday: bool = typer.Option(True, help="Use the intraday endpoint for the check."),
) -> None:
    """
    Minimal end-to-end check: token -> request -> parse -> preview.
    """
    token = settings.stockdata_api_token.get_secret_value()
    if not token:
        rprint("[red]Missing STOCKDATA_API_TOKEN[/red]")
        raise typer.Exit(code=1)

    end = datetime.utcnow()
    start = end - timedelta(days=10)  # small window (handles weekends)

    url = f"{API_BASE}/intraday" if intraday else f"{API_BASE}/eod"
    params = {
        "api_token": token,
        "symbols": symbol,
        "interval": "minute" if intraday else None,
        "date_from": start.strftime("%Y-%m-%d"),
        "date_to": end.strftime("%Y-%m-%d"),
        "sort": "asc",
        "limit": 5,
    }
    # remove None keys
    params = {k: v for k, v in params.items() if v is not None}

    rprint(f"[cyan]GET[/cyan] {url} {params}")
    resp = _http_get(url, params, timeout=30)
    rprint(f"HTTP {resp.status_code}")

    if resp.status_code != 200:
        rprint(resp.text[:500])
        raise typer.Exit(code=2)

    js = resp.json()
    if "data" not in js:
        rprint("[red]No 'data' field in response[/red]; keys: ", list(js.keys()))
        raise typer.Exit(code=3)

    df = pd.DataFrame(js["data"])
    if df.empty:
        rprint("[yellow]No rows returned[/yellow]")
        raise typer.Exit(code=4)

    # Pretty preview
    preview_cols = list(df.columns)[:8]
    table = Table(title=f"{symbol} — sample", show_lines=False)
    for c in preview_cols:
        table.add_column(c)
    for _, row in df.head(5).iterrows():
        table.add_row(*[str(row[c]) for c in preview_cols])
    console.print(table)
    rprint("[green]API basic flow ✓[/green]")


@app.command("fetch-eod")
def fetch_eod(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g., NVDA"),
    years: int = typer.Option(5, min=1, max=10, help="How many years of daily data."),
    out: Path = typer.Option(Path("data/eod.csv"), help="Output CSV path."),
    limit: int = typer.Option(10000, help="API page limit (provider cap)."),
) -> None:
    """
    Fetch daily EOD bars for a back period and save CSV.
    """
    token = settings.stockdata_api_token.get_secret_value()
    end = datetime.utcnow()
    start = end - timedelta(days=365 * years)
    url = f"{API_BASE}/eod"
    params = {
        "api_token": token,
        "symbols": symbol,
        "date_from": start.strftime("%Y-%m-%d"),
        "date_to": end.strftime("%Y-%m-%d"),
        "sort": "asc",
        "limit": limit,
    }

    rprint(f"[cyan]GET[/cyan] {url} {params}")
    resp = _http_get(url, params, timeout=60)
    if resp.status_code != 200:
        rprint(f"[red]HTTP {resp.status_code}[/red]\n{resp.text[:500]}")
        raise typer.Exit(code=2)

    js = resp.json()
    df = pd.DataFrame(js.get("data", []))
    if df.empty:
        rprint("[yellow]No data returned[/yellow]")
        raise typer.Exit(code=4)

    _save_csv(df, out)


@app.command("fetch-intraday")
def fetch_intraday(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g., NVDA"),
    days: int = typer.Option(30, min=1, help="Days back to fetch (provider/plan limits apply)."),
    step_days: int = typer.Option(7, min=1, help="Chunk size per request to respect limits."),
    limit: int = typer.Option(10000, help="API page limit (provider cap)."),
    out: Path = typer.Option(Path("data/intraday.csv"), help="Output CSV path."),
    sleep_sec: float = typer.Option(1.0, min=0.0, help="Pause between requests to avoid 429."),
) -> None:
    """
    Fetch minute-level bars in weekly chunks and save CSV.
    """
    token = settings.stockdata_api_token.get_secret_value()
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    delta = timedelta(days=step_days)

    url = f"{API_BASE}/intraday"
    all_rows: List[dict] = []
    cur = start

    while cur < end:
        date_from = cur.strftime("%Y-%m-%d")
        date_to = min(cur + delta, end).strftime("%Y-%m-%d")
        params = {
            "api_token": token,
            "symbols": symbol,
            "interval": "minute",
            "date_from": date_from,
            "date_to": date_to,
            "sort": "asc",
            "limit": limit,
        }
        rprint(f"[cyan]GET[/cyan] {url} {params}")
        resp = _http_get(url, params, timeout=60)
        if resp.status_code != 200:
            rprint(f"[red]HTTP {resp.status_code}[/red] — skipping chunk; body: {resp.text[:300]}")
        else:
            chunk = resp.json().get("data", [])
            rprint(f"  -> rows: {len(chunk)}")
            all_rows.extend(chunk)

        cur += delta
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows)
    if df.empty:
        rprint("[yellow]No minute-level data retrieved[/yellow]")
        raise typer.Exit(code=4)

    _save_csv(df, out)
