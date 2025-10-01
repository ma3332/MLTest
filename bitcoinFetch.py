import argparse
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests


BINANCE_API = "https://api.binance.com/api/v3/klines"


def parse_iso_datetime(value: str) -> int:
    """Parse ISO-8601 datetime string to Binance milliseconds since epoch."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid datetime format: {value}") from exc
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    """Fetch klines from Binance in pages and return a DataFrame."""
    all_rows: List[List] = []
    current = start_ms

    while True:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = requests.get(BINANCE_API, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        # Advance to next window: last close time + 1 ms
        next_ms = rows[-1][6] + 1
        if next_ms >= end_ms:
            break
        current = next_ms

    if not all_rows:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
    ]
    
    df = pd.DataFrame(all_rows, columns=cols)

    # Types and timestamps
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").set_index("open_time")
    return df


def save_parquet(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Binance klines and save to Parquet")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, e.g., BTCUSDT")
    parser.add_argument("--interval", default="1h", help="Binance interval, e.g., 1m, 5m, 1h, 1d")
    parser.add_argument("--start", required=True, type=parse_iso_datetime, help="Start datetime ISO (e.g., 2022-01-01T00:00:00Z)")
    parser.add_argument("--end", required=True, type=parse_iso_datetime, help="End datetime ISO (e.g., 2022-12-31T00:00:00Z)")
    parser.add_argument("--outdir", default="/home/fidec/Test/data/raw/binance/btcusdt/1h", help="Output directory for Parquet file")

    args = parser.parse_args(argv)
    
    df = fetch_klines(args.symbol, args.interval, args.start, args.end)
    if df.empty:
        print("No data returned for given range.")
        return 1

    # Build filename
    start_iso = datetime.fromtimestamp(args.start / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    end_iso = datetime.fromtimestamp(args.end / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    fname = f"binance_{args.symbol.lower()}_{args.interval}_{start_iso}_{end_iso}.parquet"
    out_path = os.path.join(args.outdir, fname)

    save_parquet(df, out_path)
    print(f"Saved {len(df):,} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Example: 
python bitcoinFetch.py --symbol BTCUSDT --interval 1h --start 2024-01-01T00:00:00Z --end 2024-02-01T00:00:00Z

python bitcoinFetch.py --symbol BTCUSDT --interval 1d --start 2020-01-01T00:00:00Z --end 2025-09-22T00:00:00Z --outdir /home/fidec/Test/data/raw
"""