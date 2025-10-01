import argparse
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

import pandas as pd
import requests


load_dotenv()

def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

def fetch_twelvedata_gold_chunk(start: datetime, end: datetime, interval: str = "1h", apikey: str = None) -> pd.DataFrame:
    """Fetch a single chunk of XAU/USD data from Twelve Data"""
    if not apikey:
        raise ValueError("Twelve Data API key required")
    
    interval_map = {"1h": "1h", "1d": "1day"}
    td_interval = interval_map.get(interval, "1h")
    
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": td_interval,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "format": "JSON",
        "outputsize": 5000
    }
    
    # Prefer header-based auth to avoid exposing keys in URLs

    headers = {"Authorization": f"apikey {apikey}"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)

    # # Fallback: if all header attempts failed or gave 401/403, use query param as last resort
    # if resp is None or resp.status_code in (401, 403):
    #     params_with_key = dict(params)
    #     params_with_key["apikey"] = apikey
    #     resp = requests.get(url, params=params_with_key, timeout=30)
    # resp.raise_for_status()
    data = resp.json()
    
    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"Twelve Data error: {data.get('message', 'Unknown error')}")
    
    values = data.get("values", [])
    if not values:
        return pd.DataFrame()
    
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
    df = df.sort_values("datetime")
    
    # Convert to numeric
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def fetch_twelvedata_gold(start: datetime, end: datetime, interval: str = "1h", apikey: str = None) -> pd.DataFrame:
    """Fetch XAU/USD from Twelve Data, splitting into 1-year chunks if needed"""
    if not apikey:
        raise ValueError("Twelve Data API key required")
    
    total_days = (end - start).days
    if total_days > 365:
        all_dfs = []
        current_start = start
        while current_start < end:
            chunk_end = min(current_start.replace(year=current_start.year + 1), end)
            chunk_df = fetch_twelvedata_gold_chunk(current_start, chunk_end, interval, apikey)
            if not chunk_df.empty:
                all_dfs.append(chunk_df)
            current_start = chunk_end
        if not all_dfs:
            return pd.DataFrame()
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="first")
    else:
        df = fetch_twelvedata_gold_chunk(start, end, interval, apikey)
    
    if df.empty:
        return pd.DataFrame()
    
    # Align to common schema
    df = df.rename(columns={"datetime": "open_time"})
    df["open_time"] = df["open_time"].dt.tz_convert("UTC")
    if interval == "1h":
        df["close_time"] = df["open_time"] + pd.Timedelta(hours=1)
    else:
        df["close_time"] = df["open_time"] + pd.Timedelta(days=1)
    
    ordered = ["open_time", "open", "high", "low", "close" ]

    df = df[[c for c in ordered if c in df.columns]]
    return df


def fetch_gold(start: datetime, end: datetime, interval: str = "1h") -> tuple[pd.DataFrame, str]:
    """Fetch Gold data.
    For 1h, prefer Twelve Data (XAU/USD) 
    Returns: (DataFrame, source_name)
    """
    td_key = os.getenv("TWELVE_DATA_API_KEY")
    try:
        df_td = fetch_twelvedata_gold(start, end, interval, td_key)
        if not df_td.empty:
            return df_td, "twelvedata"
    except Exception as e:
        print(f"Twelve Data failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Gold data and save Parquet")
    parser.add_argument("--interval", default="1h", help="Interval, e.g., 1h, 1d")
    parser.add_argument("--start", required=True, help="Start ISO, e.g., 2022-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="End ISO, e.g., 2023-01-01T00:00:00Z")
    parser.add_argument("--outdir", default="/home/fidec/Test/data/raw/gold/1h", help="Output directory")
    args = parser.parse_args()

    start_dt = parse_iso(args.start)
    end_dt = parse_iso(args.end)
    
    try:
        df, source_name = fetch_gold(start_dt, end_dt, args.interval)
        print(f"✅ Successfully fetched Gold data from {source_name}")
    except Exception as e:
        print(f"❌ Failed to fetch Gold data: {e}")
        return 1
    
    if df.empty:
        print("No data returned.")
        return 1
    
    os.makedirs(args.outdir, exist_ok=True)
    fname = f"{source_name}_gold_{args.interval}_{start_dt.strftime('%Y-%m-%dT%H-%M-%SZ')}_{end_dt.strftime('%Y-%m-%dT%H-%M-%SZ')}.parquet"
    out_path = os.path.join(args.outdir, fname)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python goldFetch.py --interval 1h --start 2020-01-01T00:00:00Z --end 2025-09-22T00:00:00Z
"""

