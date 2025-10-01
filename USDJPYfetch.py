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

def fetch_twelvedata_usdjpy_chunk(start: datetime, end: datetime, interval: str = "1h", apikey: str = None) -> pd.DataFrame:
    """Fetch a single chunk of USD/JPY data from Twelve Data"""
    if not apikey:
        raise ValueError("Twelve Data API key required")
    
    # Map interval to Twelve Data format
    interval_map = {"1h": "1h", "1d": "1day"}
    td_interval = interval_map.get(interval, "1h")
    
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "USD/JPY",
        "interval": td_interval,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "format": "JSON",
        "outputsize": 5000
    }
    headers = {"Authorization": f"apikey {apikey}"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    # Check for API errors
    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"Twelve Data error: {data.get('message', 'Unknown error')}")
    
    # Extract time series data
    values = data.get("values", [])
    if not values:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
    df = df.sort_values("datetime")
    # Keep datetime as column, don't set as index yet
    
    # Rename columns (FX data doesn't have volume)
    df = df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low", 
        "close": "close"
    })
    
    # Convert to numeric
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def fetch_twelvedata_usdjpy(start: datetime, end: datetime, interval: str = "1h", apikey: str = None) -> pd.DataFrame:
    """Fetch USD/JPY data from Twelve Data, splitting into 1-year chunks if needed"""
    if not apikey:
        raise ValueError("Twelve Data API key required")
    
    # Calculate total days
    total_days = (end - start).days
    df = pd.DataFrame()
    # If more than 1 year, split into chunks
    if total_days > 365:
        print(f"📅 Requested {total_days} days of data. Splitting into 1-year chunks...")
        
        all_dfs = []
       
        current_start = start
        
        while current_start < end:
            # Calculate chunk end (1 year later or final end date)
            chunk_end = min(
                current_start.replace(year=current_start.year + 1),
                end
            )
            
            print(f"🔄 Fetching chunk: {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            try:
                chunk_df = fetch_twelvedata_usdjpy_chunk(current_start, chunk_end, interval, apikey)
                if not chunk_df.empty:
                    all_dfs.append(chunk_df)
                    print(f"✅ Got {len(chunk_df):,} rows")
                else:
                    print("⚠️  No data for this chunk")
            except Exception as e:
                print(f"❌ Error fetching chunk: {e}")
            
            # Move to next year
            current_start = chunk_end
        
        # Combine all chunks
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.sort_values("datetime")
        df = df.drop_duplicates(subset=["datetime"], keep='first')  # Remove duplicates

    else:
        # Single request for periods <= 1 year
        df = fetch_twelvedata_usdjpy_chunk(start, end, interval, apikey)
    # Debug: Check what columns we have
    print(f"🔍 Columns before rename: {list(df.columns)}")
    
    df = df.rename(columns={"datetime": "open_time"})
    print("✅ Renamed datetime to open_time")
    
    # Ensure timezone is UTC (already timezone-aware, so just ensure it's UTC)
    df["open_time"] = df["open_time"].dt.tz_convert("UTC")

    # Create close_time
    if interval == "1h":
        df["close_time"] = df["open_time"] + pd.Timedelta(hours=1)
    else:
        df["close_time"] = df["open_time"] + pd.Timedelta(days=1)
    
    # Add placeholder columns to match schema
    for col in ["quote_asset_volume", "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]:
        df[col] = pd.NA
    
    # Order columns
    ordered = ["open_time", "open", "high", "low", "close"]
    df = df[ordered]
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch USD/JPY data via Twelve Data and save Parquet")
    parser.add_argument("--interval", default="1h", help="Interval, e.g., 1h, 1d")
    parser.add_argument("--start", required=True, help="Start ISO, e.g., 2022-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="End ISO, e.g., 2023-01-01T00:00:00Z")
    parser.add_argument("--outdir", default="/home/fidec/Test/data/raw/usdjpy/1h", help="Output directory")
    parser.add_argument("--source", choices=["twelvedata"], default="twelvedata", help="Data source")
    args = parser.parse_args()

    start_dt = parse_iso(args.start)
    end_dt = parse_iso(args.end)
    
    df = None
    
    # Try Twelve Data first
    if args.source == "twelvedata":
        apikey = os.getenv("TWELVE_DATA_API_KEY")
        if apikey:
            try:
                df = fetch_twelvedata_usdjpy(start_dt, end_dt, args.interval, apikey)
                print("✅ Successfully fetched from Twelve Data")
            except Exception as e:
                print(f"❌ Twelve Data failed: {e}")
                df = None
        else:
            print("TWELVE_DATA_API_KEY not set.")
    
    os.makedirs(args.outdir, exist_ok=True)
    source_name = "twelvedata" if args.source == "twelvedata" else "None"
    fname = f"{source_name}_usdjpy_{args.interval}_{start_dt.strftime('%Y-%m-%dT%H-%M-%SZ')}_{end_dt.strftime('%Y-%m-%dT%H-%M-%SZ')}.parquet"
    out_path = os.path.join(args.outdir, fname)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
# USD/JPY via Twelve Data (needs TWELVE_DATA_API_KEY) - limited historical data
python USDJPYfetch.py --source twelvedata --interval 1h --start 2024-01-01T00:00:00Z --end 2024-06-01T00:00:00Z

# Get free Twelve Data API key at: https://twelvedata.com/pricing

"""

