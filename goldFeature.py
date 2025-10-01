import argparse
import os
from typing import Optional
import numpy as np

import pandas as pd

def load_parquet(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time")
    # Ensure open_time is the first column
    cols = ["open_time"] + [c for c in df.columns if c != "open_time"]
    df = df[cols]
    return df


def save_parquet(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Ensure open_time remains a column in the saved file
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name is not None:
        df = df.reset_index(drop=False)
        # If reset added an 'index' column and we already have open_time, drop it
        if 'index' in df.columns and 'open_time' in df.columns:
            df = df.drop(columns=['index'])
    # Reorder to keep open_time first
    if 'open_time' in df.columns:
        cols = ["open_time"] + [c for c in df.columns if c != "open_time"]
        df = df[cols]
    df.to_parquet(output_path, index=False)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
       
    # Historical log close returns (lagged)
    df['log_close_return_lag1'] = np.log(df['close'] / df['close'].shift(1))
    df['log_close_return_lag2'] = np.log(df['close'].shift(1) / df['close'].shift(2))
    df['log_close_return_lag3'] = np.log(df['close'].shift(2) / df['close'].shift(3))
    
    # Current intraday patterns (available at time t)
    df['high_close_ratio'] = np.log(df['high'] / df['close'])
    df['low_close_ratio'] = np.log(df['low'] / df['close'])
        
    # Volatility measures (using past data)
    df['log_range'] = np.log(df['high'] / df['low'])
    df.columns = ['open_time', 'open_gold', 'high_gold', 'low_gold', 'close_gold', 'log_close_return_lag1_gold', 'log_close_return_lag2_gold', 'log_close_return_lag3_gold', 'high_close_ratio_gold', 'low_close_ratio_gold', 'log_range_gold']

    return df

def main(input_path: str, output_path: Optional[str] = None) -> str:
    df = load_parquet(input_path)
    feat_df = build_features(df)
    if output_path is None:
        base = os.path.basename(input_path).replace(".parquet", "")
        output_path = f"/home/fidec/Test/data/processed/{base}_features.parquet"
    save_parquet(feat_df, output_path)
    print(f"Saved features to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute technical indicators and save features")
    parser.add_argument("--in", dest="input_path", required=True, help="Input Parquet path from data/raw")
    parser.add_argument("--out", dest="output_path", default=None, help="Output Parquet path in data/processed")
    args = parser.parse_args()
    main(args.input_path, args.output_path)

"""
cd /home/fidec/Test
python goldFeature.py --in /home/fidec/Test/data/raw/gold/1h/twelvedata_gold_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z.parquet
or specify output
python goldFeature.py --in /home/fidec/Test/data/raw/gold/1h/twelvedata_gold_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z.parquet --out /home/fidec/Test/data/processed/gold_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z_features.parquet
"""

