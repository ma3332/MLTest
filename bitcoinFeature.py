import argparse
import os
from typing import Optional
import numpy as np

import pandas as pd


def compute_ema(series: pd.Series, span: int, adjust: bool = False) -> pd.Series:
    return series.ewm(span=span, adjust=adjust, min_periods=span).mean()


def add_ema(df: pd.DataFrame, close_col: str = "close", spans=(12, 26)) -> pd.DataFrame:
    for span in spans:
        df[f"ema_{span}"] = compute_ema(df[close_col], span)
    return df


def add_macd(df: pd.DataFrame, close_col: str = "close", fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = compute_ema(df[close_col], fast)
    ema_slow = compute_ema(df[close_col], slow)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd - macd_signal
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    return df


def add_rsi(df: pd.DataFrame, close_col: str = "close", period: int = 14) -> pd.DataFrame:
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    return df


def add_bollinger_bands(df: pd.DataFrame, close_col: str = "close", window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mavg = df[close_col].rolling(window=window, min_periods=window).mean()
    mstd = df[close_col].rolling(window=window, min_periods=window).std(ddof=0)
    upper = mavg + num_std * mstd
    lower = mavg - num_std * mstd
    df["bb_mean"] = mavg
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    # Extras: bandwidth and %b
    df["bb_bandwidth"] = (upper - lower) / mavg
    df["bb_pctb"] = (df[close_col] - lower) / (upper - lower)
    return df


def add_cci(df: pd.DataFrame, high_col: str = "high", low_col: str = "low", close_col: str = "close", period: int = 20) -> pd.DataFrame:
    tp = (df[high_col] + df[low_col] + df[close_col]) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mean_dev = tp.rolling(window=period, min_periods=period).apply(lambda x: (abs(x - x.mean())).mean(), raw=False)
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    df["cci"] = cci
    return df


def load_parquet(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
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
    add_ema(df)
    add_macd(df)
    add_rsi(df)
    add_bollinger_bands(df)
    add_cci(df)
      
    # Historical log close returns (lagged)
    df['log_close_return_lag1'] = np.log(df['close'] / df['close'].shift(1))
    df['log_close_return_lag2'] = np.log(df['close'].shift(1) / df['close'].shift(2))
    df['log_close_return_lag3'] = np.log(df['close'].shift(2) / df['close'].shift(3))

    # Historical log high returns (lagged)
    df['log_high_return_lag1'] = np.log(df['high'] / df['high'].shift(1))
    df['log_high_return_lag2'] = np.log(df['high'].shift(1) / df['high'].shift(2))
    df['log_high_return_lag3'] = np.log(df['high'].shift(2) / df['high'].shift(3))

    # Historical log low returns (lagged)
    df['log_low_return_lag1'] = np.log(df['low'] / df['low'].shift(1))
    df['log_low_return_lag2'] = np.log(df['low'].shift(1) / df['low'].shift(2))
    df['log_low_return_lag3'] = np.log(df['low'].shift(2) / df['low'].shift(3))
    
    # Current intraday patterns (available at time t)
    df['high_close_ratio'] = np.log(df['high'] / df['close'])
    df['low_close_ratio'] = np.log(df['low'] / df['close'])
    df['open_close_ratio'] = np.log(df['open'] / df['close'])
    
    # EMA-derived scale-free features
    df['price_over_ema12'] = (df['close'] / df['ema_12']) - 1
    df['ema_spread_pct'] = (df['ema_12'] / df['ema_26']) - 1
    # Optional log-ratios (commented, enable if preferred)
    # df['log_price_minus_log_ema12'] = np.log(df['close']) - np.log(df['ema_12'])
    # df['log_ema12_minus_log_ema26'] = np.log(df['ema_12']) - np.log(df['ema_26'])
    
    # Volatility measures (using past data)
    df['log_range'] = np.log(df['high'] / df['low'])
    df['realized_volatility'] = df['log_close_return_lag1'].rolling(24).std()  # 24h rolling vol
    
    # Bollinger-derived distances (already normalized by mean)
    df['dist_to_upper'] = (df['bb_upper'] - df['close']) / df['bb_mean']
    df['dist_to_lower'] = (df['close'] - df['bb_lower']) / df['bb_mean']
    
    # Volume patterns (current and lagged)
    df['log_volume'] = np.log1p(df['volume'])
    df['log_trades'] = np.log1p(df['num_trades'])

    df["target_next_close"] = np.log(df['close'].shift(-1) / df['close']) # price of (t+1)/ price of (t)

    drop_cols = [
        "ignore",
        "close_time",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
      
    return df


def main(input_path: str, output_path: Optional[str] = None) -> str:
    df = load_parquet(input_path)
    print(df.head())
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
python bitcoinFeature.py --in /home/fidec/Test/data/raw/binance/btcusdt/1h/binance_btcusdt_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z.parquet
or specify output
python bitcoinFeature.py --in /home/fidec/Test/data/raw/binance/btcusdt/1h/binance_btcusdt_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z.parquet --out /home/fidec/Test/data/processed/btcusdt_1h_features_1.parquet
"""

