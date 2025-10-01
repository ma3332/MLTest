import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import time
import gc
import torch
import shap
import streamlit as st
from sklearn.preprocessing import RobustScaler

path_bitcoin_features = "/home/fidec/Test/data/processed/binance_btcusdt_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z_features.parquet"
path_usdjpy_features = "/home/fidec/Test/data/processed/usdjpy_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z_features.parquet"
path_gold_features = "/home/fidec/Test/data/processed/gold_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z_features.parquet"

df_usdjpy = pd.read_parquet(path_usdjpy_features, engine="pyarrow")
df_gold = pd.read_parquet(path_gold_features, engine="pyarrow")
df_bitcoin = pd.read_parquet(path_bitcoin_features, engine="pyarrow")

df = df_bitcoin.merge(df_usdjpy, on='open_time', how='left').merge(df_gold, on='open_time', how='left')
df['target_next_close'] = df.pop('target_next_close')

# st.title("Data Viewer")
# st.dataframe(df, use_container_width=True, height=700)

# print(df.columns)
# print(df.head())

# Select columns to scale (numeric features except identifiers/target)
exclude_columns = ['open_time', 'target_next_close']
columns_to_scale = [
    col for col in df.columns
    if col not in exclude_columns and np.issubdtype(df[col].dtype, np.number)
]

if columns_to_scale:
    # Ensure open_time is datetime for time-based split
    if not np.issubdtype(df['open_time'].dtype, np.datetime64):
        df['open_time'] = pd.to_datetime(df['open_time'], utc=True, errors='coerce')

    # Time-based split to avoid leakage
    train_ratio = 0.8  # adjust as needed
    cutoff_time = df['open_time'].quantile(train_ratio)
    train_mask = df['open_time'] <= cutoff_time

    # Median imputation computed from training window only
    train_medians = df.loc[train_mask, columns_to_scale].median()
    df[columns_to_scale] = df[columns_to_scale].fillna(train_medians)

    # Fit on training window; transform entire series
    scaler = RobustScaler()
    scaler.fit(df.loc[train_mask, columns_to_scale])
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

