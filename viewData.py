import pandas as pd
import streamlit as st

path_bitcoin_features = "/home/fidec/Test/data/processed/binance_btcusdt_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z_features.parquet"
path_usdjpy_features = "/home/fidec/Test/data/processed/usdjpy_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z_features.parquet"
path_gold_features = "/home/fidec/Test/data/processed/gold_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z_features.parquet"

path_bitcoin_raw = "/home/fidec/Test/data/raw/binance/btcusdt/1h/binance_btcusdt_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z.parquet"
path_usdjpy_raw = "/home/fidec/Test/data/raw/usdjpy/1h/twelvedata_usdjpy_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z.parquet"
path_gold_raw = "/home/fidec/Test/data/raw/gold/1h/twelvedata_gold_1h_2020-01-01T00-00-00Z_2025-09-22T00-00-00Z.parquet"

df = pd.read_parquet(path_usdjpy_features, engine="pyarrow")

st.title("Data Viewer")
st.dataframe(df, use_container_width=True, height=700)

print(df.columns)
print(df.head())

# run file: streamlit run viewData.py