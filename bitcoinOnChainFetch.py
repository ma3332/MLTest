import requests
import pandas as pd
import os
from dotenv import load_dotenv
import json

load_dotenv()

apiKey = os.getenv("bitcoinisdata")

# Method 1: Traditional approach (limited to 120 columns)
api_url = "https://bitcoinisdata.com/api/get_data"
params = {
    "api_key": apiKey,             # Replace with your actual API key
    "start_block": 610642,                          # Example start block: 2020-01-01
    "end_block": 915758,                            # Example end block: 2025-09-22
    "format": "json",                               # Request JSON format
    "columns_list": ["heights", "difficulty", "coins"]   # List of columns
}

params_test = {
    "api_key": apiKey,             # Replace with your actual API key
    "start_block": 915700,                          # Example start block: 2020-01-01
    "end_block": 915720,                            # Example end block: 2025-09-22
    "format": "json",                               # Request JSON format
    "columns_list": ["heights", "difficulty", "coins"]   # List of columns
}

  
response = requests.get(api_url, params=params_test)
if response.status_code == 200:
    json_data = response.json()
    df = pd.DataFrame(json_data)
    print(df)
else:
    print('Error:', response)