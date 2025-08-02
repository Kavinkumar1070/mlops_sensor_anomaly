import pandas as pd
import pyodbc
import os
import sys
from exp_log.exception import CustomException
from exp_log.logger import logging
from dataclasses import dataclass

# def fetch_data():
#     conn = pyodbc.connect(
#         'DRIVER={ODBC Driver 17 for SQL Server};'
#         'SERVER=your_server;DATABASE=your_db;UID=user;PWD=password'
#     )
#     query = "SELECT * FROM your_table"
#     df = pd.read_sql(query, conn)
#     conn.close()
#     df.to_csv("raw_data.csv", index=False)
#     return df


# def fetch_data(input_path: str, save: bool = False, save_path: str = "data/raw_data.csv"):
#     try:
#         logging.info("📥 Starting data ingestion...")
#         df = pd.read_csv(input_path, low_memory=False)
#         if save:
#             df.to_csv(save_path, index=False)
#             logging.info(f"📂 Data saved to {save_path}")
#         logging.info(f"✅ Ingestion complete. Shape: {df.shape}")
#         return df
#     except Exception as e:
#         logging.error("❌ Error during data ingestion")
#         raise CustomException(e, sys)

# data_ingestion/fetch_data.py

import os
import sys
import pandas as pd
import requests
from io import StringIO
from exp_log.exception import CustomException
from exp_log.logger import logging

def fetch_data(sas_url: str) -> pd.DataFrame:
    try:
        logging.info("📥 Fetching dataset from Azure Blob Storage...")
        response = requests.get(sas_url)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        logging.info("✅ Data fetched successfully.")
        return df
    except Exception as e:
        logging.error("❌ Error fetching data from Azure Blob Storage.")
        raise CustomException(e, sys)


