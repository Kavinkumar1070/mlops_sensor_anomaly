# train.py

import os
from data_ingestion.fetch_data import fetch_data
from preprocessing.preprocess_data import run_full_preprocessing_pipeline_from_df
from model.train_model import run_all_model_training_from_df

if __name__ == "__main__":
    azure_blob_url = os.environ.get("AZURE_BLOB_URL")
    if not azure_blob_url:
        raise ValueError("AZURE_BLOB_URL is not set in environment variables.")
    df = fetch_data(azure_blob_url)

    # # Replace this with your actual SAS URL if you're testing manually
    # AZURE_BLOB_URL = "https://kavinmlstorage.blob.core.windows.net/ml-data/flidar1_raw.csv?sp=r&st=2025-08-02T07:58:54Z&se=2025-08-09T16:13:54Z&spr=https&sv=2024-11-04&sr=b&sig=UwEDbhIcVaE0fmtj%2F%2BNbPBrEaWzN6U1TfBa%2FIO%2FOmCg%3D"

    # print("📥 Downloading data from Azure Blob Storage...")
    # df = fetch_data(AZURE_BLOB_URL)

    
    processed_df = run_full_preprocessing_pipeline_from_df(df)
    run_all_model_training_from_df(processed_df, model_dir="output_models/")


