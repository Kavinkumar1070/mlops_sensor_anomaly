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
    processed_df = run_full_preprocessing_pipeline_from_df(df)
    run_all_model_training_from_df(processed_df, model_dir="output_models/")
