from data_ingestion.fetch_data import fetch_data
from preprocessing.preprocess_data import run_full_preprocessing_pipeline_from_df
from model.train_model import run_all_model_training_from_df


if __name__ == "__main__":
    df = fetch_data("dataset/flidar1_raw.csv")
    #df = fetch_data("dataset/flidar1_raw.csv", save=True)
    processed_df = run_full_preprocessing_pipeline_from_df(df)
    run_all_model_training_from_df(processed_df, model_dir="output_models/")