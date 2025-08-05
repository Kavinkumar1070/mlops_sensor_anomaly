from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import io
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
from fastapi.responses import JSONResponse
from preprocessing.preprocess_data import run_full_preprocessing_pipeline_from_df

# Load trained assets
MODEL_DIR = "output_models"
anomaly_model = joblib.load(os.path.join(MODEL_DIR, "Anomaly_model.pkl"))
unknown_model = joblib.load(os.path.join(MODEL_DIR, "failure_model.pkl"))
anomaly_encoder = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
anomaly_features = joblib.load(os.path.join(MODEL_DIR, "Anomaly_selected_features.pkl"))
unknown_features = joblib.load(os.path.join(MODEL_DIR, "failure_selected_features.pkl"))
unknown_label_encoder = joblib.load(os.path.join(MODEL_DIR, "failure_label_encoder.pkl"))
anomaly_label_map = joblib.load(os.path.join(MODEL_DIR, "Anomaly_label_encoder.pkl"))

app = FastAPI()

# 🎯 Pydantic Model
class SensorRecord(BaseModel):
    datetimekey: str
    height: int
    wind_direction: float
    wind_speed: float
    internal_flag: int
    external_flag: int
    lidar_enabled: int

import pandas as pd
import joblib
import os

def predict_failure(df: pd.DataFrame) -> pd.DataFrame:
    # Load models and encoders
    model_dir = "output_models"
    anomaly_model = joblib.load(os.path.join(model_dir, 'Anomaly_model.pkl'))
    unknown_model = joblib.load(os.path.join(model_dir, 'failure_model.pkl'))

    anomaly_features = joblib.load(os.path.join(model_dir, "Anomaly_selected_features.pkl"))
    unknown_features = joblib.load(os.path.join(model_dir, "failure_selected_features.pkl"))

    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    unknown_label_encoder = joblib.load(os.path.join(model_dir, "failure_label_encoder.pkl"))
    
    with open(os.path.join(model_dir, "Anomaly_label_encoder.pkl"), "rb") as f:
        anomaly_label_map = joblib.load(f)

    # 🔁 Preprocessing
    # Encode categorical column(s)
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))
    
    # Convert int columns
    for col in ['lidar_enabled', 'height', 'internal_flag', 'external_flag']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    #print(df)
    # ✅ Predict anomaly
    df['Failure_binary'] = anomaly_model.predict(df[anomaly_features])
    
    # Map binary to label
    reverse_map = {v: k for k, v in anomaly_label_map.items()}
    df['type_failure_pred'] = df['Failure_binary'].map(reverse_map)

    # 🔍 If predicted as normal, apply unknown failure model
    mask = df['type_failure_pred'] == 'normal'
    if mask.sum() > 0:
        df_unknown = df.loc[mask, unknown_features]
        preds = unknown_model.predict(df_unknown)
        df.loc[mask, 'type_failure_pred'] = unknown_label_encoder.inverse_transform(preds)

    return df

@app.get("/")
def read_root():
    return {"message": "🎉 FastAPI app deployed via Azure CD pipeline!"}

@app.post("/predict_wide_record")
def predict_wide_record(record: Dict[str, Any]):
    # Extract the content of 'additionalProp1' and convert it to a DataFrame
    raw_df = pd.json_normalize(record['additionalProp1'])  # Flatten the nested dictionary
    
    # Convert wind direction and speed columns to numeric format
    for i in range(1, 12):
        wind_direction_col = f"wdirtnavg{i}"
        wind_speed_col = f"hwsavg{i}"
    
        if wind_direction_col in raw_df.columns:
            raw_df[wind_direction_col] = pd.to_numeric(raw_df[wind_direction_col], errors='coerce')
        if wind_speed_col in raw_df.columns:
            raw_df[wind_speed_col] = pd.to_numeric(raw_df[wind_speed_col], errors='coerce')

    # Preprocess and run the full pipeline
    processed_df = run_full_preprocessing_pipeline_from_df(raw_df)
    #print('processed_df :',processed_df)
    result_df = predict_failure(processed_df)
    #print('result_df :',result_df)
    return result_df[['height', 'hour','type_failure_pred']].to_dict(orient="records")

# 📦 Batch endpoint
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    raw_df = pd.read_csv(io.BytesIO(contents))
    processed_df = run_full_preprocessing_pipeline_from_df(raw_df)
    result_df = predict_failure(processed_df)
    return result_df[['height', 'hour','type_failure_pred']].to_dict(orient="records")

