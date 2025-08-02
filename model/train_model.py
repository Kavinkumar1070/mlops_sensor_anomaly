import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from exp_log.exception import CustomException
from exp_log.logger import logging

warnings.filterwarnings('ignore')

def preprocess_data(df, model_dir):
    try:
        os.makedirs(model_dir, exist_ok=True)
        encoders = {}

        categorical_cols = ['wind_direction']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        int_columns = ['lidar_enabled', 'height', 'internal_flag', 'external_flag']
        for col in int_columns:
            df[col] = df[col].astype(int)

        joblib.dump(encoders, os.path.join(model_dir, 'encoders.pkl'))
        logging.info("✅ Encoders saved.")
        return df
    except Exception as e:
        logging.error("❌ Error during preprocessing.")
        raise CustomException(e, sys)

def save_anomaly_label_map(model_dir):
    try:
        label_map = {
            'normal': 0,
            'Wind_dir_fail': 1,
            'maintenance': 1,
            'sensor_fail1': 1,
            'sensor_fail2': 1,
            'wind_spd_fail': 1
        }
        with open(os.path.join(model_dir, "Anomaly_label_encoder.pkl"), "wb") as f:
            pickle.dump(label_map, f)
        logging.info("✅ Anomaly label map saved.")
    except Exception as e:
        logging.error("❌ Error saving anomaly label map.")
        raise CustomException(e, sys)

def build_anomaly_model(df, model_dir):
    try:
        df['Failure_binary'] = df['type_failure'].apply(lambda x: 0 if x == 'normal' else 1)

        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        X = df.drop(columns=['type_failure', 'Failure_binary'])
        y = df['Failure_binary']

        selector = RFECV(estimator=rf, step=1, cv=StratifiedKFold(n_splits=5),
                         scoring='f1_macro', n_jobs=-1)
        selector.fit(X, y)

        selected_features = ['wind_direction', 'internal_flag', 'external_flag', 'lidar_enabled']
        joblib.dump(selected_features, os.path.join(model_dir, "Anomaly_selected_features.pkl"))

        X = df[selected_features]
        y = df['Failure_binary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        logging.info("=== Anomaly Detection Model ===")
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        logging.info("\n" + classification_report(y_test, y_pred))
        logging.info(f"F1 Score: {f1:.3f}")
        logging.info(f"ROC AUC: {auc:.3f}")
        logging.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        if f1 < 0.80:
            raise ValueError(f"❌ F1 Score too low for Anomaly model: {f1:.3f}. Training stopped.")

        model.fit(X, y)
        joblib.dump(model, os.path.join(model_dir, 'Anomaly_model.pkl'))
        joblib.dump(model, os.path.join(model_dir, 'Anomaly_model.joblib'))

        logging.info("✅ Anomaly model trained and saved.")
    except Exception as e:
        logging.error("❌ Error in building anomaly model.")
        raise CustomException(e, sys)

def build_unknown_model(df, model_dir):
    try:
        df1 = df[df['type_failure'] != 'normal']
        X = df1.drop(columns=['type_failure', 'Failure_binary'])
        y = df1['type_failure']

        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        joblib.dump(le, os.path.join(model_dir, "failure_label_encoder.pkl"))

        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        selector = RFECV(estimator=rf, step=1, cv=StratifiedKFold(n_splits=5),
                         scoring='f1_macro', n_jobs=-1)
        selector.fit(X, y_enc)

        selected_features = ['wind_speed', 'wind_direction']
        joblib.dump(selected_features, os.path.join(model_dir, "failure_selected_features.pkl"))

        X = df1[selected_features]
        y_enc = le.transform(df1['type_failure'].astype(str))
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, stratify=y_enc, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average='macro')

        logging.info("=== Unknown Failure Classification Model ===")
        logging.info("\n" + classification_report(y_test, y_pred))
        logging.info(f"F1 Score: {f1:.3f}")
        logging.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        if f1 < 0.80:
            raise ValueError(f"❌ F1 Score too low for Unknown model: {f1:.3f}. Training stopped.")

        joblib.dump(model, os.path.join(model_dir, 'failure_model.pkl'))
        joblib.dump(model, os.path.join(model_dir, 'failure_model.joblib'))
        logging.info("✅ Unknown model trained and saved.")
    except Exception as e:
        logging.error("❌ Error in building unknown failure model.")
        raise CustomException(e, sys)
    
def run_all_model_training_from_df(df: pd.DataFrame, model_dir: str) -> str:
    try:
        logging.info("🚀 Model training pipeline started...")

        # Preprocess encodings
        df = preprocess_data(df, model_dir)
        save_anomaly_label_map(model_dir)

        # Train anomaly model
        logging.info("🚀 Training Anomaly Detection Model...")
        build_anomaly_model(df, model_dir)

        # Train unknown failure model
        logging.info("🚀 Training Unknown Failure Classification Model...")
        build_unknown_model(df, model_dir)

        logging.info(f"✅ All models built and saved in '{model_dir}'.")
        return 'model build success'

    except Exception as e:
        logging.error("❌ Error in full training pipeline.")
        raise CustomException(e, sys)
