from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import mlflow
from pathlib import Path

FEATURES = ['vendor_id','passenger_count','trip_distance','pickup_hour','pickup_weekday','rate_code','payment_type']
TARGET = 'duration_min'

def main(train_path: str, model_artifact: str):
    df = pd.read_parquet(train_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("nyc_taxi_duration")

    with mlflow.start_run():
        params = {"n_estimators": 200, "random_state": 42}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)

        mlflow.log_params(params)
        mlflow.log_metric("val_mae", float(mae))

        Path(model_artifact).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_artifact)
        mlflow.log_artifact(model_artifact)
        print(f"Model saved to {model_artifact}; val MAE={mae:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--model_artifact", default="artifacts/model.joblib")
    args = parser.parse_args()
    main(args.train, args.model_artifact)
