from __future__ import annotations
from typing import Any
import joblib
import os
import numpy as np
from src.common.schemas import PredictRequest

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")

def load_model():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        # Return a dummy model if none exists (unit-test friendly)
        return {"mean_duration": 12.0}
    return joblib.load(DEFAULT_MODEL_PATH)

def predict_duration(model, req: PredictRequest) -> float:
    # If dummy
    if isinstance(model, dict) and "mean_duration" in model:
        return float(model["mean_duration"])

    x = np.array([[req.vendor_id, req.passenger_count, req.trip_distance,
                   req.pickup_hour, req.pickup_weekday, req.rate_code, req.payment_type]])
    pred = model.predict(x)[0]
    # convert seconds to minutes if needed; here we assume model predicts minutes
    return float(pred)
