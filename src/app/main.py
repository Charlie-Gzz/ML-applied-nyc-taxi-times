from fastapi import FastAPI
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from src.common.schemas import PredictRequest, PredictResponse
from src.models.predict import load_model, predict_duration

app = FastAPI(title="NYC Taxi Trip Duration API", version="0.1.0")

PRED_COUNTER = Counter("pred_requests_total", "Total prediction requests")

model = None

@app.on_event("startup")
def _load_model():
    global model
    model = load_model()

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global model
    PRED_COUNTER.inc()
    pred = predict_duration(model, req)
    return PredictResponse(predicted_duration_min=float(pred))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
