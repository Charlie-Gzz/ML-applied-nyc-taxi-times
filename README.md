# AI/ML Applied SWE Starter — NYC Taxi Trip Duration

**Goal:** Learn the full ML product workflow: data ingestion → validation → feature store (stub) → training with tracking → deployment (FastAPI) → monitoring (drift/quality) → CI/CD → (optional) continuous retraining.

### Tech Stack (initial)
- Python 3.11
- **FastAPI** (inference API) + **Uvicorn**
- **scikit-learn** / **XGBoost** (baseline model)
- **MLflow** (experiment tracking/registry)
- **Prefect** (orchestration) — optional; stubbed
- **Great Expectations** (data validation)
- **Evidently** (drift monitoring)
- **pre-commit**: ruff, black, isort, mypy
- **pytest** for unit/integration tests
- **Docker** (containerize API)
- **GitHub Actions** (CI)

### Dataset
Use the public NYC TLC trip records. Start with a single month (CSV/Parquet).  
Download example:
```
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet -O data/raw/yellow_tripdata_2023-01.parquet
```

### Quickstart
```bash
# 1) Create and activate a venv (example with uv or python -m venv)
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run quality checks
make lint && make typecheck && make test

# 4) Ingest + preprocess a sample
python -m src.pipelines.ingest --input data/raw/yellow_tripdata_2023-01.parquet --output data/processed/train_sample.parquet

# 5) Train a baseline model with MLflow tracking
python -m src.models.train --train data/processed/train_sample.parquet --model_artifact artifacts/model.joblib

# 6) Start the API
make run-api
# or
uvicorn src.app.main:app --reload

# 7) Request a prediction (example)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @examples/predict_sample.json

# 8) Run drift report (toy)
python -m src.monitoring.drift_report --ref data/processed/train_sample.parquet --cur data/processed/train_sample.parquet --out artifacts/drift_report.html
```

### Repo Layout
```
src/
  app/            # FastAPI app
  pipelines/      # ingestion, validation, feature building
  features/       # (feature store stubs/hooks)
  models/         # training, evaluation, registry hooks
  monitoring/     # drift/quality jobs
  common/         # utils, schemas, config
data/
  raw/            # raw datasets
  processed/      # cleaned/train-ready datasets
artifacts/        # models, reports
infra/            # k8s manifests (optional)
.github/workflows # CI jobs
```

### Roadmap (suggested)
- Sprint 1 (MVP): ingest → validate → baseline model → API → tests → CI
- Sprint 2: logging + monitoring + drift reports + MLflow registry + Docker
- Sprint 3: scheduled retraining (Prefect), basic feature store stub, K8s optional

