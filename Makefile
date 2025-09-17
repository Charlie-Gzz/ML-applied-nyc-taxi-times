.PHONY: format lint typecheck test run-api train ingest drift

format:
	black src
	isort src

lint:
	ruff check src

typecheck:
	mypy src

test:
	pytest -q

run-api:
	uvicorn src.app.main:app --reload --port 8000

ingest:
	python -m src.pipelines.ingest --input data/raw/yellow_tripdata_2023-01.parquet --output data/processed/train_sample.parquet

train:
	python -m src.models.train --train data/processed/train_sample.parquet --model_artifact artifacts/model.joblib

drift:
	python -m src.monitoring.drift_report --ref data/processed/train_sample.parquet --cur data/processed/train_sample.parquet --out artifacts/drift_report.html
