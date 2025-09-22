from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipelines.validate import validate_raw_schema, validate_features


def main(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)

    # 1) Normalize column names first (so validation is case-insensitive)
    df = df.rename(columns=str.lower)

    # 2) RAW SCHEMA VALIDATION (fail fast if required raw columns are missing)
    validate_raw_schema(df)

    # 3) Minimal cleaning & feature engineering
    # Drop obvious outliers
    df = df[(df["trip_distance"] >= 0) & (df["trip_distance"] <= 100)]

    # Engineer time features
    df["pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday

    # Target: duration in minutes
    df["duration_min"] = (
        pd.to_datetime(df["tpep_dropoff_datetime"]) - df["pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # Keep reasonable trips only
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 180)]

    # Coerce numerics in case some columns come in as strings
    for c in ["passenger_count", "trip_distance", "ratecodeid", "payment_type", "vendorid"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep reasonable passenger counts (NYC TLC usually 0..6; allow up to 8)
    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] >= 0) & (df["passenger_count"] <= 8)]

    # Select and rename to model input names
    cols = [
        "vendorid",
        "passenger_count",
        "trip_distance",
        "ratecodeid",
        "payment_type",
        "pickup_hour",
        "pickup_weekday",
        "duration_min",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].dropna()

    rename = {
        "vendorid": "vendor_id",
        "ratecodeid": "rate_code",
    }
    df = df.rename(columns=rename)

    # 4) FEATURE-LEVEL VALIDATION (ranges, nulls, minimal cardinality, etc.)
    validate_features(df)

    # 5) Save processed dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved processed: {output_path}, rows={len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
