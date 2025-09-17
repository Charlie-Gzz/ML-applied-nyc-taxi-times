from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

def main(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)
    # Minimal cleaning & feature engineering demo
    df = df.rename(columns=str.lower)
    # Drop obvious outliers
    df = df[(df['trip_distance'] >= 0) & (df['trip_distance'] <= 100)]
    # engineer time features
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    # target: duration in minutes
    df['duration_min'] = (pd.to_datetime(df['tpep_dropoff_datetime']) - df['pickup_datetime']).dt.total_seconds() / 60.0
    df = df[(df['duration_min'] > 0) & (df['duration_min'] <= 180)]
    cols = ['vendorid','passenger_count','trip_distance','ratecodeid','payment_type','pickup_hour','pickup_weekday','duration_min']
    cols = [c for c in cols if c in df.columns]
    df = df[cols].dropna()
    # rename to model input names
    rename = {
        'vendorid':'vendor_id',
        'ratecodeid':'rate_code',
    }
    df = df.rename(columns=rename)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved processed: {output_path}, rows={len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
