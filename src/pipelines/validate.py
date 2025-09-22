from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import great_expectations as gx


REQUIRED_RAW_COLS: List[str] = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "vendorid",
    "ratecodeid",
    "payment_type",
    "passenger_count",
    "trip_distance",
]

REQUIRED_FEATURE_COLS: List[str] = [
    "vendor_id", "passenger_count", "trip_distance",
    "rate_code", "payment_type", "pickup_hour",
    "pickup_weekday", "duration_min",
]

def _fail_if_false(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(f"[DATA VALIDATION FAILED] {msg}")

def validate_raw_schema(df: pd.DataFrame) -> None:
    # compare using lowercase so raw casing doesn't matter
    cols_lower = {c.lower() for c in df.columns}
    missing = [c for c in (c.lower() for c in REQUIRED_RAW_COLS) if c not in cols_lower]
    _fail_if_false(len(missing) == 0, f"Missing required raw columns: {missing}")

def validate_features(df: pd.DataFrame) -> None:
    # GE validator from pandas (simple, no context files)
    v = gx.from_pandas(df)

    # required columns present
    present = all(c in df.columns for c in REQUIRED_FEATURE_COLS)
    _fail_if_false(present, f"Missing required feature columns: "
                  f"{[c for c in REQUIRED_FEATURE_COLS if c not in df.columns]}")

    # basic null checks
    for col in REQUIRED_FEATURE_COLS:
        res = v.expect_column_values_to_not_be_null(col)
        _fail_if_false(res.success, f"Nulls in {col}")

    # ranges
    ranges: dict[str, Tuple[float, float]] = {
        "trip_distance": (0, 100),
        "pickup_hour": (0, 23),
        "pickup_weekday": (0, 6),
        "passenger_count": (0, 8),
        "duration_min": (0.0, 180.0),
    }
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            res = v.expect_column_values_to_be_between(col, min_value=lo, max_value=hi)
            _fail_if_false(res.success, f"{col} out of range [{lo}, {hi}]")

    # minimal cardinality sanity checks
    # e.g., payment_type/rate_code should have at least a couple distinct values
    for discrete in ["payment_type", "rate_code", "vendor_id"]:
        if discrete in df.columns:
            if df[discrete].nunique(dropna=True) < 2:
                raise ValueError(f"[DATA VALIDATION FAILED] {discrete} has <2 distinct values")
