from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    code = proc.wait()
    if code != 0:
        raise SystemExit(code)


def ensure_dirs() -> None:
    for p in ["data/raw", "data/processed", "reports", "artifacts"]:
        Path(p).mkdir(parents=True, exist_ok=True)


def download_hint(raw_name: str, raw_path: str) -> None:
    print(f"[hint] Raw file not found: {raw_path}")
    print("[hint] Download it with PowerShell:")
    print(
        f'       Invoke-WebRequest -Uri "https://d37ci6vzurychx.cloudfront.net/trip-data/{raw_name}" '
        f'-OutFile "{raw_path}"'
    )


def run_month(year: int, month: int, ref_year: int | None, ref_month: int | None) -> None:
    ensure_dirs()

    raw_name = f"yellow_tripdata_{year:04d}-{month:02d}.parquet"
    raw_path = f"data/raw/{raw_name}"
    out_path = f"data/processed/train_{year:04d}_{month:02d}.parquet"
    model_path = "artifacts/model.joblib"

    if not Path(raw_path).exists():
        download_hint(raw_name, raw_path)
        raise FileNotFoundError(raw_path)

    # 1) ingest
    run([sys.executable, "-m", "src.pipelines.ingest", "--input", raw_path, "--output", out_path])

    # 2) train (overwrites artifacts/model.joblib by design)
    run([sys.executable, "-m", "src.models.train", "--train", out_path, "--model_artifact", model_path])

    # 3) drift
    if ref_year is not None and ref_month is not None:
        ref_path = f"data/processed/train_{ref_year:04d}_{ref_month:02d}.parquet"
        if not Path(ref_path).exists():
            print(f"[hint] Reference file not found: {ref_path}")
            print(f"[hint] Generate it first (run this pipeline for {ref_year}-{ref_month}).")
            raise FileNotFoundError(ref_path)
        drift_out = f"reports/drift_{ref_year:04d}{ref_month:02d}_vs_{year:04d}{month:02d}.html"
        run([
            sys.executable, "-m", "src.monitoring.simple_drift_report",
            "--ref", ref_path, "--cur", out_path, "--out", drift_out
        ])
    else:
        drift_out = f"reports/drift_{year:04d}{month:02d}_selfcheck.html"
        run([
            sys.executable, "-m", "src.monitoring.simple_drift_report",
            "--ref", out_path, "--cur", out_path, "--out", drift_out
        ])

    print("\n[done] month pipeline finished 🚀")
    print(f"processed: {out_path}")
    print(f"model:     {model_path}")
    print(f"drift:     {drift_out}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run monthly NYC taxi pipeline (ingest -> train -> drift).")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--ref-year", type=int)
    parser.add_argument("--ref-month", type=int)
    args = parser.parse_args()
    run_month(args.year, args.month, args.ref_year, args.ref_month)


if __name__ == "__main__":
    _cli()
