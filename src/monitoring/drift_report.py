from __future__ import annotations
import argparse
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

def main(ref_path: str, cur_path: str, out_html: str):
    ref = pd.read_parquet(ref_path)
    cur = pd.read_parquet(cur_path)
    features = [c for c in cur.columns if c != 'duration_min']
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=ref[features + ['duration_min']], current_data=cur[features + ['duration_min']])
    report.save_html(out_html)
    print(f"Drift report saved to {out_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--cur", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.ref, args.cur, args.out)
