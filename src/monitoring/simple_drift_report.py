from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index (lower is better; >0.2 moderate, >0.3 high)."""
    e = pd.Series(expected).dropna()
    a = pd.Series(actual).dropna()
    # Need at least a couple points
    if len(e) < 2 or len(a) < 2:
        return float("nan")

    # 1) Try quantile-based bins on reference with duplicates dropped
    edges: np.ndarray
    try:
        _, edges = pd.qcut(e, q=bins, duplicates="drop", retbins=True)
    except Exception:
        edges = np.linspace(e.min(), e.max(), num=min(bins, max(2, int(e.nunique()))))

    # 2) Ensure unique, valid edges
    edges = np.unique(edges)
    if len(edges) < 3:
        # Not enough distinct edges to form bins → PSI not meaningful
        return 0.0

    # Expand to cover all values robustly
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    # 3) Bin reference & actual using the same edges
    e_bins = pd.cut(e, bins=edges, include_lowest=True)
    a_bins = pd.cut(a, bins=edges, include_lowest=True)

    e_counts = e_bins.value_counts().sort_index()
    a_counts = a_bins.value_counts().sort_index()

    # 4) Convert to percentages with tiny smoothing to avoid div/zero
    eps = 1e-6
    k = len(e_counts)
    e_perc = (e_counts.to_numpy() + eps) / (e_counts.sum() + eps * k)
    a_perc = (a_counts.to_numpy() + eps) / (a_counts.sum() + eps * k)

    # 5) PSI
    return float(np.sum((e_perc - a_perc) * np.log(e_perc / a_perc)))


def main(ref_path: str, cur_path: str, out_html: str) -> None:
    ref = pd.read_parquet(ref_path)
    cur = pd.read_parquet(cur_path)

    target_col = "duration_min"
    # only numeric features that exist in both
    feature_cols: List[str] = [
        c
        for c in ref.columns.intersection(cur.columns).tolist()
        if c != target_col and pd.api.types.is_numeric_dtype(ref[c])
    ]

    rows = []
    for col in feature_cols:
        val = psi(ref[col], cur[col], bins=10)
        ref_mean, cur_mean = ref[col].mean(), cur[col].mean()
        ref_std, cur_std = ref[col].std(), cur[col].std()
        if pd.isna(val):
            level = "N/A"
        elif val >= 0.3:
            level = "HIGH"
        elif val >= 0.2:
            level = "MODERATE"
        elif val >= 0.1:
            level = "LOW"
        else:
            level = "NONE"
        rows.append(
            dict(
                feature=col,
                psi=0.0 if pd.isna(val) else round(float(val), 4),
                level=level,
                ref_mean=round(float(ref_mean), 4) if pd.notna(ref_mean) else None,
                cur_mean=round(float(cur_mean), 4) if pd.notna(cur_mean) else None,
                ref_std=round(float(ref_std), 4) if pd.notna(ref_std) else None,
                cur_std=round(float(cur_std), 4) if pd.notna(cur_std) else None,
            )
        )

    df = pd.DataFrame(rows).sort_values(
        ["level", "psi"], ascending=[False, False], key=lambda s: s.replace({"N/A": "0"})
    )

    style = """
    <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    h1 { margin-bottom: 0; }
    .meta { color: #666; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin-top: 16px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
    th { background: #f6f6f6; }
    td:first-child, th:first-child { text-align: left; }
    .HIGH { background:#ffe5e5; }
    .MODERATE { background:#fff3cd; }
    .LOW { background:#e7f3ff; }
    .NONE { background:#f7fff3; }
    .NA { background:#f0f0f0; }
    </style>
    """
    header = f"<h1>Simple Drift Report (PSI)</h1><div class='meta'>Ref: {ref_path}<br/>Cur: {cur_path}</div>"
    # Assign row classes for quick scanning
    classes = df["level"].astype(str).tolist()
    table = df.to_html(index=False, classes=classes)
    html = f"<!doctype html><html><head><meta charset='utf-8'>{style}</head><body>{header}{table}</body></html>"

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(html, encoding="utf-8")
    print(f"Drift report saved to {out_html}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--cur", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    main(args.ref, args.cur, args.out)
