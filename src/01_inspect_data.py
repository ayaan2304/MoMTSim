"""
01_inspect_data.py — Step 0 of the project: inspect the raw datasets.
USE: Before any modelling we must KNOW what we actually downloaded — real
column names, real transaction-type values, real fraud counts, real step
ranges. This script prints those facts and saves them to
outputs/results/inspection.json. It also decides which Mendeley file is
MoMTSim V1 (1.72M rows) and which is V2 (4.23M rows), purely from row counts,
and saves the harmonised + feature-built parquet files used by later scripts.
Run:  python src/01_inspect_data.py
"""

# --- Imports ---
# json: save inspection facts as a machine-readable result file.
import json
# sys/pathlib: make the sibling modules (config, features) importable when
# this script is run directly from the project root.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pandas: reading and summarising the CSVs.
import pandas as pd

# Project modules (USE: paths/constants from config; harmonisation and
# feature building from features — the same functions every script uses).
from config import MOMTSIM_FILES, PAYSIM_FILE, MOMTSIM_PROCESSED, PAYSIM_PROCESSED, RES_DIR
from features import load_harmonised, build_features


def inspect(df, name):
    """
    USE: Compute the factual summary of one harmonised dataset: size, fraud
    count/%, step range, per-type counts and per-type fraud rates, nulls.
    Returns a dict that is printed and stored in inspection.json.
    """
    # Basic size and label facts (USE: fraud % is the single most important
    # number in this project — it drives the whole imbalance discussion).
    fraud_n = int(df["isFraud"].sum())
    info = {
        "dataset": name,
        "rows": int(len(df)),
        "fraud_rows": fraud_n,
        "fraud_pct": round(100 * fraud_n / len(df), 4),
        "step_min": int(df["step"].min()),
        "step_max": int(df["step"].max()),
        "nulls_total": int(df.isna().sum().sum()),
        # Per-type transaction counts (USE: confirms the type harmonisation —
        # after mapping, only the 5 common type names may appear).
        "type_counts": {k: int(v) for k, v in df["type"].value_counts().items()},
        # Per-type fraud rate % (USE: shows WHERE fraud lives in each dataset,
        # e.g. PaySim fraud only in TRANSFER/CASH_OUT vs MoMTSim everywhere).
        "type_fraud_rate_pct": {
            k: round(100 * float(v), 4)
            for k, v in df.groupby("type")["isFraud"].mean().items()
        },
    }
    # Print a readable version for the console log.
    print(f"\n===== {name} =====")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


def main():
    # Make sure the results folder exists (USE: first run on a fresh clone).
    RES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Inspect BOTH Mendeley files (USE: the download gave us two CSVs with
    #     opaque names; row count is the only reliable way to tell V1 from V2).
    mom_frames = {}
    for f in MOMTSIM_FILES:
        df = load_harmonised(f, "momtsim")
        results[f.name] = inspect(df, f"MoMTSim file {f.name}")
        mom_frames[f.name] = df

    # The smaller MoMTSim file is V1 (USE: our plan's primary dataset is V1 —
    # 1.72M rows, ~10% fraud; V2's ~53% fraud is kept aside, not used).
    v1_name = min(mom_frames, key=lambda k: len(mom_frames[k]))
    mom_v1 = mom_frames[v1_name]
    results["momtsim_v1_file"] = v1_name
    print(f"\nMoMTSim V1 identified as: {v1_name} ({len(mom_v1)} rows)")

    # --- Inspect PaySim (USE: its numbers must match the published stats —
    #     6,362,620 rows and 8,213 fraud — which is how we verify this mirror
    #     copy is the genuine, complete PaySim dataset).
    paysim = load_harmonised(PAYSIM_FILE, "paysim")
    results["paysim.csv"] = inspect(paysim, "PaySim")

    # --- Save inspection facts (USE: the paper/write-up quotes numbers from
    #     this file, not from console scrollback).
    (RES_DIR / "inspection.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {RES_DIR / 'inspection.json'}")

    # --- Build features and save parquet (USE: feature building on 6M+ rows
    #     takes minutes; doing it ONCE here means the EDA and training scripts
    #     load ready-made parquet files in seconds).
    for df, out in [(mom_v1, MOMTSIM_PROCESSED), (paysim, PAYSIM_PROCESSED)]:
        X, y, meta = build_features(df)
        # Reassemble one table: features + target + step (USE: step is needed
        # by the training script for the temporal split).
        table = X.copy()
        table["isFraud"] = y.values
        table["step_raw"] = meta.values
        table.to_parquet(out)
        print(f"Saved: {out}  shape={table.shape}  fraud%={100*y.mean():.4f}")


# Standard entry point (USE: runs main() only when executed as a script).
if __name__ == "__main__":
    main()
