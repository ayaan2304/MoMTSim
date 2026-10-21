"""
config.py — Central configuration for the whole project.
USE: Every other script imports paths and shared constants from here, so the
folder layout and the harmonised transaction types are defined in ONE place.
If a path or a type name ever changes, it changes here only.
"""

# --- Import: pathlib for OS-safe file paths (USE: no hard-coded '/' strings) ---
from pathlib import Path

# --- Project root = two levels up from this file (src/ -> project/) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data folder (USE: raw downloaded CSVs live here; they are NOT pushed to GitHub) ---
DATA_DIR = PROJECT_ROOT / "data"

# --- Output folders (USE: all figures and result tables the scripts generate) ---
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
RES_DIR = PROJECT_ROOT / "outputs" / "results"

# --- Raw dataset files (USE: exact filenames produced by the download step) ---
# MoMTSim: the Mendeley dataset ships two CSVs; which one is V1 (1.72M rows) and
# which is V2 (4.23M rows) is confirmed by 01_inspect_data.py from the row counts.
MOMTSIM_FILES = [DATA_DIR / "MoMTSim_20240722202413_1000_dataset.csv", DATA_DIR / "synthetic_mobile_money_transaction_dataset.csv"]
PAYSIM_FILE = DATA_DIR / "paysim.csv"

# --- Processed files (USE: harmonised + feature-built tables, saved as parquet
#     so later scripts load in seconds instead of re-reading big CSVs) ---
MOMTSIM_PROCESSED = DATA_DIR / "momtsim_v1_features.parquet"
PAYSIM_PROCESSED = DATA_DIR / "paysim_features.parquet"

# --- The 5 harmonised transaction types (USE: both datasets are mapped onto
#     exactly these names, so a model trained on one dataset can read the other) ---
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# --- Random seed (USE: same seed everywhere so runs are reproducible) ---
SEED = 42
