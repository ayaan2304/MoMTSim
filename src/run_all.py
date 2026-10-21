"""
run_all.py — One-click runner for the complete project pipeline.
USE: Runs every stage in the correct order —
  00 download datasets -> 01 inspect + build features -> 02 EDA visuals
  -> 03 train + evaluate + SHAP
so the whole project reproduces with a single command:
    python src/run_all.py
Each stage is a separate script (you can still run them individually —
see README); this runner simply calls them one after another and stops
immediately if any stage fails, printing which stage broke.
"""

# --- Imports ---
# subprocess: run each stage as its own Python process (USE: a stage's
# memory is fully released before the next stage starts — important on
# machines with ~8 GB RAM, since PaySim processing peaks high).
import subprocess
# sys/pathlib: locate the stage scripts and use the SAME Python interpreter
# (the active venv) for every stage.
import sys
from pathlib import Path

# The src/ folder holding all stage scripts (this file's own folder).
SRC = Path(__file__).resolve().parent

# Stage order (USE: each stage consumes the previous stage's outputs —
# download -> parquet features -> figures -> training results).
STAGES = [
    "00_download_data.py",    # datasets into data/ (skips files already present)
    "01_inspect_data.py",     # verified facts + feature parquet files
    "02_eda_visuals.py",      # data charts into outputs/figures/
    "03_train_evaluate.py",   # training, experiments E1-E4 + E6, eval figures, SHAP
]


def main():
    for stage in STAGES:
        print(f"\n{'=' * 70}\nRUNNING: {stage}\n{'=' * 70}", flush=True)
        # check=True (USE: a failing stage raises and stops the pipeline —
        # later stages would only fail confusingly on missing inputs).
        subprocess.run([sys.executable, str(SRC / stage)], check=True)
    print("\nPIPELINE COMPLETE — see outputs/figures/ and outputs/results/")


# Standard entry point.
if __name__ == "__main__":
    main()
