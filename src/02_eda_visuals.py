"""
02_eda_visuals.py — Exploratory data analysis visuals for both datasets.
USE: Produces the "about the data" figures for the project/paper:
class balance, transactions by type, fraud rate by type, amount
distributions (legit vs fraud), activity over time, and a feature
correlation heatmap — separately for MoMTSim V1 and PaySim, with identical
plot code so the two datasets can be compared side by side.
Input : the processed parquet files from 01_inspect_data.py
Output: PNG files in outputs/figures/ named  eda_<dataset>_<chart>.png
Run:  python src/02_eda_visuals.py
"""

# --- Imports ---
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# matplotlib: all plotting (project requirement). Agg backend = no display
# needed, figures are written straight to files (USE: works on a server/VM).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# numpy/pandas: data handling and the log scales used for amount plots.
import numpy as np
import pandas as pd

# Project config (USE: processed-file paths and the figures folder).
from config import MOMTSIM_PROCESSED, PAYSIM_PROCESSED, FIG_DIR
from features import load_harmonised
from config import MOMTSIM_FILES, PAYSIM_FILE
import json
from config import RES_DIR


def plot_dataset(name, raw, feats):
    """
    USE: Draw all EDA charts for ONE dataset.
    `raw`   = harmonised transactions (types, amounts, steps, labels)
    `feats` = processed feature table (for the correlation heatmap)
    """
    # Short tag for filenames (USE: keeps figure names short and sortable).
    tag = name.lower().replace(" ", "_")

    # ---------- 1. Class balance (USE: visual proof of the imbalance problem —
    # PaySim's fraud slice is almost invisible, MoMTSim's is ~10%) ----------
    counts = raw["isFraud"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legitimate", "Fraud"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({100*v/counts.sum():.3f}%)", ha="center", va="bottom")
    ax.set_title(f"{name} — class balance")
    ax.set_ylabel("Transactions")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"eda_{tag}_class_balance.png", dpi=150); plt.close(fig)

    # ---------- 2. Transactions by type (USE: shows the mix of transaction
    # types each simulator produces) ----------
    tc = raw["type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tc.index, tc.values, color="#55A868")
    ax.set_title(f"{name} — transactions by type")
    ax.set_ylabel("Transactions")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"eda_{tag}_type_counts.png", dpi=150); plt.close(fig)

    # ---------- 3. Fraud rate by type (USE: the key structural difference —
    # PaySim fraud exists only in TRANSFER/CASH_OUT, MoMTSim fraud is spread
    # across types; this chart is quoted in the paper's dataset discussion) --
    fr = raw.groupby("type")["isFraud"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(fr.index, fr.values, color="#C44E52")
    ax.set_title(f"{name} — fraud rate by type (%)")
    ax.set_ylabel("Fraud %")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"eda_{tag}_fraud_rate_by_type.png", dpi=150); plt.close(fig)

    # ---------- 4. Amount distribution, legit vs fraud (USE: fraud amounts
    # behave differently; log scale because amounts span a huge range) ------
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        sample = raw.loc[raw["isFraud"] == label, "amount"]
        # Sample for plotting speed (USE: histogram of 6M points is slow and
        # visually identical to a 200k sample).
        sample = sample.sample(min(len(sample), 200_000), random_state=42)
        ax.hist(np.log1p(sample), bins=50, alpha=0.55,
                label="Fraud" if label else "Legitimate", color=color)
    ax.set_title(f"{name} — log(amount) distribution")
    ax.set_xlabel("log(1 + amount)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / f"eda_{tag}_amount_dist.png", dpi=150); plt.close(fig)

    # ---------- 5. Activity over time (USE: transactions and fraud per step —
    # shows the temporal structure that motivates the temporal split) -------
    per_step = raw.groupby("step").agg(total=("isFraud", "size"), fraud=("isFraud", "sum"))
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(per_step.index, per_step["total"], lw=0.7)
    axes[0].set_title(f"{name} — transactions per step (1 step = 1 hour)")
    axes[1].plot(per_step.index, per_step["fraud"], lw=0.7, color="#C44E52")
    axes[1].set_title(f"{name} — fraud per step")
    axes[1].set_xlabel("step")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"eda_{tag}_activity_over_time.png", dpi=150); plt.close(fig)

    # ---------- 6. Correlation heatmap of numeric features (USE: shows which
    # engineered features move together / with fraud; sample 100k rows so the
    # correlation computation stays fast on the big PaySim table) ----------
    num = feats.drop(columns=["isFraud", "step_raw"]).sample(
        min(len(feats), 100_000), random_state=42)
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{name} — feature correlation heatmap")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"eda_{tag}_correlation.png", dpi=150); plt.close(fig)

    print(f"{name}: 6 EDA figures saved (tag={tag})")


def main():
    # Ensure the figures folder exists (USE: fresh-clone safety).
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Identify the MoMTSim V1 file from the inspection results (USE: same
    # decision 01_inspect_data.py made — the smaller of the two Mendeley CSVs).
    insp = json.loads((RES_DIR / "inspection.json").read_text())
    v1_file = insp["momtsim_v1_file"]

    # Load raw (for type/amount/step charts) + processed (for correlations).
    mom_raw = load_harmonised(MOMTSIM_FILES[0].parent / v1_file, "momtsim")
    pay_raw = load_harmonised(PAYSIM_FILE, "paysim")
    mom_feats = pd.read_parquet(MOMTSIM_PROCESSED)
    pay_feats = pd.read_parquet(PAYSIM_PROCESSED)

    # Draw both datasets with the SAME function (USE: identical charts =
    # honest side-by-side comparison in the paper).
    plot_dataset("MoMTSim V1", mom_raw, mom_feats)
    plot_dataset("PaySim", pay_raw, pay_feats)


# Standard entry point.
if __name__ == "__main__":
    main()
