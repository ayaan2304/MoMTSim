# Explainable Ensemble Learning for Mobile Money Fraud Detection
### MoMTSim (primary) + PaySim (external validation) — one pipeline, two datasets

A research project that extends the reference paper *"Explainable AI-enhanced
ensemble learning for financial fraud detection in mobile money transactions"*
(Vijayanand & Smrithy, *Intelligent Decision Technologies*, 2025) in three ways:

1. **A second, newer dataset** — MoMTSim V1 (2025) instead of PaySim alone.
   PaySim has only 8,213 fraud cases (0.129%), which biases models toward the
   legitimate class and causes missed fraud. MoMTSim V1 has 175,518 fraud
   cases (10.20%), so the model sees enough fraud to actually learn it.
2. **Honest validation** — temporal splits (train on the past, test on the
   future) instead of random splits, and PR-AUC as the headline metric
   instead of accuracy (with 0.129% fraud, a model that predicts "legitimate"
   for everything already scores 99.87% accuracy — accuracy means nothing here).
3. **Cross-dataset generalisation + explainability** — train on MoMTSim, test
   on PaySim (and reverse), plus SHAP-stability analysis: does the model's
   *reasoning* survive a change of dataset?

---

## Table of contents
- [Verified dataset facts](#verified-dataset-facts-checked-2026-10-21-from-the-actual-files)
- [Method](#method-one-pipeline-both-datasets)
- [Experiments](#experiments)
- [Setup instructions (step by step)](#setup-instructions-step-by-step)
- [What each output file is](#what-each-output-file-is)
- [Troubleshooting](#troubleshooting)
- [Datasets & licences](#datasets--licences)

---

## Verified dataset facts (checked 2026-10-21 from the actual files)

| | MoMTSim V1 | MoMTSim V2 (unused) | PaySim |
|---|---|---|---|
| File | `synthetic_mobile_money_transaction_dataset.csv` | `MoMTSim_20240722202413_1000_dataset.csv` | `paysim.csv` |
| Rows | 1,720,181 | 4,225,958 | 6,362,620 |
| Fraud | 175,518 (10.2035%) | 2,233,118 (52.8429%) | 8,213 (0.1291%) |
| Steps (1 step = 1 hour) | 0–143 | 0–192 | 1–743 |
| Fraud by type | TRANSFER only (30.83% of transfers are fraud) | TRANSFER only (88.94%) | TRANSFER (0.769% of transfers) + CASH_OUT (0.184%) |
| Source | Mendeley Data, DOI 10.17632/zhj366m53p.2 (CC BY 4.0) | same | Kaggle `ealaxi/paysim1` |

Both Mendeley files were verified byte-for-byte against the official SHA-256
hashes published on Mendeley Data. PaySim's row/fraud counts match the
published statistics exactly. Full details: `outputs/results/inspection.json`.

> **Why V2 is not used:** a 52.84% fraud rate is not a realistic
> fraud-detection benchmark — real-world fraud is a rare event.

## Method (one pipeline, both datasets)

The same code processes both datasets — that identity is what makes the
cross-dataset experiments fair.

- **Harmonisation** (`src/features.py` → `load_harmonised`): MoMTSim's
  columns/types are mapped to PaySim's naming
  (`transactionType→type`, `initiator→nameOrig`, `oldBalInitiator→oldbalanceOrg`,
  `WITHDRAWAL→CASH_OUT`, `DEPOSIT→CASH_IN`, …). Result for both datasets:
  `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest,
  oldbalanceDest, newbalanceDest, isFraud`.
  Account IDs are **never** model features (they don't exist across datasets),
  and PaySim's `isFlaggedFraud` rule column is dropped on purpose.
- **Features** (`build_features`, identical for both datasets):
  - *Balance-error features* — `errorOrig = newbalanceOrig + amount − oldbalanceOrg`,
    `errorDest`, `diffOrig`, `diffDest`. Honest transactions reconcile to 0;
    fraud breaks the arithmetic.
  - *Ratios/flags* — amount ÷ sender balance, amount ÷ receiver balance,
    `log_amount`, account-emptied flag, `amount == oldbalanceOrg` flag
    (this last one is the known PaySim simulator artifact — see experiment E6).
  - *Time* — hour of day, day, night flag (from `step`).
  - *Behavioural, past-only* — per-account past transaction count, past mean
    amount, amount vs. the account's own habit, time since last transaction,
    distinct past receivers, receiver past count/mean, new-receiver flag.
    Computed strictly from **earlier** transactions (no future leakage), per
    dataset, with the same code.
- **Split:** temporal 70% / 15% / 15% by `step` → train / validation / test.
  Never random.
- **Imbalance handling:** class weights — XGBoost `scale_pos_weight`,
  LightGBM `scale_pos_weight`, Random Forest `class_weight="balanced_subsample"`.
  No SMOTE (too slow and unrealistic at this scale, and it would need
  different settings per dataset).
- **Models:** LightGBM + XGBoost + Random Forest **soft-voting ensemble**
  (average of the three fraud probabilities), then **isotonic calibration**
  on the validation split, and the decision **threshold is tuned on
  validation only** (max F1). Random Forest trains on a stratified sample
  capped at 800,000 rows when the training set is larger (same cap rule for
  both datasets — see `RF_TRAIN_CAP` in `src/03_train_evaluate.py`).
- **Metrics:** PR-AUC / average precision (**primary**), ROC-AUC, Precision,
  Recall, False Negative Rate, F1, MCC, Brier score, full confusion counts.

## Experiments

| ID | Train | Test | Question it answers |
|---|---|---|---|
| E1 | MoMTSim V1 | MoMTSim V1 | Main in-domain result |
| E2 | PaySim | PaySim | Reference-paper baseline (fair temporal split) |
| E3 | MoMTSim V1 | PaySim | Cross-dataset: does MoMTSim training transfer? |
| E4 | PaySim | MoMTSim V1 | Reverse transfer (expected weaker — PaySim training saw few, single-pattern frauds) |
| E6a | MoMTSim V1 without `amount_equals_balance` | MoMTSim V1 | Ablation: is the model living on the simulator artifact? |
| E6b | MoMTSim V1 without `amount_equals_balance` | PaySim | Same ablation, cross-dataset |

Results land in `outputs/results/metrics.csv` (+ `.json`), SHAP rankings in
`outputs/results/shap_importance.csv`, figures in `outputs/figures/`.

---

## Setup instructions (step by step)

### 0. Requirements
- Python **3.10+** (developed on 3.12), Git
- ~8 GB free RAM recommended (PaySim feature building peaks on 6.36M rows),
  ~3 GB free disk for the raw datasets
- OS: Linux/macOS/Windows (commands below are Linux/macOS; on Windows use
  `.venv\Scripts\activate` in Step 2)

### 1. Clone this repository
The dataset CSVs (~970 MB) are stored in this repo with **Git LFS**, so
install Git LFS first (https://git-lfs.com), then:
```bash
git lfs install
git clone https://github.com/ayaan2304/MoMTSim.git
cd MoMTSim
```
If you cloned without Git LFS you will see small text "pointer" files instead
of real CSVs in `data/` — fix that with `git lfs pull` inside the repo.
No Git LFS? Skip the clone's data and use the download script in Step 3.

### 2. Create a virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost,
lightgbm, shap, pyarrow.

### 3. Get the datasets into `data/`
The raw CSVs are included in this repo via Git LFS (Step 1) — if `data/`
already contains the three real CSVs (check sizes: ~157 MB, ~366 MB,
~494 MB), skip to Step 4. Otherwise download them with the project's own
script, which verifies MoMTSim against Mendeley's official SHA-256 hashes:
```bash
python src/00_download_data.py
```
Or fetch them manually from the original sources:

**MoMTSim (both files):**
1. Open https://data.mendeley.com/datasets/zhj366m53p/2
   (DOI: 10.17632/zhj366m53p.2)
2. Download both CSVs, keeping their original names:
   - `synthetic_mobile_money_transaction_dataset.csv` ← this is **V1** (primary)
   - `MoMTSim_20240722202413_1000_dataset.csv` ← this is V2 (downloaded for
     inspection; the pipeline identifies V1 by row count automatically)
3. Put both files in `data/`.

**PaySim:**
1. Open https://www.kaggle.com/datasets/ealaxi/paysim1 (Kaggle account needed)
2. Download and unzip. The CSV inside is named
   `PS_20174392719_1491204439457_log.csv` — **rename it to `paysim.csv`**
3. Put `paysim.csv` in `data/`.

Your `data/` folder should now contain exactly:
```
data/
├── synthetic_mobile_money_transaction_dataset.csv
├── MoMTSim_20240722202413_1000_dataset.csv
└── paysim.csv
```

### 4. Step 1 — Inspect the datasets and build feature tables
```bash
python src/01_inspect_data.py
```
What it does: loads and harmonises all three files, prints rows / fraud % /
step range / per-type counts and fraud rates, identifies the MoMTSim V1 file
by row count, saves `outputs/results/inspection.json`, then builds the
engineered features once and saves them as fast-loading parquet files
(`data/momtsim_v1_features.parquet`, `data/paysim_features.parquet`).
**Check your output matches the verified facts table above** — if your
numbers differ, your downloaded files are wrong/incomplete.
⏱ ~2 minutes.

### 5. Step 2 — Generate the EDA (data) visuals
```bash
python src/02_eda_visuals.py
```
What it does: creates 12 charts in `outputs/figures/` — for each dataset:
class balance, transactions by type, fraud rate by type, log-amount
distribution (legit vs fraud), activity over time, feature correlation
heatmap. ⏱ ~1 minute.

### 6. Step 3 — Train the models and evaluate everything
```bash
python src/03_train_evaluate.py
```
What it does, in order: trains the ensemble on MoMTSim (E1), tests that same
untouched model on PaySim (E3), trains on PaySim (E2), tests that model on
MoMTSim (E4), runs the E6 artifact ablation, writes
`outputs/results/metrics.csv`, draws all evaluation figures, and computes
SHAP importances for the MoMTSim model on **both** test sets
(`outputs/results/shap_importance.csv` + SHAP figures).
⏱ This is the long step: roughly 30–90 minutes depending on CPU
(developed on a 2-core VM). Progress lines print as each experiment finishes,
e.g. `E1_MoMTSim_to_MoMTSim: PR-AUC=... Recall=...`.

### 7. Read the results
- `outputs/results/metrics.csv` — one row per experiment: PR-AUC, ROC-AUC,
  precision, recall, F1, MCC, FNR, FPR, Brier, TP/FP/TN/FN, threshold used.
- `outputs/figures/eval_metric_comparison.png` — the one-glance summary chart.
- `outputs/results/shap_importance.csv` — compare the MoMTSim-test and
  PaySim-test columns: same top features = the model's explanations
  generalise across datasets.

### Re-running only part of the project
- Figures only: re-run Step 2 (EDA) — training figures are regenerated by Step 3.
- After changing features in `src/features.py`: delete the two parquet files
  in `data/` and re-run from Step 1 (the parquet is a cache of the features).

---

## What each output file is

**`outputs/results/`**
| File | Contents |
|---|---|
| `inspection.json` | Verified facts about every raw file (rows, fraud, types, steps) |
| `metrics.csv` / `metrics.json` | All experiments' scores (the paper's results table) |
| `shap_importance.csv` | Mean \|SHAP\| per feature, MoMTSim model on both test sets |

**`outputs/figures/`** (all generated with matplotlib)
| File pattern | Chart |
|---|---|
| `eda_<dataset>_class_balance.png` | Legitimate vs fraud counts |
| `eda_<dataset>_type_counts.png` | Transactions per type |
| `eda_<dataset>_fraud_rate_by_type.png` | Fraud % per type |
| `eda_<dataset>_amount_dist.png` | log(amount): legit vs fraud |
| `eda_<dataset>_activity_over_time.png` | Transactions + fraud per step |
| `eda_<dataset>_correlation.png` | Feature correlation heatmap |
| `eval_roc_pr_curves.png` | ROC + Precision–Recall curves, E1–E4 |
| `eval_confusion_matrices.png` | TP/FP/TN/FN per experiment |
| `eval_calibration.png` | Predicted vs observed fraud probability |
| `eval_metric_comparison.png` | PR-AUC / ROC-AUC / Recall / MCC bars, E1–E4 |
| `eval_importance_momtsim.png` / `eval_importance_paysim.png` | LightGBM feature importance |
| `eval_shap_momtsim_test.png` / `eval_shap_paysim_test.png` | SHAP summary bars |

## Project structure
```
├── README.md
├── requirements.txt
├── .gitignore             # excludes parquet caches, venv, logs
├── .gitattributes         # dataset CSVs tracked with Git LFS
├── data/                  # raw dataset CSVs (Git LFS) — see Step 3
├── src/
│   ├── config.py          # paths, constants, the 5 harmonised type names
│   ├── features.py        # THE shared harmonisation + feature pipeline
│   ├── 01_inspect_data.py # Step 1: facts + feature parquet building
│   ├── 02_eda_visuals.py  # Step 2: data charts
│   └── 03_train_evaluate.py # Step 3: training, E1–E4 + E6, eval figures, SHAP
└── outputs/
    ├── figures/           # generated charts (committed)
    └── results/           # generated tables (committed)
```
Every code section carries a comment explaining what it is used for.

## Troubleshooting
- **`FileNotFoundError` in Step 1** — the CSVs are not in `data/` or PaySim
  was not renamed to `paysim.csv`. See Step 3.
- **Numbers in Step 1 don't match the verified table** — wrong or partial
  download. Re-download; check file sizes (V1 ≈ 157 MB, V2 ≈ 366 MB,
  PaySim ≈ 494 MB unzipped).
- **Out of memory during Step 1/3** — close other apps; PaySim processing
  needs ~6–8 GB free RAM. Do not raise `RF_TRAIN_CAP`.
- **LightGBM `eval_set` deprecation warning** — harmless, printed by newer
  LightGBM versions; training still uses the validation set for early stopping.
- **Very slow Random Forest** — expected on low-core machines; it is capped
  at 800k training rows by design (`RF_TRAIN_CAP`).

## Datasets & licences
- **MoMTSim** — Azamuke, Katarahweire & Bainomugisha, *"A labeled synthetic
  mobile money transaction dataset"*, Data in Brief 60 (2025), Article 111534.
  Data: Mendeley Data, DOI 10.17632/zhj366m53p.2, **CC BY 4.0**.
- **PaySim** — Lopez-Rojas, Elmir & Axelsson, *"PaySim: A financial mobile
  money simulator for fraud detection"* (2016). Synthetic data generated from
  real mobile-money logs of a multinational provider. Available on Kaggle.

Both datasets are **synthetic** — no real customer transactions are used.

## Results

Full results, interpretation and caveats: **[RESULTS.md](RESULTS.md)**
(metrics in `outputs/results/metrics.csv`, figures in `outputs/figures/`).
