# Results — MoMTSim + PaySim Fraud Detection (run completed 2026-10-21)

All numbers below are from `outputs/results/metrics.csv` (temporal test
splits; thresholds tuned on validation only). Figures: `outputs/figures/eval_*`.

## Headline table

| Exp | Train → Test | PR-AUC | ROC-AUC | Precision | Recall | F1 | MCC | FNR | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| E1 | MoMTSim → MoMTSim | 0.3363 | 0.8916 | 0.310 | 0.9988 | 0.473 | 0.490 | 0.0012 | 0.271 |
| E2 | PaySim → PaySim | 0.9968 | 0.9984 | 1.000 | 0.9968 | 0.998 | 0.998 | 0.0032 | 1.000 |
| E3 | MoMTSim → PaySim | 0.0190 | 0.5350 | 0.000 | 0.0000 | 0.000 | −0.005 | 1.0000 | 0.271 |
| E4 | PaySim → MoMTSim | 0.0909 | 0.5000 | 0.000 | 0.0000 | 0.000 | 0.000 | 1.0000 | 1.000 |
| E6a | MoMTSim (no artifact feature) → MoMTSim | 0.3380 | 0.8915 | 0.310 | 0.9995 | 0.473 | 0.491 | 0.0005 | 0.285 |
| E6b | MoMTSim (no artifact feature) → PaySim | 0.0301 | 0.5281 | 0.289 | 0.0535 | 0.090 | 0.119 | 0.9465 | 0.285 |

Test sizes: MoMTSim test = 294,035 rows (26,715 fraud); PaySim test = 89,466
rows (1,252 fraud). Splits are by step *range* (70/15/15 of the step axis),
so row counts per split are uneven — transaction volume is not uniform
across steps in either simulator.

## What the results say

1. **In-domain, both models work — very differently.** PaySim in-domain is
   near-perfect (E2, PR-AUC 0.997), reproducing the reference paper's story
   under a *fair temporal split*. MoMTSim in-domain (E1) catches 99.88% of
   fraud (FNR 0.0012) but at 31% precision — many false alarms (FPR 22%).
   MoMTSim fraud is genuinely harder to separate than PaySim fraud.
2. **Cross-dataset transfer fails in BOTH directions.** E3 ROC-AUC 0.535 and
   E4 ROC-AUC 0.500 are random-level ranking: a model trained on one
   simulator has essentially no fraud-ranking skill on the other. This is
   the project's central finding — single-dataset results (like E2's 0.997)
   do **not** imply a model has learned general fraud behaviour.
   Likely drivers: completely different money scales (MoMTSim amounts are
   tens–hundreds; PaySim's are thousands–lakhs, and tree splits are on
   absolute values), different fraud-generation rules, and the 10% → 0.13%
   prevalence shift (visible in the calibration figure).
3. **The artifact feature is NOT what carries the MoMTSim model.** Removing
   `amount_equals_balance` (E6a) leaves MoMTSim performance unchanged
   (0.3380 vs 0.3363) and slightly *improves* cross-dataset PR-AUC
   (E6b 0.0301 vs E3 0.0190) — the model is not living on the simulator
   shortcut, yet transfer still fails on ranking (ROC ~0.53).
4. **SHAP: the top explanation is stable, the rest is not.**
   (`outputs/results/shap_importance.csv`) For the MoMTSim model,
   `type_TRANSFER` is the #1 feature on *both* test sets, and `diffOrig` /
   `errorDest` appear in both top-8 lists — but the middle ranks shuffle
   (`newbalanceOrig`, `step`, `time_since_last_dest` rise on PaySim).
   Partial explanation stability did not rescue cross-dataset performance.

## Honest caveats
- Both datasets are synthetic; "transfer" here means simulator-to-simulator.
- E2/E4's validation-tuned threshold came out at 1.000 (the isotonic-calibrated
  PaySim ensemble separates its validation split almost perfectly), which is
  why E4 shows exactly 0 predictions above threshold on MoMTSim — judge E4
  by its threshold-free ROC-AUC (0.500), not by the zero recall alone.
- Random Forest was trained on a stratified 800k-row cap (documented in
  code); LightGBM/XGBoost saw all training rows.
- A first full run of this pipeline was OOM-killed during E6b on a 7 GB
  machine; the committed code contains the memory fixes (see git history).

## Suggested next steps (not yet run)
- Scale-invariant feature set (ratios / logs / within-dataset ranks only)
  for a dedicated cross-dataset model — the natural follow-up to finding 2.
- Threshold re-calibration on a small labelled slice of the target dataset
  (how much target data is needed to recover performance?).
- Cost-sensitive evaluation: missed-fraud cost vs false-alarm review cost.
