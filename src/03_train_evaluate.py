"""
03_train_evaluate.py — Training + evaluation for all experiments.
USE: Trains the soft-voting ensemble (LightGBM + XGBoost + Random Forest)
with the ONE shared pipeline, then runs:
  E1  train MoMTSim -> test MoMTSim   (main in-domain result)
  E2  train PaySim  -> test PaySim    (reference-paper baseline, fairly split)
  E3  train MoMTSim -> test PaySim    (cross-dataset generalisation)
  E4  train PaySim  -> test MoMTSim   (reverse cross-dataset)
  E6  ablation: MoMTSim model WITHOUT the 'amount_equals_balance' artifact
      feature, tested on both datasets (is the model living on the shortcut?)
Outputs:
  outputs/results/metrics.csv / metrics.json  — every experiment's scores
  outputs/results/shap_importance.csv        — SHAP rankings on both datasets
  outputs/figures/eval_*.png                  — PR/ROC curves, confusion
      matrices, calibration curves, metric comparison, feature importance,
      SHAP bars
Run:  python src/03_train_evaluate.py
"""

# --- Imports ---
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# matplotlib (Agg): all evaluation figures are written to files, no display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# numpy/pandas: data handling.
import numpy as np
import pandas as pd

# scikit-learn: metrics, Random Forest, calibration (isotonic), sampling cap.
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             brier_score_loss)

# LightGBM / XGBoost: the two gradient-boosted tree models of the ensemble.
import lightgbm as lgb
import xgboost as xgb

# SHAP: explainability — TreeExplainer gives per-feature contributions whose
# stability across datasets is one of this project's results.
import shap

# Project config + the frozen feature list (USE: same features everywhere).
from config import MOMTSIM_PROCESSED, PAYSIM_PROCESSED, FIG_DIR, RES_DIR, SEED
from features import FEATURES

# RF training cap (USE: Random Forest on 4.5M rows is too slow/heavy for a
# student machine; when the train set is bigger than this, RF trains on a
# stratified sample of this size — the SAME cap rule for both datasets, so
# the one-pipeline promise still holds; LightGBM/XGBoost always see all rows).
RF_TRAIN_CAP = 800_000


def temporal_split(table):
    """
    USE: Split a processed table into train/val/test by TIME (step), never
    randomly — first 70% of steps train, next 15% validation, last 15% test.
    Random splits leak future behaviour into training and inflate scores.
    """
    # Cut points from the step range (USE: proportional on steps, so both
    # datasets use the same rule even though their step counts differ).
    smin, smax = int(table["step_raw"].min()), int(table["step_raw"].max())
    cut1 = smin + 0.70 * (smax - smin)
    cut2 = smin + 0.85 * (smax - smin)
    # Boolean masks per partition.
    tr = table[table["step_raw"] <= cut1]
    va = table[(table["step_raw"] > cut1) & (table["step_raw"] <= cut2)]
    te = table[table["step_raw"] > cut2]
    return tr, va, te


def train_ensemble(Xtr, ytr, Xva, yva, feature_cols):
    """
    USE: Train LightGBM + XGBoost + Random Forest on the given training data
    and return the three fitted models. Imbalance is handled with class
    weights / scale_pos_weight (neg/pos ratio) — the same rule for a 10%
    fraud dataset and a 0.13% fraud dataset; no SMOTE anywhere.
    """
    # Imbalance ratio (USE: tells each model how much more a fraud example
    # counts than a legitimate one).
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    # --- LightGBM (USE: fastest strong model on millions of rows; early
    #     stopping on validation PR-AUC so it doesn't overfit) ---
    lgbm = lgb.LGBMClassifier(
        objective="binary", n_estimators=1000, learning_rate=0.05,
        num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        scale_pos_weight=spw, n_jobs=2, random_state=SEED, verbose=-1)
    lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
             callbacks=[lgb.early_stopping(50, verbose=False)])

    # --- XGBoost (USE: second boosted-tree model; different boosting
    #     implementation = useful diversity for the voting ensemble) ---
    xgbm = xgb.XGBClassifier(
        tree_method="hist", max_depth=7, learning_rate=0.05, n_estimators=1000,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="aucpr", early_stopping_rounds=50, n_jobs=2,
        random_state=SEED)
    xgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    # --- Random Forest (USE: bagged trees = a third, decorrelated opinion
    #     for the ensemble; trained on the capped stratified sample if the
    #     training set exceeds RF_TRAIN_CAP — see the constant's comment) ---
    if len(Xtr) > RF_TRAIN_CAP:
        idx = (pd.DataFrame({"y": ytr}).groupby("y", group_keys=False)
               .apply(lambda g: g.sample(
                   n=int(RF_TRAIN_CAP * len(g) / len(Xtr)), random_state=SEED))
               .index)
        Xrf, yrf = Xtr.loc[idx], ytr.loc[idx]
    else:
        Xrf, yrf = Xtr, ytr
    rf = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=2, random_state=SEED)
    rf.fit(Xrf, yrf)
    return {"lgbm": lgbm, "xgb": xgbm, "rf": rf}


def ensemble_prob(models, X):
    """
    USE: Soft-voting prediction — the average of the three models' fraud
    probabilities. Averaging probabilities (not hard votes) keeps the full
    confidence information for PR-AUC, calibration and threshold tuning.
    """
    return (models["lgbm"].predict_proba(X)[:, 1]
            + models["xgb"].predict_proba(X)[:, 1]
            + models["rf"].predict_proba(X)[:, 1]) / 3.0


def best_f1_threshold(y_true, prob):
    """
    USE: Pick the decision threshold on VALIDATION data only, by maximising
    F1 along the precision-recall curve. The test set never chooses its own
    threshold — that would be a subtle form of cheating.
    """
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[max(np.argmax(f1[:-1]), 0)]) if len(thr) else 0.5


def evaluate(y_true, prob, threshold):
    """
    USE: Compute the full evaluation bundle for one experiment at one
    threshold. Accuracy is deliberately NOT the headline: PR-AUC (average
    precision) is primary because it stays honest under extreme imbalance.
    """
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "fnr": float(fn / max(fn + tp, 1)),          # fraud we MISSED — the costly error
        "fpr": float(fp / max(fp + tn, 1)),
        "brier": float(brier_score_loss(y_true, prob)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(threshold),
    }


def run_experiment(name, train_table, test_table, feature_cols,
                     retrain_from=None, curves=None, cms=None):
    """
    USE: One full experiment: train (or reuse) an ensemble on train_table,
    calibrate + tune the threshold on its validation split, then evaluate on
    test_table's TEMPORAL TEST split. For cross-dataset runs the test table
    is a different dataset — but its features came from the same
    build_features() code, so columns align exactly.
    Returns (metrics_dict, trained_bundle) where trained_bundle can be
    reused (E6 reuses nothing; E3/E4 are produced by passing the E1/E2
    bundles via retrain_from to avoid training twice).
    """
    t0 = time.time()
    # Temporal split of the TRAINING table (USE: gives train/val for fitting,
    # early stopping, calibration and threshold tuning).
    tr, va, _ = temporal_split(train_table)
    Xtr, ytr = tr[feature_cols], tr["isFraud"]
    Xva, yva = va[feature_cols], va["isFraud"]

    if retrain_from is None:
        # Train the three models (USE: the only place training happens).
        models = train_ensemble(Xtr, ytr, Xva, yva, feature_cols)
        # Calibrate the ensemble on validation (USE: isotonic regression maps
        # raw averaged probabilities to honest probabilities, so a threshold
        # chosen on MoMTSim means something when applied to PaySim).
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(ensemble_prob(models, Xva), yva)
        # Threshold from validation only (see best_f1_threshold).
        thr = best_f1_threshold(yva, iso.predict(ensemble_prob(models, Xva)))
        bundle = {"models": models, "iso": iso, "thr": thr}
    else:
        # Reuse a previously trained bundle (USE: cross-dataset tests must use
        # the EXACT model + calibrator + threshold from in-domain training).
        bundle = retrain_from

    # Temporal TEST split of the test table (USE: for cross-dataset runs this
    # is the other dataset's future period — never its training rows).
    _, _, te = temporal_split(test_table)
    Xte, yte = te[feature_cols], te["isFraud"]

    # Final test probabilities: ensemble -> calibration (USE: same transform
    # the validation threshold was tuned on).
    prob = bundle["iso"].predict(ensemble_prob(bundle["models"], Xte))
    m = evaluate(yte, prob, bundle["thr"])
    m.update({"experiment": name, "train_rows": int(len(Xtr)),
              "test_rows": int(len(Xte)), "minutes": round((time.time() - t0) / 60, 2)})
    print(f"{name}: PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
          f"Recall={m['recall']:.4f} F1={m['f1']:.4f} MCC={m['mcc']:.4f} "
          f"FNR={m['fnr']:.4f} thr={bundle['thr']:.3f}")

    # Collect curve/confusion data for the combined figures (USE: drawn once
    # at the end so E1–E4 share axes and are honestly comparable).
    if curves is not None:
        fpr, tpr, _ = roc_curve(yte, prob)
        pr, rc, _ = precision_recall_curve(yte, prob)
        curves[name] = {"fpr": fpr, "tpr": tpr, "pr": pr, "rc": rc,
                        "prob": prob, "y": yte.values}
    if cms is not None:
        cms[name] = confusion_matrix(yte, (prob >= bundle["thr"]).astype(int), labels=[0, 1])
    return m, bundle


def main():
    # Folders (USE: fresh-clone safety).
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # Load the processed tables (USE: built once by 01_inspect_data.py).
    mom = pd.read_parquet(MOMTSIM_PROCESSED)
    pay = pd.read_parquet(PAYSIM_PROCESSED)
    print(f"MoMTSim V1: {mom.shape}  fraud%={100*mom['isFraud'].mean():.3f}")
    print(f"PaySim    : {pay.shape}  fraud%={100*pay['isFraud'].mean():.4f}")

    curves, cms, all_metrics = {}, {}, []

    # E1 — main result: train MoMTSim, test MoMTSim.
    m1, bundle_mom = run_experiment("E1_MoMTSim_to_MoMTSim", mom, mom, FEATURES,
                                    curves=curves, cms=cms)
    all_metrics.append(m1)
    # E3 — cross-dataset: the E1 model, untouched, tested on PaySim.
    m3, _ = run_experiment("E3_MoMTSim_to_PaySim", mom, pay, FEATURES,
                           retrain_from=bundle_mom, curves=curves, cms=cms)
    all_metrics.append(m3)
    # E2 — baseline: train PaySim, test PaySim (reference paper's setting,
    # but with a fair temporal split instead of a random one).
    m2, bundle_pay = run_experiment("E2_PaySim_to_PaySim", pay, pay, FEATURES,
                                    curves=curves, cms=cms)
    all_metrics.append(m2)
    # E4 — reverse cross-dataset: the E2 model tested on MoMTSim (expected
    # worse: PaySim training saw few, single-pattern fraud cases).
    m4, _ = run_experiment("E4_PaySim_to_MoMTSim", pay, mom, FEATURES,
                           retrain_from=bundle_pay, curves=curves, cms=cms)
    all_metrics.append(m4)

    # E6 — ablation without the simulator-artifact feature (USE: trains a
    # fresh MoMTSim ensemble on FEATURES minus 'amount_equals_balance' and
    # tests on both datasets; if scores barely move, the model is honest).
    feats_e6 = [f for f in FEATURES if f != "amount_equals_balance"]
    m6a, bundle_e6 = run_experiment("E6_noArtifact_MoMTSim_to_MoMTSim", mom, mom, feats_e6)
    all_metrics.append(m6a)
    m6b, _ = run_experiment("E6_noArtifact_MoMTSim_to_PaySim", mom, pay, feats_e6,
                            retrain_from=bundle_e6)
    all_metrics.append(m6b)

    # ---------- Save metrics (USE: the paper's results table comes from here)
    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(RES_DIR / "metrics.csv", index=False)
    (RES_DIR / "metrics.json").write_text(mdf.to_json(orient="records", indent=2))
    print(f"\nSaved metrics: {RES_DIR / 'metrics.csv'}")

    # ---------- Figure: ROC + PR curves for E1–E4 (USE: threshold-free view —
    # the fairest comparison across datasets with different fraud %) --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, c in curves.items():
        axes[0].plot(c["fpr"], c["tpr"], lw=1.2, label=name)
        axes[1].plot(c["rc"], c["pr"], lw=1.2, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curves"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall curves"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eval_roc_pr_curves.png", dpi=150); plt.close(fig)

    # ---------- Figure: confusion matrices (USE: raw TP/FP/TN/FN counts per
    # experiment at the validation-tuned threshold) -------------------------
    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 4))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
        ax.set_xticks([0, 1], ["Pred legit", "Pred fraud"])
        ax.set_yticks([0, 1], ["True legit", "True fraud"])
        ax.set_title(name, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eval_confusion_matrices.png", dpi=150); plt.close(fig)

    # ---------- Figure: calibration curves (USE: does a predicted 0.8 mean
    # ~80% fraud? Compares datasets — calibration is what breaks first under
    # the 10% -> 0.13% prevalence shift) ------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, c in curves.items():
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(c["prob"], bins) - 1
        xs, ys = [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() > 50:
                xs.append(c["prob"][mask].mean()); ys.append(c["y"][mask].mean())
        ax.plot(xs, ys, "o-", lw=1.2, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraud fraction")
    ax.set_title("Calibration curves"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eval_calibration.png", dpi=150); plt.close(fig)

    # ---------- Figure: headline metric comparison (USE: one-glance summary
    # of the four main experiments for the paper/presentation) --------------
    main4 = mdf[mdf["experiment"].str.startswith(("E1", "E2", "E3", "E4"))]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(main4))
    for i, metric in enumerate(["pr_auc", "roc_auc", "recall", "mcc"]):
        ax.bar(x + (i - 1.5) * 0.2, main4[metric], width=0.2, label=metric)
    ax.set_xticks(x, main4["experiment"], rotation=15, fontsize=7)
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Experiment comparison (higher = better)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eval_metric_comparison.png", dpi=150); plt.close(fig)

    # ---------- Feature importance (LightGBM) for both in-domain models -----
    # USE: a first, cheap look at what each model relies on; the SHAP section
    # below gives the rigorous, comparable version.
    for bundle, tag in [(bundle_mom, "momtsim"), (bundle_pay, "paysim")]:
        imp = pd.Series(bundle["models"]["lgbm"].feature_importances_, index=FEATURES)
        if tag == "momtsim" and len(imp) != len(FEATURES):
            imp = pd.Series(bundle["models"]["lgbm"].feature_importances_, index=FEATURES)
        top = imp.sort_values(ascending=False).head(15)[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top.index, top.values, color="#55A868")
        ax.set_title(f"LightGBM feature importance — trained on {tag}")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"eval_importance_{tag}.png", dpi=150); plt.close(fig)

    # ---------- SHAP: MoMTSim model explained on BOTH test sets ------------
    # USE: XAI-stability result — compute mean |SHAP| per feature for the E1
    # LightGBM model on a sample of the MoMTSim test split AND the PaySim
    # test split. Same top features on both = explanations generalise;
    # different rankings = the model's reasoning is simulator-specific.
    # USE: explain the booster's underlying tree model directly (booster_) —
    # shap's TreeExplainer needs the raw LightGBM booster, not the sklearn wrapper.
    explainer = shap.TreeExplainer(bundle_mom["models"]["lgbm"].booster_)
    shap_rows = []
    for table, tag in [(mom, "MoMTSim_test"), (pay, "PaySim_test")]:
        _, _, te = temporal_split(table)
        sample = te[FEATURES].sample(min(len(te), 3000), random_state=SEED)
        sv = explainer.shap_values(sample)
        sv = sv[1] if isinstance(sv, list) else sv
        if sv.ndim == 3:  # newer shap returns (n, features, classes)
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
        mean_abs = np.abs(sv).mean(axis=0)
        for f, v in zip(FEATURES, mean_abs):
            shap_rows.append({"dataset": tag, "feature": f, "mean_abs_shap": float(v)})
        # SHAP bar figure for this test set.
        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(sv, sample, plot_type="bar", show=False, max_display=15)
        plt.title(f"SHAP importance — MoMTSim model on {tag}")
        plt.tight_layout(); plt.savefig(FIG_DIR / f"eval_shap_{tag.lower()}.png", dpi=150); plt.close(fig)
    sdf = pd.DataFrame(shap_rows)
    sdf.to_csv(RES_DIR / "shap_importance.csv", index=False)
    print(f"Saved SHAP: {RES_DIR / 'shap_importance.csv'}")
    print("\nALL DONE — figures in outputs/figures/, tables in outputs/results/")


# Standard entry point.
if __name__ == "__main__":
    main()
