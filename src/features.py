"""
features.py — The ONE shared pipeline: harmonisation + feature engineering.
USE: This file is the heart of the "one pipeline, two datasets" design.
01_inspect_data.py, 02_eda_visuals.py and 03_train_evaluate.py all call the
functions here, so MoMTSim and PaySim are processed by IDENTICAL code.
That identity is what makes the cross-dataset experiment (train MoMTSim ->
test PaySim) a fair test instead of two unrelated pipelines.
"""

# --- Imports ---
# pandas/numpy: table handling and numeric feature maths.
import numpy as np
import pandas as pd

# Local config: the 5 harmonised type names (USE: one-hot columns are built
# for exactly these types, in this order, for every dataset).
from config import TYPES

# --- Column renaming map for MoMTSim (USE: MoMTSim names its columns
#     differently from PaySim; this map translates them to the PaySim names,
#     which we use as the common language of the whole project) ---
MOMTSIM_RENAME = {
    "transactionType": "type",
    "initiator": "nameOrig",
    "oldBalInitiator": "oldbalanceOrg",
    "newBalInitiator": "newbalanceOrig",
    "recipient": "nameDest",
    "oldBalRecipient": "oldbalanceDest",
    "newBalRecipient": "newbalanceDest",
}

# --- Transaction-type synonyms (USE: MoMTSim uses words like DEPOSIT /
#     WITHDRAWAL where PaySim uses CASH_IN / CASH_OUT; this map converts every
#     spelling to the 5 harmonised names in config.TYPES) ---
TYPE_SYNONYMS = {
    "DEPOSIT": "CASH_IN",
    "CASHIN": "CASH_IN",
    "CASH_IN": "CASH_IN",
    "WITHDRAWAL": "CASH_OUT",
    "CASHOUT": "CASH_OUT",
    "CASH_OUT": "CASH_OUT",
    "TRANSFER": "TRANSFER",
    "PAYMENT": "PAYMENT",
    "DEBIT": "DEBIT",
}


def load_harmonised(path, source):
    """
    USE: Read one raw CSV (MoMTSim or PaySim) and return a DataFrame with the
    common column names and the 5 harmonised type names.
    `source` is 'momtsim' or 'paysim' and only decides the rename map.
    """
    # Read only the columns we need (USE: drops PaySim's isFlaggedFraud rule
    # column on purpose — it is not a real feature and MoMTSim does not have it).
    df = pd.read_csv(path)

    # Rename MoMTSim columns to the common names (USE: after this line both
    # datasets have identical columns and the rest of the code never branches).
    if source == "momtsim":
        df = df.rename(columns=MOMTSIM_RENAME)

    # Keep exactly the 10 common columns, in a fixed order (USE: guarantees the
    # feature builder always sees the same input shape for both datasets).
    keep = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest",
            "newbalanceDest", "isFraud"]
    df = df[keep]

    # Normalise the type text (USE: upper-case, '-'/' ' -> '_', then synonym
    # map, so 'Cash-Out', 'CASH_OUT' and 'WITHDRAWAL' all become 'CASH_OUT').
    norm = (df["type"].astype(str).str.upper().str.strip()
            .str.replace("-", "_", regex=False).str.replace(" ", "_", regex=False))
    df["type"] = norm.map(TYPE_SYNONYMS).fillna(norm)

    # Downcast numeric columns (USE: float32/int32 halves memory on 6M+ rows;
    # balances are kept float64 inside build_features where exact equality
    # checks like amount == oldbalanceOrg matter — see note there).
    df["step"] = df["step"].astype(np.int32)
    df["isFraud"] = df["isFraud"].astype(np.int8)
    return df


def build_features(df):
    """
    USE: Turn a harmonised transaction table into model-ready features.
    Input : DataFrame from load_harmonised()
    Output: (X, y, meta) — X = feature DataFrame (same columns for every
            dataset), y = isFraud series, meta = the step column (kept
            outside X because the temporal split is done on step, and step
            itself is also a feature via hour/day below).

    IMPORTANT DESIGN RULES (each is commented where it is applied):
    1. Account IDs are NEVER model features (they don't transfer across
       datasets) — they are only used to compute behavioural features.
    2. Behavioural features use ONLY past transactions (shifted), so no
       future information leaks into a row.
    """
    # Work on a copy sorted by time (USE: behavioural features and the later
    # temporal split both require strict chronological order).
    df = df.sort_values("step", kind="mergesort").reset_index(drop=True)

    # Factorise account IDs to integer codes (USE: groupby on integer codes is
    # much faster and lighter than on raw ID strings; codes are internal only).
    orig_code = pd.factorize(df["nameOrig"])[0].astype(np.int32)
    dest_code = pd.factorize(df["nameDest"])[0].astype(np.int32)
    df["_orig"] = orig_code
    df["_dest"] = dest_code

    # Work in float64 for the balance maths (USE: equality/zero checks below,
    # e.g. amount == oldbalanceOrg, are the actual PaySim fraud artifact;
    # float32 rounding would corrupt them).
    amount = df["amount"].astype(np.float64)
    old_o = df["oldbalanceOrg"].astype(np.float64)
    new_o = df["newbalanceOrig"].astype(np.float64)
    old_d = df["oldbalanceDest"].astype(np.float64)
    new_d = df["newbalanceDest"].astype(np.float64)

    X = pd.DataFrame(index=df.index)

    # ============ A. RAW FEATURES (USE: the common columns every model sees) ==
    X["step"] = df["step"].astype(np.float64)
    X["amount"] = amount
    X["oldbalanceOrg"] = old_o
    X["newbalanceOrig"] = new_o
    X["oldbalanceDest"] = old_d
    X["newbalanceDest"] = new_d

    # ============ B. BALANCE-ERROR FEATURES ==================================
    # USE: In an honest transaction the balances reconcile exactly. Fraud
    # breaks the arithmetic, so these differences are the strongest signal in
    # both datasets — and they are computable identically on both.
    X["errorOrig"] = new_o + amount - old_o        # != 0 when sender books don't reconcile
    X["errorDest"] = old_d + amount - new_d        # != 0 when receiver books don't reconcile
    X["diffOrig"] = old_o - new_o                  # how much the sender actually lost
    X["diffDest"] = new_d - old_d                  # how much the receiver actually gained
    X["amount_ratio_orig"] = amount / (old_o + 1)  # amount relative to sender's balance
    X["amount_ratio_dest"] = amount / (old_d + 1)  # amount relative to receiver's balance
    X["log_amount"] = np.log1p(amount)             # compresses the huge amount range
    # USE: 1 when the transaction empties the sender's account — a classic
    # account-takeover pattern in both simulators.
    X["orig_emptied"] = ((new_o == 0) & (old_o > 0)).astype(np.float64)
    # USE: 1 when amount == sender's full balance. NOTE: this is the known
    # PaySim simulator artifact; the ablation experiment (E6) retrains without
    # it to prove the model is not living on this one shortcut.
    X["amount_equals_balance"] = (amount == old_o).astype(np.float64)

    # ============ C. TIME FEATURES (USE: step = 1 hour in both datasets) ======
    step = df["step"].astype(np.int64)
    X["hour_of_day"] = (step % 24).astype(np.float64)          # daily rhythm
    X["day"] = (step // 24).astype(np.float64)                 # which day of the month
    X["is_night"] = ((step % 24) <= 5).astype(np.float64)      # 00:00-05:00 flag

    # ============ D. BEHAVIOURAL FEATURES — PAST ONLY ========================
    # USE: A fraudster behaves differently from the account's own history.
    # Every feature below is computed from transactions BEFORE the current
    # row only (cumcount/cumsum minus the current row, shifted last-step),
    # so nothing from the future leaks in. Computed inside each dataset with
    # this same code — the model sees behaviour NUMBERS, never identities,
    # which is exactly why these features transfer MoMTSim -> PaySim.
    g_orig = df.groupby("_orig", sort=False)
    past_count_o = g_orig.cumcount().astype(np.float64)                    # sender's past txn count
    past_sum_o = g_orig["amount"].cumsum() - amount                        # sender's past amount total
    past_mean_o = (past_sum_o / past_count_o.replace(0, np.nan)).fillna(0)  # sender's usual amount
    last_step_o = g_orig["step"].shift(1)                                  # sender's previous txn step
    X["orig_past_count"] = past_count_o
    X["orig_past_amount_mean"] = past_mean_o
    # USE: how unusual this amount is for THIS sender (>1 = bigger than usual).
    X["amount_vs_orig_mean"] = np.where(past_count_o > 0, amount / (past_mean_o + 1), 1.0)
    # USE: hours since the sender's last transaction (first txn: use its own step).
    X["time_since_last_orig"] = (step - last_step_o.fillna(step)).astype(np.float64)

    # USE: number of DISTINCT receivers this sender had already paid before
    # (fraud often sprays money to a brand-new receiver). Built from a
    # first-time-pair flag accumulated per sender, past rows only.
    first_pair = (df.groupby(["_orig", "_dest"], sort=False).cumcount() == 0).astype(np.float64)
    X["orig_unique_dest_past"] = first_pair.groupby(df["_orig"], sort=False).cumsum() - first_pair

    g_dest = df.groupby("_dest", sort=False)
    past_count_d = g_dest.cumcount().astype(np.float64)                    # receiver's past txn count
    past_sum_d = g_dest["amount"].cumsum() - amount                        # receiver's past amount total
    past_mean_d = (past_sum_d / past_count_d.replace(0, np.nan)).fillna(0)
    last_step_d = g_dest["step"].shift(1)                                  # receiver's previous txn step
    X["dest_past_count"] = past_count_d
    X["dest_past_amount_mean"] = past_mean_d
    X["time_since_last_dest"] = (step - last_step_d.fillna(step)).astype(np.float64)
    # USE: 1 when this receiver has never been seen before in the dataset
    # (brand-new receivers are a known fraud destination pattern).
    X["dest_is_new"] = (past_count_d == 0).astype(np.float64)

    # ============ E. TRANSACTION TYPE, ONE-HOT ===============================
    # USE: one binary column per harmonised type, ALWAYS all 5 in the same
    # order (config.TYPES), so a MoMTSim-trained model's columns line up
    # exactly with PaySim's columns at cross-dataset test time.
    for t in TYPES:
        X[f"type_{t}"] = (df["type"] == t).astype(np.float64)

    # Target (USE: kept separate so it can never leak into the features).
    y = df["isFraud"].astype(int)

    # Meta (USE: step is returned for the temporal split in the train script).
    meta = df["step"].copy()
    return X.astype(np.float32), y, meta


# --- The exact model feature list (USE: single source of truth; the ablation
#     E6 simply removes 'amount_equals_balance' from this list) ---
FEATURES = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "errorOrig", "errorDest", "diffOrig", "diffDest",
    "amount_ratio_orig", "amount_ratio_dest", "log_amount", "orig_emptied",
    "amount_equals_balance", "hour_of_day", "day", "is_night",
    "orig_past_count", "orig_past_amount_mean", "amount_vs_orig_mean",
    "time_since_last_orig", "orig_unique_dest_past", "dest_past_count",
    "dest_past_amount_mean", "time_since_last_dest", "dest_is_new",
] + [f"type_{t}" for t in TYPES]
