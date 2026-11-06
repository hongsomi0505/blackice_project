"""v2: replaces v1's loose "비슷한 역할" extra-variable substitute with the ACTUAL columns
her RWIS_extra/ASOS_extra name (read directly from her model_config.json):
  RWIS_extra = ["최대순간풍향 (Degree)","최대순간풍속 (m/s)","적설 (cm)","운고1","운고2","운고3","운량"]
  ASOS_extra = ["강수량(mm)","3시간신적설(cm)","운형(운형약어)","최저운고(100m)","현상번호(국내식)"]
ASOS_extra maps 1:1 onto case3/4's own asos1_*/asos2_* columns (same underlying ASOS schema,
verified by exact name match) -- a faithful port, not an approximation.
RWIS_extra's 7 names are a SUBSET of case3/4's own SELF_SENSOR_COLS (both pipelines process the
same raw RWIS schema). Verified empirically: in case3/4, these 7 columns are ~14-64% non-null
for the 14 TRAIN stations but EXACTLY 0% non-null (fully NaN) for the true 3 holdout stations --
i.e. "real during training, structurally absent at the unobserved stations", the exact same
present/missing pattern her RWIS_extra has at her own excluded official stations. So this is not
a byte-exact replication of her specific neighbor-RWIS source (case3/4 doesn't have one), but it
correctly recreates the STRUCTURAL ROLE her stage-2 "extra" design relies on (same semantics,
same missingness pattern, same extra_available_count idea) using case3/4's own matching signal.
The other 9 SELF_SENSOR_COLS names (not in her RWIS_extra list) stay fully excluded everywhere,
same as every other case3/4 run in this session.
"""
import os, sys, time, json
# Portable: this folder (wherever the script actually lives), not a fixed session path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from train_eval_v2 import (JIHU, RWIS_TIME, RWIS_ID, HOLDOUT_IDS, PERIODS,
                            select_features, period_avg_f1, daily_f1_from_rows,
                            make_station_folds, log)
from train_eval_unobserved import SELF_SENSOR_COLS

SEED = 20261103
CB_PARAMS = dict(iterations=450, depth=5, learning_rate=0.05, loss_function="Logloss",
                  class_weights=[1, 1], l2_leaf_reg=8, thread_count=2,
                  verbose=False, allow_writing_files=False)

RWIS_EXTRA = ["최대순간풍향 (Degree)", "최대순간풍속 (m/s)", "적설 (cm)", "운고1", "운고2", "운고3", "운량"]
ASOS_EXTRA_SUFFIXES = ["강수량(mm)", "3시간신적설(cm)", "운형(운형약어)", "최저운고(100m)", "현상번호(국내식)"]
OTHER_SELF_SENSOR = sorted(set(SELF_SENSOR_COLS) - set(RWIS_EXTRA))

TH_GRID = sorted(set(list(np.arange(0.0002, 0.01, 0.0002)) + list(np.arange(0.01, 0.99, 0.01))))


def fit_cb(X, y, seed):
    m = CatBoostClassifier(**{**CB_PARAMS, "random_seed": seed})
    m.fit(X.replace([np.inf, -np.inf], np.nan), y)
    return m


def prep_stage2(X, cat_features):
    # 소미's own documented convention (model_config.json): "numeric NaN; categorical __MISSING__"
    X = X.copy()
    for c in cat_features:
        mask = X[c].isna()
        X[c] = X[c].astype(str)
        X.loc[mask, c] = "__MISSING__"
    num_cols = [c for c in X.columns if c not in cat_features]
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    return X


def fit_cb2(X, y, seed, cat_features):
    m = CatBoostClassifier(**{**CB_PARAMS, "random_seed": seed},
                            cat_features=cat_features if cat_features else None)
    m.fit(prep_stage2(X, cat_features), y)
    return m


def run_case_stack(case_name, k_folds=4):
    t0 = time.time()
    log(f"\n========================= OOFSTACK-v2 {case_name} =========================")
    df = pd.read_parquet(f"{JIHU}/{case_name}_dataset.parquet")
    df[RWIS_TIME] = pd.to_datetime(df[RWIS_TIME])
    df["date"] = df[RWIS_TIME].dt.date
    df[RWIS_ID] = pd.to_numeric(df[RWIS_ID], errors="coerce").astype("Int64")
    df["y"] = df["label_ice"].fillna(0).astype(int)

    # candidate pool keeps RWIS_extra (unlike every other script here) so it can be routed to
    # stage2; the other 9 self-sensor names stay excluded everywhere, same as always.
    all_candidates = [f for f in select_features(df) if f not in OTHER_SELF_SENSOR and f != "y"]

    extra_cols = list(RWIS_EXTRA)
    for pfx in ("asos1_", "asos2_"):
        extra_cols += [f"{pfx}{s}" for s in ASOS_EXTRA_SUFFIXES]
    extra_cols = [c for c in extra_cols if c in df.columns]
    base_cols = [c for c in all_candidates if c not in extra_cols]
    cat_extra = [c for c in extra_cols if not pd.api.types.is_numeric_dtype(df[c])]
    log(f"base(stage1)={len(base_cols)} extra(stage2-only)={len(extra_cols)} -> {extra_cols}")
    log(f"categorical among extras (passed to CatBoost as cat_features): {cat_extra}")

    df["extra_available_count"] = df[extra_cols].notna().sum(axis=1)

    train_mask = df["dataset_split"] == "train"
    holdout_mask = df[RWIS_ID].isin(HOLDOUT_IDS.keys())
    train_df = df.loc[train_mask].reset_index(drop=True)
    hold_df = df.loc[holdout_mask].reset_index(drop=True)
    log("train_rows:", len(train_df), "train_pos:", int(train_df["y"].sum()),
        "hold_rows:", len(hold_df), "hold_pos:", int(hold_df["y"].sum()))
    log("extra_available_count mean -- train:", train_df["extra_available_count"].mean(),
        "holdout:", hold_df["extra_available_count"].mean())

    station_ids = sorted(train_df[RWIS_ID].dropna().unique().tolist())
    folds = make_station_folds(station_ids, k_folds, seed=1)
    log("station folds:", folds)

    # ---- stage1: broad CatBoost (base_cols only), station OOF ----
    stage1_oof = np.zeros(len(train_df))
    for fi, fold_stations in enumerate(folds):
        val_idx = train_df[RWIS_ID].isin(fold_stations).values
        fit_idx = ~val_idx
        tf = time.time()
        m1 = fit_cb(train_df.loc[fit_idx, base_cols], train_df.loc[fit_idx, "y"], seed=SEED + fi)
        stage1_oof[val_idx] = m1.predict_proba(
            train_df.loc[val_idx, base_cols].replace([np.inf, -np.inf], np.nan))[:, 1]
        log(f"  stage1 fold {fi}: val_stations={fold_stations} took={time.time()-tf:.1f}s")
    train_df["stage1_score"] = stage1_oof

    # ---- stage2: corrector CatBoost on stage1_score (OOF) + extras + extra_available_count ----
    stage2_cols = ["stage1_score"] + extra_cols + ["extra_available_count"]
    stage2_oof = np.zeros(len(train_df))
    for fi, fold_stations in enumerate(folds):
        val_idx = train_df[RWIS_ID].isin(fold_stations).values
        fit_idx = ~val_idx
        tf = time.time()
        m2 = fit_cb2(train_df.loc[fit_idx, stage2_cols], train_df.loc[fit_idx, "y"], seed=SEED + 100 + fi,
                      cat_features=cat_extra)
        stage2_oof[val_idx] = m2.predict_proba(
            prep_stage2(train_df.loc[val_idx, stage2_cols], cat_extra))[:, 1]
        log(f"  stage2 fold {fi}: took={time.time()-tf:.1f}s")
    train_df["oof"] = stage2_oof

    best = {"th": None, "f1": -1}
    for th in TH_GRID:
        train_df["pred_tmp"] = (train_df["oof"] >= th).astype(int)
        f1, _ = period_avg_f1(train_df, "y", "pred_tmp")
        if f1 > best["f1"]:
            best = {"th": float(th), "f1": f1}
    log(f"[OOF threshold selection] th={best['th']:.4f} OOF-period-avg-F1={best['f1']:.4f}")
    try:
        ap_oof = average_precision_score(train_df["y"], train_df["oof"])
        auc_oof = roc_auc_score(train_df["y"], train_df["oof"])
    except Exception:
        ap_oof = auc_oof = float("nan")
    log(f"[OOF overall] AP={ap_oof:.4f} AUC={auc_oof:.4f}")

    # ---- final refit on full train (stage1 full -> stage2 uses OOF stage1 scores, no leak) ----
    m1f = fit_cb(train_df[base_cols], train_df["y"], seed=SEED)
    hold_df["stage1_score"] = m1f.predict_proba(hold_df[base_cols].replace([np.inf, -np.inf], np.nan))[:, 1]
    m2f = fit_cb2(train_df[stage2_cols], train_df["y"], seed=SEED + 100, cat_features=cat_extra)
    hold_df["score"] = m2f.predict_proba(prep_stage2(hold_df[stage2_cols], cat_extra))[:, 1]
    hold_df["pred"] = (hold_df["score"] >= best["th"]).astype(int)

    f1_hold, detail_hold = period_avg_f1(hold_df, "y", "pred")
    overall = daily_f1_from_rows(hold_df, "y", "pred")
    try:
        ap_hold = average_precision_score(hold_df["y"], hold_df["score"])
        auc_hold = roc_auc_score(hold_df["y"], hold_df["score"])
    except Exception:
        ap_hold = auc_hold = float("nan")

    log(f"[TRUE HOLDOUT 3-station] period_avg_F1={f1_hold:.4f} pooled_daily={overall} "
        f"AP={ap_hold:.4f} AUC={auc_hold:.4f}")
    log(f"[TRUE HOLDOUT detail by period] {detail_hold}")
    for sid, sname in HOLDOUT_IDS.items():
        s = hold_df[hold_df[RWIS_ID] == sid]
        sr = daily_f1_from_rows(s, "y", "pred")
        log(f"    station {sname}({sid}) overall: ice_days={sr['n_ice_days']} TP={sr['tp']} FP={sr['fp']} FN={sr['fn']} F1={sr['f1']:.3f}")
    log(f"case {case_name} oofstack-v2 total_time={time.time()-t0:.1f}s")
    return dict(case=case_name, oof_f1=best["f1"], oof_ap=ap_oof, oof_auc=auc_oof,
                hold_f1=f1_hold, hold_overall=overall, hold_ap=ap_hold, hold_auc=auc_hold, th=best["th"])


if __name__ == "__main__":
    # case4 only: case3 (organizer's raw, unprocessed data) is out of scope for submission.
    results = {}
    for case in ["case4"]:
        results[case] = run_case_stack(case)
        log(f"\n>>> RESULT SO FAR: {json.dumps(results, default=str, ensure_ascii=False)}\n")
    log("\n===== FINAL SUMMARY =====")
    log(json.dumps(results, default=str, ensure_ascii=False, indent=2))
