import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

RWIS_TIME = "관측시각 (KST)"
RWIS_ID = "관측 지점번호"
# Portable by default: resolves to the folder this file itself lives in, so if you place
# case4_dataset.parquet (produced by case_builder.py) in the same folder as these .py files,
# everything works with zero edits. Override with the JIHU_DATA_DIR env var if you'd rather
# keep data elsewhere, e.g. JIHU_DATA_DIR=/path/to/data python train_eval_oofstack_case34_v2.py
JIHU = os.environ.get("JIHU_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

HOLDOUT_IDS = {33411: "삼봉", 12410: "부곡교", 23402: "해전"}
PERIODS = [
    ("2024-11-01", "2025-02-28", "기간1(24-25)"),
    ("2025-11-01", "2026-02-28", "기간2(25-26)"),
]

NON_FEATURE_EXACT = {
    RWIS_TIME, RWIS_ID, "관측 지점명", "label_ice", "target_ice", "target_road_temp",
    "dataset_split", "RWIS_ID", "RWIS_지점명", "고속도로", "관측소 종류",
    "노면상태", "노면온도 (광학)",
    "struct_ice_rate", "struct_temp_mean", "struct_night_temp_mean",
}


def log(*a):
    print(*a, flush=True)


def select_features(df):
    feats = []
    for c in df.columns:
        if c in NON_FEATURE_EXACT or c.endswith("_ID") or c == "RWIS_ID":
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            feats.append(c)
    return feats


def daily_f1_from_rows(frame, actual_col, pred_col):
    daily = frame.groupby([RWIS_ID, "date"]).agg(
        actual=(actual_col, "max"), pred=(pred_col, "max")
    ).reset_index()
    a, p = daily["actual"].values.astype(int), daily["pred"].values.astype(int)
    tp = int(((a == 1) & (p == 1)).sum())
    fp = int(((a == 0) & (p == 1)).sum())
    fn = int(((a == 1) & (p == 0)).sum())
    tn = int(((a == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1,
                n_days=len(daily), n_ice_days=int(a.sum()))


def period_avg_f1(frame, actual_col, pred_col, periods=PERIODS):
    scores = []
    detail = {}
    for start, end, label in periods:
        m = (frame["date"] >= pd.to_datetime(start).date()) & (frame["date"] <= pd.to_datetime(end).date())
        r = daily_f1_from_rows(frame.loc[m], actual_col, pred_col)
        scores.append(r["f1"])
        detail[label] = r
    return float(np.mean(scores)), detail


def lgbm_bag(X_tr, y_tr, X_pred, n_seeds, n_estimators=150, is_unbalance=True, **kw):
    probas = []
    imps = []
    for seed in range(n_seeds):
        m = lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
            min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            is_unbalance=is_unbalance, random_state=seed, n_jobs=-1, verbosity=-1, **kw,
        )
        m.fit(X_tr, y_tr)
        probas.append(m.predict_proba(X_pred)[:, 1])
        imps.append(m.feature_importances_)
    return np.mean(probas, axis=0), np.mean(imps, axis=0)


def downsample_negatives(X, y, neg_ratio, seed):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y.values == 1)[0]
    neg_idx = np.where(y.values == 0)[0]
    n_neg = min(len(neg_idx), int(len(pos_idx) * neg_ratio))
    keep_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep)
    return X.iloc[keep], y.iloc[keep]


def lgbm_bag_downsampled(X_tr, y_tr, X_pred, n_seeds, neg_ratio=25, n_estimators=150):
    # classic technique for extreme imbalance: keep all positives, randomly subsample negatives
    # down to neg_ratio:1, train several such resamples and average (also much cheaper to fit).
    probas = []
    imps = []
    for seed in range(n_seeds):
        Xs, ys = downsample_negatives(X_tr, y_tr, neg_ratio, seed)
        m = lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
            min_child_samples=15, subsample=0.9, subsample_freq=1, colsample_bytree=0.7,
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
        m.fit(Xs, ys)
        probas.append(m.predict_proba(X_pred)[:, 1])
        imps.append(m.feature_importances_)
    return np.mean(probas, axis=0), np.mean(imps, axis=0)


def make_station_folds(station_ids, k, seed=0):
    rng = np.random.RandomState(seed)
    ids = list(station_ids)
    rng.shuffle(ids)
    return [ids[i::k] for i in range(k)]


def run_case(case_name, feature_subset=None, oof_seeds=3, final_seeds=9, k_folds=4):
    log(f"\n========================= {case_name} =========================")
    df = pd.read_parquet(f"{JIHU}/{case_name}_dataset.parquet")
    df[RWIS_TIME] = pd.to_datetime(df[RWIS_TIME])
    df["date"] = df[RWIS_TIME].dt.date
    df[RWIS_ID] = pd.to_numeric(df[RWIS_ID], errors="coerce").astype("Int64")
    df["y"] = df["label_ice"].fillna(0).astype(int)

    feats = feature_subset or select_features(df)
    log("n_features:", len(feats))

    train_mask = df["dataset_split"] == "train"
    holdout_mask = df[RWIS_ID].isin(HOLDOUT_IDS.keys())
    train_df = df.loc[train_mask].reset_index(drop=True)
    hold_df = df.loc[holdout_mask].reset_index(drop=True)

    station_ids = sorted(train_df[RWIS_ID].dropna().unique().tolist())
    folds = make_station_folds(station_ids, k_folds, seed=1)
    log("station folds:", folds)

    oof_lgbm = np.zeros(len(train_df))
    oof_ds = np.zeros(len(train_df))
    fold_importances = []

    for fi, fold_stations in enumerate(folds):
        val_idx = train_df[RWIS_ID].isin(fold_stations).values
        fit_idx = ~val_idx
        X_fit, y_fit = train_df.loc[fit_idx, feats], train_df.loc[fit_idx, "y"]
        X_val = train_df.loc[val_idx, feats]
        log(f"  fold {fi}: val_stations={fold_stations} fit_rows={fit_idx.sum()} val_rows={val_idx.sum()} fit_pos={y_fit.sum()}")

        p_lgbm, imp1 = lgbm_bag(X_fit, y_fit, X_val, n_seeds=oof_seeds)
        oof_lgbm[val_idx] = p_lgbm
        p_ds, imp2 = lgbm_bag_downsampled(X_fit, y_fit, X_val, n_seeds=oof_seeds)
        oof_ds[val_idx] = p_ds
        fold_importances.append(imp1)
        gc.collect()

    train_df["oof_lgbm"] = oof_lgbm
    train_df["oof_ds"] = oof_ds
    train_df["oof_avg"] = (oof_lgbm + oof_ds) / 2

    avg_importance = pd.Series(np.mean(fold_importances, axis=0), index=feats).sort_values(ascending=False)

    # choose best model-combo + threshold using OOF daily F1 (train stations only, never touches
    # the true 3 holdout stations) so model selection itself cannot leak from the real test set.
    best = {"combo": None, "th": None, "f1": -1}
    for combo_name in ["oof_lgbm", "oof_ds", "oof_avg"]:
        for th in np.arange(0.05, 0.95, 0.02):
            train_df["pred_tmp"] = (train_df[combo_name] >= th).astype(int)
            f1, _ = period_avg_f1(train_df, "y", "pred_tmp")
            if f1 > best["f1"]:
                best = {"combo": combo_name, "th": float(th), "f1": f1}
    log(f"[OOF model selection] best combo={best['combo']} th={best['th']:.2f} OOF-period-avg-F1={best['f1']:.3f}")
    for combo_name in ["oof_lgbm", "oof_ds", "oof_avg"]:
        # report each combo's own best threshold too, for transparency
        b2 = (-1, None)
        for th in np.arange(0.05, 0.95, 0.02):
            train_df["pred_tmp"] = (train_df[combo_name] >= th).astype(int)
            f1, _ = period_avg_f1(train_df, "y", "pred_tmp")
            if f1 > b2[0]:
                b2 = (f1, th)
        log(f"    {combo_name}: best OOF F1={b2[0]:.3f} @ th={b2[1]:.2f}")

    # ---- final fit on ALL 14 train stations with the winning combo, predict true holdout ----
    X_all, y_all = train_df[feats], train_df["y"]
    X_hold = hold_df[feats]
    hold_df["y"] = hold_df["label_ice"].fillna(0).astype(int)

    final_probs = {}
    final_imp = None
    if best["combo"] in ("oof_lgbm", "oof_avg"):
        p_lgbm_h, final_imp = lgbm_bag(X_all, y_all, X_hold, n_seeds=final_seeds)
        final_probs["oof_lgbm"] = p_lgbm_h
    if best["combo"] in ("oof_ds", "oof_avg"):
        p_ds_h, imp_ds = lgbm_bag_downsampled(X_all, y_all, X_hold, n_seeds=final_seeds)
        final_probs["oof_ds"] = p_ds_h
        if final_imp is None:
            final_imp = imp_ds
    if best["combo"] == "oof_avg":
        hold_proba = (final_probs["oof_lgbm"] + final_probs["oof_ds"]) / 2
    else:
        hold_proba = final_probs[best["combo"]]

    hold_df["pred"] = (hold_proba >= best["th"]).astype(int)
    final_f1, detail = period_avg_f1(hold_df, "y", "pred")
    log(f">>> {case_name} TRUE HOLDOUT 최종 F1 (OOF로 고른 설정 그대로 적용): {final_f1:.4f}")
    for label, r in detail.items():
        log(f"  [{label}] days={r['n_days']} ice_days={r['n_ice_days']} TP={r['tp']} FP={r['fp']} FN={r['fn']} "
            f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    # per-station breakdown across the whole holdout period combined, for visibility
    for sid, sname in HOLDOUT_IDS.items():
        s = hold_df[hold_df[RWIS_ID] == sid]
        sr = daily_f1_from_rows(s, "y", "pred")
        log(f"    station {sname}({sid}) overall: ice_days={sr['n_ice_days']} TP={sr['tp']} FP={sr['fp']} FN={sr['fn']} F1={sr['f1']:.3f}")

    log("\nTop 25 OOF feature importance (avg across folds, lgbm side):")
    for name, val in avg_importance.head(25).items():
        log(f"  {name}: {val:.1f}")

    return {
        "case": case_name, "n_features": len(feats),
        "oof_selection": best, "final_f1": final_f1, "period_detail": detail,
        "top_features": avg_importance.head(40).index.tolist(),
    }


if __name__ == "__main__":
    # case4 only: case3 (organizer's raw, unprocessed data) is out of scope for submission.
    which = sys.argv[1] if len(sys.argv) > 1 else "case4"
    out = {}
    if which in ("case3", "both"):
        out["case3"] = run_case("case3")
    if which in ("case4", "both"):
        out["case4"] = run_case("case4")
    with open(f"{JIHU}/eval_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log("\nsaved eval_results_v2.json")
