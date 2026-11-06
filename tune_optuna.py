"""Tune both CatBoost stages of the Case 4 stack by station-grouped daily F1."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from train_eval_oofstack_case34_v2 import (
    ASOS_EXTRA_SUFFIXES, CB_PARAMS, CatBoostClassifier, HOLDOUT_IDS, JIHU,
    OTHER_SELF_SENSOR, PERIODS, RWIS_EXTRA, RWIS_ID, RWIS_TIME, SEED, TH_GRID,
    daily_f1_from_rows, make_station_folds, period_avg_f1, prep_stage2,
    select_features,
)

PARAM_NAMES = ("iterations", "depth", "learning_rate", "l2_leaf_reg")
STAGE1_SCORE = "stage1_score"
AVAILABLE = "extra_available_count"


def suggest_catboost(trial, stage):
    return {
        "iterations": trial.suggest_int(f"{stage}_iterations", 150, 650, step=50),
        "depth": trial.suggest_int(f"{stage}_depth", 3, 7),
        "learning_rate": trial.suggest_float(f"{stage}_learning_rate", 0.015, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float(f"{stage}_l2_leaf_reg", 1.0, 30.0, log=True),
    }


def fit_model(X, y, params, seed, threads, cat_features=None):
    settings = {**CB_PARAMS, **params, "random_seed": seed, "thread_count": threads}
    model = CatBoostClassifier(**settings, cat_features=cat_features)
    model.fit(X, y)
    return model


def inputs(frame, base_cols, extra_cols, cat_cols):
    base = frame[base_cols].replace([np.inf, -np.inf], np.nan)
    extra = prep_stage2(frame[extra_cols], cat_cols)
    extra[AVAILABLE] = frame[extra_cols].notna().sum(axis=1)
    return base, extra


def stage2_input(extra, probabilities):
    X = extra.copy()
    X.insert(0, STAGE1_SCORE, probabilities)
    return X


def period_daily_groups(frame):
    groups = frame.groupby([RWIS_ID, "date"], sort=True)
    daily = groups["y"].max().reset_index()
    group_index = groups.ngroup().to_numpy(dtype=int)
    periods = []
    for start, end, _ in PERIODS:
        in_period = daily["date"].between(pd.Timestamp(start).date(), pd.Timestamp(end).date())
        periods.append((in_period.to_numpy(), daily.loc[in_period, "y"].to_numpy(dtype=bool)))
    return group_index, len(daily), periods


def best_daily_threshold(probabilities, daily_groups):
    group_index, n_days, periods = daily_groups
    daily_scores = np.full(n_days, -np.inf)
    np.maximum.at(daily_scores, group_index, probabilities)

    thresholds = np.asarray(TH_GRID)
    period_f1 = []
    for in_period, actual in periods:
        predicted = daily_scores[in_period, None] >= thresholds
        tp = (predicted & actual[:, None]).sum(axis=0)
        denominator = predicted.sum(axis=0) + actual.sum()
        f1 = np.divide(2 * tp, denominator, out=np.zeros(len(thresholds)), where=denominator != 0)
        period_f1.append(f1)
    scores = np.mean(period_f1, axis=0)
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def stack_oof(X_base, X_extra, y, folds, p1, p2, cat_cols, threads):
    n_folds = len(folds)
    single_oof = np.empty(len(y))
    stage2_oof = np.empty(len(y))

    # Each pair model excludes the two indicated folds and can serve both
    # directions of the inner cross-fit.
    pair_scores = {}
    for left, right in combinations(range(n_folds), 2):
        fit = ~(folds[left] | folds[right])
        model = fit_model(X_base.loc[fit], y.loc[fit], p1, SEED + 1000 + left * n_folds + right, threads)
        for index in (left, right):
            pair_scores[left, right, index] = model.predict_proba(X_base.loc[folds[index]])[:, 1]

    for outer, valid in enumerate(folds):
        fit = ~valid
        stage1 = fit_model(X_base.loc[fit], y.loc[fit], p1, SEED + outer, threads)
        single_oof[valid] = stage1.predict_proba(X_base.loc[valid])[:, 1]

        # Every stage-1 score used to train this fold's stage 2 was produced
        # without the outer validation stations or the row's own stations.
        inner_oof = np.empty(len(y))
        for inner, inner_valid in enumerate(folds):
            if inner == outer:
                continue
            left, right = sorted((outer, inner))
            inner_oof[inner_valid] = pair_scores[left, right, inner]

        stage2 = fit_model(
            stage2_input(X_extra.loc[fit], inner_oof[fit]), y.loc[fit],
            p2, SEED + 100 + outer, threads, cat_cols,
        )
        stage2_oof[valid] = stage2.predict_proba(
            stage2_input(X_extra.loc[valid], single_oof[valid])
        )[:, 1]

    return single_oof, stage2_oof


def run(args):
    frame = pd.read_parquet(args.dataset)
    frame[RWIS_TIME] = pd.to_datetime(frame[RWIS_TIME])
    frame["date"] = frame[RWIS_TIME].dt.date
    frame[RWIS_ID] = pd.to_numeric(frame[RWIS_ID], errors="coerce").astype("Int64")
    frame["y"] = frame["label_ice"].fillna(0).astype(int)
    extra_cols = RWIS_EXTRA + [
        f"asos{slot}_{name}" for slot in (1, 2) for name in ASOS_EXTRA_SUFFIXES
    ]
    extra_cols = [name for name in extra_cols if name in frame]
    base_cols = [
        name for name in select_features(frame)
        if name not in OTHER_SELF_SENSOR and name not in extra_cols and name != "y"
    ]
    cat_cols = [name for name in extra_cols if not pd.api.types.is_numeric_dtype(frame[name])]
    holdout_mask = frame[RWIS_ID].isin(HOLDOUT_IDS)
    train_mask = frame["dataset_split"].eq("train") & ~holdout_mask & frame[RWIS_ID].notna()
    train = frame.loc[train_mask].reset_index(drop=True)
    holdout = frame.loc[holdout_mask].reset_index(drop=True)
    del frame

    station_ids = sorted(train[RWIS_ID].unique().tolist())
    if len(station_ids) < 4 or set(holdout[RWIS_ID].unique()) != set(HOLDOUT_IDS):
        raise ValueError("Expected at least four training stations and all three holdout stations.")
    fold_ids = make_station_folds(station_ids, 4, seed=1)
    folds = [train[RWIS_ID].isin(ids).to_numpy(dtype=bool) for ids in fold_ids]
    y = train["y"]
    for left, right in combinations(folds, 2):
        if y.loc[~(left | right)].nunique() != 2:
            raise ValueError("An inner training split does not contain both classes.")

    X_base, X_extra = inputs(train, base_cols, extra_cols, cat_cols)
    daily_groups = period_daily_groups(train)
    print(f"Training rows={len(train):,}, base={len(base_cols)}, extras={len(extra_cols)}, folds={fold_ids}", flush=True)
    del train

    best_single_oof = None
    best_score = -1.0

    def objective(trial):
        nonlocal best_single_oof, best_score
        p1 = suggest_catboost(trial, "s1")
        p2 = suggest_catboost(trial, "s2")
        single_oof, stack_probs = stack_oof(X_base, X_extra, y, folds, p1, p2, cat_cols, args.threads)
        threshold, score = best_daily_threshold(stack_probs, daily_groups)
        trial.set_user_attr("threshold", threshold)
        if score > best_score:
            best_single_oof, best_score = single_oof, score
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({
        f"{stage}_{name}": CB_PARAMS[name]
        for stage in ("s1", "s2") for name in PARAM_NAMES
    })
    args.out.mkdir(parents=True, exist_ok=True)

    def save_trials(study, trial):
        study.trials_dataframe().to_csv(args.out / "optuna_trials.csv", index=False, encoding="utf-8-sig")

    study.optimize(objective, n_trials=args.trials, callbacks=[save_trials])
    best = study.best_trial
    p1 = {name: best.params[f"s1_{name}"] for name in PARAM_NAMES}
    p2 = {name: best.params[f"s2_{name}"] for name in PARAM_NAMES}
    threshold = best.user_attrs["threshold"]

    H_base, H_extra = inputs(holdout, base_cols, extra_cols, cat_cols)
    m1 = fit_model(X_base, y, p1, SEED, args.threads)
    stage1_holdout = m1.predict_proba(H_base)[:, 1]
    m2 = fit_model(stage2_input(X_extra, best_single_oof), y, p2, SEED + 100, args.threads, cat_cols)
    holdout["score"] = m2.predict_proba(stage2_input(H_extra, stage1_holdout))[:, 1]
    holdout["pred"] = (holdout["score"] >= threshold).astype(int)

    holdout_f1, by_period = period_avg_f1(holdout, "y", "pred")
    by_station = {
        name: daily_f1_from_rows(holdout.loc[holdout[RWIS_ID].eq(sid)], "y", "pred")
        for sid, name in HOLDOUT_IDS.items()
    }
    result = {
        "best_trial": best.number,
        "baseline_oof_f1": study.trials[0].value,
        "best_oof_f1": best.value,
        "threshold": threshold,
        "stage1_params": p1,
        "stage2_params": p2,
        "holdout_period_avg_f1": holdout_f1,
        "holdout_by_period": by_period,
        "holdout_by_station": by_station,
        "base_features": base_cols,
        "stage2_features": [STAGE1_SCORE, *extra_cols, AVAILABLE],
        "stage2_categorical_features": cat_cols,
    }
    (args.out / "optuna_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    holdout[[RWIS_ID, RWIS_TIME, "label_ice", "score", "pred"]].to_csv(
        args.out / "holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    m1.save_model(str(args.out / "stage1.cbm"))
    m2.save_model(str(args.out / "stage2.cbm"))
    print(
        f"Baseline OOF F1={result['baseline_oof_f1']:.4f} | Best OOF F1={best.value:.4f} "
        f"| Threshold={threshold:.4f} | Holdout F1={holdout_f1:.4f}", flush=True
    )
    print(f"Results: {args.out.resolve()}", flush=True)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(JIHU) / "case4_dataset.parquet")
    parser.add_argument("--trials", type=positive_int, default=20, help="Includes the baseline trial.")
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "optuna_output")
    run(parser.parse_args())
