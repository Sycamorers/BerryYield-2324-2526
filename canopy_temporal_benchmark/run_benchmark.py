#!/usr/bin/env python3
"""Run fixed-parameter canopy-only models on shared, audited forecast samples."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from reports.build_canopy_report import load_canopy, load_yield, build_pairs
from .neural import predict_neural, get_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "canopy_temporal_benchmark" / "artifacts"
FEATURES = ["canopy_lag2", "canopy_lag1", "canopy_current"]
STAGES = ["Early", "Middle", "Late"]
SEEDS = [42, 123, 2026]
MODEL_INFO = {
    "Zero": ("Benchmark", "none"),
    "Train mean": ("Benchmark", "none"),
    "Train median": ("Benchmark", "none"),
    "OLS": ("Traditional regression", "current"),
    "Quadratic OLS": ("Traditional regression", "current"),
    "Spline ridge": ("Traditional regression", "current"),
    "Random forest": ("Machine learning regression", "current"),
    "SVR": ("Machine learning regression", "current"),
    "Gradient boosting": ("Machine learning regression", "current"),
    "Lag OLS": ("Traditional temporal", "history"),
    "Lag ridge": ("Traditional temporal", "history"),
    "Lag random forest": ("Machine learning temporal", "history"),
    "RNN": ("Deep temporal", "history"),
    "LSTM": ("Deep temporal", "history"),
    "Transformer": ("Deep temporal", "history"),
}
NEURAL = {"RNN", "LSTM", "Transformer"}


def build_samples():
    canopy, yields = load_canopy(), load_yield()
    assert not canopy.duplicated(["season", "plot", "date"]).any()
    assert not yields.duplicated(["season", "plot", "date"]).any()
    pairs = build_pairs(canopy, yields)
    pairs = pairs[pairs.horizon_weeks.gt(0)].copy().drop(columns="stage")
    bounds, mapping = [], {}
    for season, part in yields.groupby("season"):
        dates = np.array(sorted(part.date.unique()), dtype="datetime64[ns]")
        for stage, group in zip(STAGES, np.array_split(dates, 3)):
            for date in group:
                mapping[(season, pd.Timestamp(date))] = stage
            bounds.append(dict(season=season, stage=stage,
                start=str(pd.Timestamp(group[0]).date()), end=str(pd.Timestamp(group[-1]).date()),
                harvest_dates=len(group)))
    histories = {}
    for (season, plot), part in canopy.dropna(subset=["canopy_size_pixels"]).groupby(["season", "plot"]):
        histories[(season, plot)] = part.sort_values("date")
    rows = []
    for row in pairs.to_dict("records"):
        history = histories[(row["season"], row["plot"])]; history = history[history.date.le(row["canopy_date"])].tail(3)
        assert len(history) and history.date.max() == row["canopy_date"]
        values = history.canopy_size_pixels.to_numpy(dtype=float)
        row.update(dict(zip(FEATURES, np.pad(values, (3-len(values), 0), mode="edge"))))
        row["history_count"] = len(history)
        row["history_first_date"] = history.date.min()
        row["history_span_days"] = int((row["canopy_date"]-history.date.min()).days)
        row["history_dates"] = "|".join(history.date.dt.strftime("%Y-%m-%d"))
        row["stage"] = mapping[(row["season"], row["yield_date"])]
        row["sample_id"] = f'{row["season"]}_{row["horizon_weeks"]}_{row["canopy_date"]:%Y%m%d}_{row["plot"]}_{row["yield_date"]:%Y%m%d}'
        rows.append(row)
    samples = pd.DataFrame(rows).sort_values(["season", "horizon_weeks", "canopy_date", "plot"])
    assert samples.sample_id.is_unique
    assert np.isfinite(samples[FEATURES+["yield_g"]].to_numpy()).all()
    assert (samples.yield_date > samples.canopy_date).all()
    assert (samples.history_first_date <= samples.canopy_date).all()
    assert (samples.canopy_current == samples.canopy_size_pixels).all()
    return samples, pd.DataFrame(bounds), canopy, yields


def make_estimator(name):
    if name in {"OLS", "Lag OLS"}:
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Quadratic OLS":
        return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression())
    if name == "Spline ridge":
        return make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=2,
            include_bias=False, extrapolation="linear"), Ridge(alpha=10.0))
    if name == "Lag ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name in {"Random forest", "Lag random forest"}:
        return RandomForestRegressor(n_estimators=150, max_depth=3,
            min_samples_leaf=8, random_state=42, n_jobs=1)
    if name == "SVR":
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
            SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")), transformer=StandardScaler())
    if name == "Gradient boosting":
        return HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7,
            min_samples_leaf=10, learning_rate=0.05, l2_regularization=1.0,
            early_stopping=False, random_state=42)
    raise ValueError(name)


def evaluate_frame(frame):
    y, pred = frame.yield_g.to_numpy(), frame.prediction.to_numpy()
    return {"n": len(frame), "origin_dates": frame.canopy_date.nunique(), "target_dates": frame.yield_date.nunique(),
        "MAE": float(mean_absolute_error(y, pred)), "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if len(y)>1 and np.var(y)>0 else np.nan,
        "Bias": float(np.mean(pred-y)), "origin_start": frame.canopy_date.min(),
        "origin_end": frame.canopy_date.max(), "target_start": frame.yield_date.min(), "target_end": frame.yield_date.max()}


def metric_table(predictions, columns):
    rows = []
    for keys, frame in predictions.groupby(columns, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(columns, keys)) | evaluate_frame(frame))
    return pd.DataFrame(rows)


def prediction_batch(train, test, scenario, fit_id, seed_rows, fit_rows, *, evaluation_mode="chronological"):
    assert len(train) and len(test)
    assert set(train.sample_id).isdisjoint(test.sample_id)
    labels_precede_origins = bool(train.yield_date.max() < test.canopy_date.min())
    if evaluation_mode == "chronological":
        assert labels_precede_origins, (scenario, fit_id)
    elif evaluation_mode == "season_holdout":
        assert set(train.season).isdisjoint(set(test.season)), (scenario, fit_id)
    else:
        raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
    tx, vx, ty = train[FEATURES].to_numpy(), test[FEATURES].to_numpy(), train.yield_g.to_numpy()
    outputs = []
    fit_rows.append({"fit_id": fit_id, "scenario": scenario, "horizon_weeks": int(test.horizon_weeks.iloc[0]),
        "stage": str(test.stage.iloc[0]) if scenario.startswith("stage") else "All",
        "n_train": len(train), "n_test": len(test), "train_origin_dates": train.canopy_date.nunique(),
        "train_target_dates": train.yield_date.nunique(), "train_latest_target": train.yield_date.max(),
        "test_first_origin": test.canopy_date.min(), "test_last_origin": test.canopy_date.max(),
        "label_available_before_forecast": labels_precede_origins,
        "evaluation_mode": evaluation_mode, "heldout_labels_excluded": True,
        "training_seasons": "|".join(sorted(set(train.season))),
        "test_seasons": "|".join(sorted(set(test.season))),
        "train_sample_ids": "|".join(train.sample_id),
        "test_sample_ids": "|".join(test.sample_id)})
    for name, (family, info) in MODEL_INFO.items():
        if name == "Zero": pred = np.zeros(len(test))
        elif name == "Train mean": pred = np.full(len(test), ty.mean())
        elif name == "Train median": pred = np.full(len(test), np.median(ty))
        elif name in NEURAL:
            estimates = []
            for seed in SEEDS:
                pred_seed = np.maximum(predict_neural(name, tx, ty, vx, seed=seed), 0)
                estimates.append(pred_seed)
                sf = test.copy(); sf["prediction"] = pred_seed
                sf["model"], sf["seed"], sf["scenario"], sf["fit_id"] = name, seed, scenario, fit_id
                seed_rows.append(sf)
            pred = np.mean(estimates, axis=0)
        else:
            x, xt = (tx[:, -1:], vx[:, -1:]) if info == "current" else (tx, vx)
            estimator = make_estimator(name); estimator.fit(x, ty); pred = estimator.predict(xt)
        pred = np.maximum(np.asarray(pred).reshape(-1), 0)
        assert np.isfinite(pred).all() and len(pred) == len(test)
        frame = test.copy(); frame["prediction"] = pred
        frame["model"], frame["family"], frame["input"] = name, family, info
        frame["scenario"], frame["fit_id"] = scenario, fit_id
        outputs.append(frame)
    return outputs


def cluster_differences(predictions):
    """Paired target-date cluster bootstrap; descriptive with very few dates."""
    rows, rng = [], np.random.default_rng(2026)
    for (scenario, horizon), frame in predictions.groupby(["scenario", "horizon_weeks"]):
        ref = frame[frame.model.eq("Train mean")].set_index("sample_id")
        for model, group in frame.groupby("model"):
            current = group.set_index("sample_id"); assert set(current.index) == set(ref.index)
            paired = current[["yield_g", "yield_date", "prediction"]].copy()
            paired["delta"] = np.abs(paired.prediction-paired.yield_g)-np.abs(ref.loc[paired.index].prediction-paired.yield_g)
            agg = paired.groupby("yield_date").delta.agg(["sum", "count"])
            idx = rng.integers(0, len(agg), size=(2000, len(agg)))
            boot = agg["sum"].to_numpy()[idx].sum(axis=1)/agg["count"].to_numpy()[idx].sum(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append(dict(scenario=scenario, horizon_weeks=horizon, model=model,
                MAE_difference_vs_train_mean=paired.delta.mean(), lower95=lo, upper95=hi,
                target_date_clusters=len(agg), bootstrap_replicates=2000))
    return pd.DataFrame(rows)


def main():
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    samples, bounds, canopy, yields = build_samples()
    samples.to_csv(OUT / "samples.csv", index=False); bounds.to_csv(OUT / "stage_boundaries.csv", index=False)
    pred_frames, seed_frames, fit_rows, unavailable = [], [], [], []
    with threadpool_limits(limits=1):
        for horizon in (1, 2, 3):
            train = samples[samples.season.eq("2324") & samples.horizon_weeks.eq(horizon)]
            test = samples[samples.season.eq("2425") & samples.horizon_weeks.eq(horizon)]
            print(f"Horizon {horizon}: transfer {len(train)} -> {len(test)}", flush=True)
            pred_frames.extend(prediction_batch(train, test, "pooled_transfer", f"transfer_h{horizon}", seed_frames, fit_rows))
            for stage in STAGES:
                tr, te = train[train.stage.eq(stage)], test[test.stage.eq(stage)]
                if len(tr) and len(te):
                    print(f"  stage transfer {stage}: {len(tr)} -> {len(te)}", flush=True)
                    pred_frames.extend(prediction_batch(tr, te, "stage_transfer", f"stage_transfer_h{horizon}_{stage}", seed_frames, fit_rows))
                else:
                    unavailable.append(dict(scenario="stage_transfer", horizon_weeks=horizon,
                        stage=stage, reason="No historical training pairs" if not len(tr) else "No evaluation pairs",
                        n_train=len(tr), n_test=len(te)))
            for origin, te in test.groupby("canopy_date", sort=True):
                tr = test[test.yield_date.lt(origin)]
                if tr.yield_date.nunique() >= 2:
                    print(f"  forward {origin:%Y-%m-%d}: {len(tr)} -> {len(te)}", flush=True)
                    pred_frames.extend(prediction_batch(tr, te, "pooled_forward", f"forward_h{horizon}_{origin:%Y%m%d}", seed_frames, fit_rows))
                else:
                    unavailable.append(dict(scenario="pooled_forward", horizon_weeks=horizon, stage="All", origin=origin,
                        reason="Fewer than two completed target dates", n_train=len(tr), n_test=len(te)))
                for stage, stage_test in te.groupby("stage"):
                    stage_train = tr[tr.stage.eq(stage)]
                    if stage_train.yield_date.nunique() >= 2:
                        pred_frames.extend(prediction_batch(stage_train, stage_test, "stage_forward",
                            f"stage_forward_h{horizon}_{stage}_{origin:%Y%m%d}", seed_frames, fit_rows))
                    else:
                        unavailable.append(dict(scenario="stage_forward", horizon_weeks=horizon, stage=stage, origin=origin,
                            reason="Fewer than two completed same-stage target dates", n_train=len(stage_train), n_test=len(stage_test)))
    predictions, seed_predictions = pd.concat(pred_frames, ignore_index=True), pd.concat(seed_frames, ignore_index=True)
    assert not predictions.duplicated(["scenario", "model", "sample_id"]).any()
    for _, group in predictions.groupby(["scenario", "horizon_weeks"]):
        sets = [set(g.sample_id) for _, g in group.groupby("model")]
        assert len(sets) == len(MODEL_INFO) and all(x == sets[0] for x in sets)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    seed_predictions.to_csv(OUT / "neural_seed_predictions.csv", index=False)
    metric_table(predictions, ["scenario", "horizon_weeks", "model", "family", "input"]).to_csv(OUT / "metrics.csv", index=False)
    metric_table(predictions, ["scenario", "horizon_weeks", "stage", "model", "family", "input"]).to_csv(OUT / "stage_metrics.csv", index=False)
    metric_table(seed_predictions, ["scenario", "horizon_weeks", "model", "seed"]).to_csv(OUT / "neural_seed_metrics.csv", index=False)
    matched = []
    for horizon, forward in predictions[predictions.scenario.eq("pooled_forward")].groupby("horizon_weeks"):
        ids = set(forward.sample_id)
        matched.append(predictions[predictions.horizon_weeks.eq(horizon) & predictions.scenario.eq("pooled_transfer") & predictions.sample_id.isin(ids)])
        matched.append(forward)
    metric_table(pd.concat(matched), ["scenario", "horizon_weeks", "model"]).to_csv(OUT / "matched_transfer_forward_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(OUT / "fit_audit.csv", index=False)
    pd.DataFrame(unavailable).to_csv(OUT / "unavailable.csv", index=False)
    cluster_differences(predictions).to_csv(OUT / "paired_mae_bootstrap.csv", index=False)
    sources = [ROOT / "canopy_pipeline/final_results/gnv_canopy_traits_3years.csv"]
    sources += sorted((ROOT / "data/real_data").glob("*_GNV_processed/*/consolidated_summary_with_yield.csv"))
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    import torch
    versions["torch"] = torch.__version__
    audit = {"run_date": "2026-10-26", "report_date": "2026-10-27", "elapsed_seconds": round(time.time()-started, 2),
        "versions": versions, "data_sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "models": MODEL_INFO, "neural_config": get_config(), "neural_seeds": SEEDS,
        "neural_reported_prediction": "Mean of three nonnegative seed predictions",
        "prediction_constraint": "max(0, raw_prediction) for every method",
        "input_definition": "Only final canopy_size_pixels; latest or trailing 3 observations, repeat-earliest left padding. No dates, yield lags, plant count, roundness, weather or season as predictors.",
        "history_policy": "Same-season same-plot observations at/before origin; irregular intervals retained, no future interpolation.",
        "horizon_matching": "1/2/3 calendar weeks; nearest observed harvest within +/-2 days; target follows origin",
        "stage_definition": "Fixed thirds of ordered harvest dates per season, shared across horizons and methods; retrospective calendar phases, not physiological stages.",
        "forward_training": "Only 2425 labels with target_date strictly before origin; at least 2 distinct target dates",
        "hyperparameters": "Fixed exploratory defaults; no test-based tuning, model selection or early stopping",
        "metric_aggregation": "Observation-weighted MAE, RMSE, R2 and mean(prediction-yield), per horizon/scenario",
        "source_canopy_rows": canopy.groupby("season").size().to_dict(),
        "source_yield_summary": yields.groupby("season").yield_g.agg(["count", "mean", "median", "min", "max"]).to_dict("index"),
        "sample_counts": samples.groupby(["season", "horizon_weeks"]).size().rename("n").reset_index().to_dict("records"),
        "history_counts": samples.history_count.value_counts().sort_index().to_dict(),
        "history_span_days_summary": samples.history_span_days.describe().to_dict(),
        "fits": len(fit_rows), "prediction_rows": len(predictions),
        "checks": {"future_labels_excluded_from_all_fits": True, "same_test_samples_across_models": True,
            "no_duplicate_predictions": True, "history_ends_at_origin": True, "2526_excluded_no_yield_labels": True}}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, default=str)+"\n")
    print(f"Completed {len(fit_rows)} shared-data fits; {len(predictions)} predictions in {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__": main()
