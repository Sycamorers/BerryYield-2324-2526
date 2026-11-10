"""A compact, leakage-controlled yield benchmark on the verified two-season set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
OUT = BASE / "analysis" / "verified_counts"
IMAGE = ["canopy_size_pixels", "mean_roundness"]
MANUAL = ["manual_FL", "manual_G", "manual_W", "manual_P"]
YOLO = ["yolo_FL", "yolo_G", "yolo_W", "yolo_P"]
FEATURE_SETS = {
    "Image": IMAGE,
    "Image + manual": IMAGE + MANUAL,
    "Image + YOLO": IMAGE + YOLO,
    "YOLO": YOLO,
}


def predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], method: str) -> np.ndarray:
    if method == "Ridge":
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif method == "Random forest":
        estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                          max_features=0.8, random_state=42, n_jobs=1)
    else:
        raise ValueError(method)
    estimator.fit(train[columns], train.yield_g)
    return np.maximum(0, estimator.predict(test[columns]))


def main() -> None:
    observations = pd.read_csv(OUT / "healthy_plot_dates.csv", dtype={"season": str})
    observations["plot"] = observations["plot"].str.upper()
    samples = pd.read_csv(BASE / "analysis/samples.csv", dtype={"season": str})
    samples = samples[["season", "plot", "canopy_date", "yield_date", "yield_g",
                       "horizon_weeks", "sample_id"]]
    merged = samples.merge(observations, left_on=["season", "plot", "canopy_date"],
                           right_on=["season", "plot", "date"], how="inner",
                           validate="many_to_one")
    merged["canopy_date"] = pd.to_datetime(merged.canopy_date)
    merged["yield_date"] = pd.to_datetime(merged.yield_date)
    merged.to_csv(OUT / "model_samples.csv", index=False)
    results = []
    predictions = []
    methods = ["Ridge", "Random forest"]
    for horizon in (1, 2, 3):
        all_h = merged[merged.horizon_weeks.eq(horizon)].copy()
        for scenario, train_season, test_season in (
            ("2324_to_2526", "2324", "2526"),
            ("2526_to_2324", "2526", "2324"),
        ):
            train = all_h[all_h.season.eq(train_season)]
            test = all_h[all_h.season.eq(test_season)]
            for feature_set, columns in FEATURE_SETS.items():
                for method in methods:
                    yhat = predict(train, test, columns, method)
                    for row, value in zip(test.itertuples(), yhat):
                        predictions.append({"scenario": scenario, "horizon_weeks": horizon,
                                            "feature_set": feature_set, "method": method,
                                            "sample_id": row.sample_id, "yield_date": row.yield_date.date(),
                                            "observed_g": row.yield_g, "predicted_g": value})
                    results.append({"scenario": scenario, "horizon_weeks": horizon,
                                    "feature_set": feature_set, "method": method,
                                    "n": len(test), "target_dates": test.yield_date.nunique(),
                                    "MAE": mean_absolute_error(test.yield_g, yhat),
                                    "RMSE": np.sqrt(mean_squared_error(test.yield_g, yhat)),
                                    "R2": r2_score(test.yield_g, yhat)})
        for season in ("2324", "2526"):
            part = all_h[all_h.season.eq(season)]
            for feature_set, columns in FEATURE_SETS.items():
                for method in methods:
                    scored = []
                    for origin in sorted(part.canopy_date.unique()):
                        train = part[part.yield_date.lt(origin)]
                        test = part[part.canopy_date.eq(origin)]
                        if train.yield_date.nunique() < 2 or test.empty:
                            continue
                        yhat = predict(train, test, columns, method)
                        for row, value in zip(test.itertuples(), yhat):
                            scored.append((row.yield_g, value, row.yield_date))
                            predictions.append({"scenario": "within_" + season,
                                                "horizon_weeks": horizon,
                                                "feature_set": feature_set, "method": method,
                                                "sample_id": row.sample_id,
                                                "yield_date": row.yield_date.date(),
                                                "observed_g": row.yield_g, "predicted_g": value})
                    if scored:
                        observed, forecast, dates = map(np.array, zip(*scored))
                        results.append({"scenario": "within_" + season,
                                        "horizon_weeks": horizon,
                                        "feature_set": feature_set, "method": method,
                                        "n": len(scored), "target_dates": len(set(dates)),
                                        "MAE": mean_absolute_error(observed, forecast),
                                        "RMSE": np.sqrt(mean_squared_error(observed, forecast)),
                                        "R2": r2_score(observed, forecast)})
    pd.DataFrame(results).to_csv(OUT / "simple_model_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "simple_model_predictions.csv", index=False)
    metadata = {
        "scope": "2324 and 2526 healthy plot-dates; 2425 excluded",
        "task": "predict recorded harvest g/plot nearest to 7, 14 or 21 days after image date (±2 days)",
        "model_settings": {"Ridge": "StandardScaler fitted to training data, alpha=10",
                           "Random forest": "100 trees, min_samples_leaf=5, max_features=0.8, seed=42"},
        "within_season_rule": "train only on target harvest dates before the image date; require two completed dates",
        "input_policy": "Only FL, G, W, P, canopy size and roundness are predictors; R and retained plant-track count are descriptive only",
        "feature_sets": FEATURE_SETS,
        "sample_rows": merged.groupby(["season", "horizon_weeks"]).size().to_dict(),
    }
    metadata["sample_rows"] = {f"{s}_h{h}": int(n) for (s, h), n in metadata["sample_rows"].items()}
    (OUT / "simple_model_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(results).query('horizon_weeks == 1').to_string(index=False))


if __name__ == "__main__":
    main()
