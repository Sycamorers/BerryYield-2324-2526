"""Audit and summarize 12 predictive models with paired count sources."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "analysis/verified_counts/full_models"
SCENARIOS = ("2324_to_2526", "2526_to_2324", "within_2324", "within_2526")
SCENARIO_LABELS = ("2324 → 2526", "2526 → 2324", "Within 2324", "Within 2526")
SOURCE_FREE = {"Canopy", "Roundness", "Canopy + roundness"}
METHOD_ORDER = ("OLS", "Quadratic OLS",
                "Spline ridge", "Random forest", "SVR", "Gradient boosting",
                "Lag OLS", "Lag ridge", "Lag random forest", "RNN", "LSTM",
                "Transformer")


def main() -> None:
    metrics = []
    phase_metrics = []
    matched_phase_metrics = []
    samples = {}
    expected_predictors = {"canopy_size_pixels", "mean_roundness", "flower", "green", "white", "pink"}
    for source in ("manual", "yolo"):
        folder = OUT / source
        run_summary = json.loads((folder / "summary.json").read_text())
        assert set(run_summary["feature_sets"]["All features"]) == expected_predictors
        assert all(set(names) <= expected_predictors for names in run_summary["feature_sets"].values())
        frame = pd.read_csv(folder / "metrics.csv")
        frame["count_source"] = source
        frame = frame[frame.model.isin(METHOD_ORDER)].copy()
        frame["feature_set"] = frame.feature_set.replace({"All traits": "All features"})
        metrics.append(frame)
        phase = pd.read_csv(folder / "stage_metrics.csv")
        phase = phase[phase.model.isin(METHOD_ORDER)].copy()
        phase["feature_set"] = phase.feature_set.replace({"All traits": "All features"})
        phase["count_source"] = source
        phase_metrics.append(phase)
        matched = pd.read_csv(folder / "matched_phase_training_metrics.csv")
        matched = matched[matched.model.isin(METHOD_ORDER)].copy()
        matched["feature_set"] = matched.feature_set.replace({"All traits": "All features"})
        matched["count_source"] = source
        matched_phase_metrics.append(matched)
        samples[source] = pd.read_csv(folder / "samples.csv",
                                      usecols=["sample_id", "yield_g", "yield_date"])
    all_model_metrics = pd.concat(metrics, ignore_index=True)
    all_model_metrics.to_csv(OUT / "all_model_metrics.csv", index=False)
    all_metrics = all_model_metrics[all_model_metrics.scenario.isin(SCENARIOS)].copy()
    all_metrics.to_csv(OUT / "all_primary_metrics.csv", index=False)
    paired = metrics[0][metrics[0].scenario.isin(SCENARIOS)].merge(
                              metrics[1][metrics[1].scenario.isin(SCENARIOS)],
                              on=["feature_set", "scenario", "horizon_weeks", "model"],
                              suffixes=("_manual", "_yolo"), validate="one_to_one")
    paired["yolo_minus_manual_MAE"] = paired.MAE_yolo - paired.MAE_manual
    paired["yolo_minus_manual_R2"] = paired.R2_yolo - paired.R2_manual
    paired.to_csv(OUT / "paired_source_metrics.csv", index=False)
    all_phase = pd.concat(phase_metrics, ignore_index=True)
    all_phase.to_csv(OUT / "all_phase_metrics.csv", index=False)
    paired_phase = phase_metrics[0].merge(
        phase_metrics[1], on=["feature_set", "scenario", "horizon_weeks", "stage", "model"],
        suffixes=("_manual", "_yolo"), validate="one_to_one")
    paired_phase["yolo_minus_manual_MAE"] = paired_phase.MAE_yolo - paired_phase.MAE_manual
    if not (paired_phase.n_manual == paired_phase.n_yolo).all():
        raise AssertionError("Manual and YOLO phase comparisons use different test sample counts")
    paired_phase.to_csv(OUT / "paired_phase_metrics.csv", index=False)
    matched_phase = pd.concat(matched_phase_metrics, ignore_index=True)
    matched_phase.to_csv(OUT / "matched_phase_training_metrics.csv", index=False)
    paired_matched_phase = matched_phase_metrics[0].merge(
        matched_phase_metrics[1], on=["feature_set", "scenario", "horizon_weeks", "stage", "model"],
        suffixes=("_manual", "_yolo"), validate="one_to_one")
    paired_matched_phase["yolo_minus_manual_MAE"] = paired_matched_phase.MAE_yolo - paired_matched_phase.MAE_manual
    if not (paired_matched_phase.n_manual == paired_matched_phase.n_yolo).all():
        raise AssertionError("Manual and YOLO matched phase fits use different test sample counts")
    paired_matched_phase.to_csv(OUT / "paired_matched_phase_metrics.csv", index=False)

    ids_manual = samples["manual"].set_index("sample_id").sort_index()
    ids_yolo = samples["yolo"].set_index("sample_id").sort_index()
    same_ids = ids_manual.index.equals(ids_yolo.index)
    same_targets = same_ids and ids_manual.yield_g.equals(ids_yolo.yield_g)
    same_dates = same_ids and ids_manual.yield_date.equals(ids_yolo.yield_date)
    same_n = bool((paired.n_manual == paired.n_yolo).all())
    same_harvest_dates = bool((paired.target_dates_manual == paired.target_dates_yolo).all())
    unaffected = paired[paired.feature_set.isin(SOURCE_FREE)]
    max_unaffected_mae_diff = float(unaffected.yolo_minus_manual_MAE.abs().max())
    if not (same_ids and same_targets and same_dates and same_n and same_harvest_dates):
        raise AssertionError("Manual and YOLO comparisons use different test samples")

    learned = all_metrics
    best = learned.loc[learned.groupby(["count_source", "feature_set", "scenario",
                                        "horizon_weeks"]).MAE.idxmin()].copy()
    best.to_csv(OUT / "best_learned_descriptive.csv", index=False)
    rf = all_metrics[all_metrics.model.eq("Random forest")].copy()
    rf.to_csv(OUT / "fixed_random_forest_metrics.csv", index=False)
    phase_parts = []
    for source in ("manual", "yolo"):
        stage = phase_metrics[0 if source == "manual" else 1]
        stage = stage[(stage.scenario.isin(SCENARIOS)) &
                            stage.feature_set.eq("All features") &
                      stage.model.eq("Random forest") &
                      stage.horizon_weeks.eq(1)].copy()
        stage["count_source"] = source
        phase_parts.append(stage)
    phase = pd.concat(phase_parts, ignore_index=True)
    phase.to_csv(OUT / "phase_comparison_h1.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, scenario, label in zip(axes.flat, SCENARIOS, SCENARIO_LABELS):
        block = all_metrics[(all_metrics.scenario == scenario) &
                            (all_metrics.horizon_weeks == 1) &
                            (all_metrics.feature_set == "All features")]
        keep = list(METHOD_ORDER)
        y = np.arange(len(keep))
        for offset, source, color in ((-.18, "manual", "#286b91"),
                                      (.18, "yolo", "#d27c3f")):
            part = block[block.count_source.eq(source)].set_index("model")
            ax.barh(y + offset, [part.loc[m, "MAE"] for m in keep],
                    height=.34, color=color, label=source)
        ax.set_yticks(y, keep)
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel("1-week MAE (g/plot)")
        ax.grid(axis="x", alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "model_comparison_h1.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6), constrained_layout=True)
    feature_order = ["Canopy", "Roundness", "Counts", "Canopy + roundness",
                     "Canopy + counts", "All features"]
    short = ["Canopy", "Roundness", "FL/G/W/P", "Canopy + roundness",
             "Canopy + counts", "All features"]
    for ax, scenario, label in zip(axes.flat, SCENARIOS, SCENARIO_LABELS):
        block = rf[(rf.scenario == scenario) & rf.horizon_weeks.eq(1)]
        x = np.arange(len(feature_order))
        for offset, source, color in ((-.19, "manual", "#286b91"),
                                      (.19, "yolo", "#d27c3f")):
            part = block[block.count_source.eq(source)].set_index("feature_set")
            ax.bar(x + offset, [part.loc[f, "MAE"] for f in feature_order],
                   width=.36, color=color, label=source)
        ax.set_xticks(x, short, rotation=50, ha="right", fontsize=8)
        ax.set_title(label)
        ax.set_ylabel("1-week MAE (g/plot)")
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "feature_set_comparison_h1.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), constrained_layout=True)
    for ax, scenario, label in zip(axes.flat, SCENARIOS, SCENARIO_LABELS):
        block = rf[(rf.scenario == scenario) & rf.feature_set.eq("All features")]
        for source, color in (("manual", "#286b91"), ("yolo", "#d27c3f")):
            part = block[block.count_source.eq(source)].sort_values("horizon_weeks")
            ax.plot(part.horizon_weeks, part.MAE, marker="o", linewidth=2,
                    color=color, label=source)
        ax.set_xticks([1, 2, 3], ["1 week", "2 weeks", "3 weeks"])
        ax.set_title(label)
        ax.set_ylabel("MAE (g/plot)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "horizon_comparison.png", dpi=200)
    plt.close(fig)

    with pd.ExcelWriter(OUT / "full_comparison.xlsx", engine="openpyxl") as writer:
        for name, frame in (("All primary metrics", all_metrics),
                            ("Paired source metrics", paired),
                            ("All model metrics", all_model_metrics),
                            ("All phase metrics", all_phase),
                            ("Paired phase metrics", paired_phase),
                            ("Matched phase fits", matched_phase),
                            ("Paired matched phases", paired_matched_phase),
                            ("Fixed random forest", rf),
                            ("Early middle late", phase),
                            ("Best learned descriptive", best)):
            frame.to_excel(writer, sheet_name=name, index=False)
    summary = {
        "models": len(METHOD_ORDER), "feature_sets": 6,
        "count_sources": ["manual", "yolo"], "horizons_weeks": [1, 2, 3],
        "evaluation_settings": list(SCENARIOS),
        "predictive_features": ["FL", "G", "W", "P", "canopy_size_pixels", "mean_roundness"],
        "mature_red_and_plant_tracks_excluded_from_predictors": True,
        "same_sample_ids": same_ids, "same_yield_targets": same_targets,
        "same_harvest_dates": same_dates, "same_n_by_comparison": same_n,
        "same_target_date_count_by_comparison": same_harvest_dates,
        "maximum_MAE_difference_for_count_independent_inputs": max_unaffected_mae_diff,
        "same_n_by_phase_comparison": True,
        "same_n_by_matched_phase_fit": True,
        "primary_metric_rows": len(all_metrics),
        "paired_metric_rows": len(paired),
        "all_model_metric_rows": len(all_model_metrics),
        "all_phase_metric_rows": len(all_phase),
        "paired_phase_metric_rows": len(paired_phase),
        "matched_phase_metric_rows": len(matched_phase),
        "interpretation": "Fixed settings, no test tuning; all best-model ranks are descriptive after inspection.",
    }
    (OUT / "comparison_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
