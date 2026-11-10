"""Summarize healthy feature distributions and their future-yield correlations."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "analysis" / "verified_counts"
FEATURES = ["canopy_size_pixels", "mean_roundness", "plant_count",
            *[f"manual_{x}" for x in ("FL", "G", "W", "P", "R")],
            *[f"yolo_{x}" for x in ("FL", "G", "W", "P", "R")]]
PREDICTIVE = {"canopy_size_pixels", "mean_roundness",
              *[f"{source}_{code}" for source in ("manual", "yolo")
                for code in ("FL", "G", "W", "P")]}


def main():
    data = pd.read_csv(OUT / "healthy_plot_dates.csv", dtype={"season": str})
    sample = pd.read_csv(OUT / "model_samples.csv", dtype={"season": str})
    distributions = []
    for season, group in data.groupby("season"):
        for feature in FEATURES:
            values = group[feature].dropna()
            distributions.append({"season": season, "feature": feature,
                                  "prediction_eligible": feature in PREDICTIVE,
                                  "n": len(values), "missing": group[feature].isna().sum(),
                                  "zeros": int(values.eq(0).sum()), "mean": values.mean(),
                                  "median": values.median(), "q1": values.quantile(.25),
                                  "q3": values.quantile(.75), "max": values.max()})
    pd.DataFrame(distributions).to_csv(OUT / "verified_feature_distributions.csv", index=False)

    correlations = []
    for (season, horizon), group in sample.groupby(["season", "horizon_weeks"]):
        for feature in FEATURES:
            pair = group[[feature, "yield_g"]].dropna()
            correlations.append({"season": season, "horizon_weeks": horizon,
                                 "prediction_eligible": feature in PREDICTIVE,
                                 "feature": feature, "n": len(pair),
                                 "harvest_dates": group.yield_date.nunique(),
                                 "pearson_r": pair[feature].corr(pair.yield_g),
                                 "spearman_r": pair[feature].corr(pair.yield_g, method="spearman")})
    pd.DataFrame(correlations).to_csv(OUT / "verified_feature_yield_correlations.csv", index=False)

    yields = pd.read_csv(OUT.parent / "yields.csv", dtype={"season": str})
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2), constrained_layout=True)
    for ax, feature, label, scale in zip(
        axes,
        ("canopy_size_pixels", "mean_roundness", "plant_count", "yield_g"),
        ("Canopy (million pixels)", "Roundness", "Plant tracks", "Yield (g/plot)"),
        (1e6, 1, 1, 1),
    ):
        frame = yields if feature == "yield_g" else data
        values = [frame.loc[frame.season.eq(season), feature].dropna() / scale
                  for season in ("2324", "2526")]
        ax.boxplot(values, labels=["2324", "2526"], showfliers=False, widths=.6)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "verified_image_yield_distributions.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(11, 5), constrained_layout=True)
    for row, season in enumerate(("2324", "2526")):
        part = data[data.season.eq(season)]
        for col, code in enumerate(("FL", "G", "W", "P", "R")):
            ax = axes[row, col]
            ax.boxplot([part[f"manual_{code}"], part[f"yolo_{code}"]],
                       labels=["Manual", "YOLO"], showfliers=False, widths=.6)
            ax.set_title(f"{season} {code}")
            ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "manual_yolo_distributions.png", dpi=200)
    plt.close(fig)

    one = pd.DataFrame(correlations).query("horizon_weeks == 1")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, season in zip(axes, ("2324", "2526")):
        part = one[one.season.eq(season)].set_index("feature")
        labels = ["Canopy", "Roundness", "FL M", "FL Y", "G M", "G Y",
                  "W M", "W Y", "P M", "P Y"]
        features = ["canopy_size_pixels", "mean_roundness",
                    "manual_FL", "yolo_FL", "manual_G", "yolo_G",
                    "manual_W", "yolo_W", "manual_P", "yolo_P"]
        values = [part.loc[f, "pearson_r"] for f in features]
        ax.bar(np.arange(len(features)), values,
               color=["#376f91" if v >= 0 else "#c57358" for v in values])
        ax.set_xticks(np.arange(len(features)), labels, rotation=55, ha="right")
        ax.set_title(season)
        ax.set_ylim(-.55, .85)
        ax.axhline(0, color="black", linewidth=.6)
        ax.set_ylabel("Pearson r with next harvest yield")
    fig.savefig(OUT / "verified_feature_yield_correlations.png", dpi=200)
    plt.close(fig)
    with pd.ExcelWriter(OUT / "verified_results.xlsx", engine="openpyxl") as writer:
        for name, filename in (
            ("Healthy data", "healthy_plot_dates.csv"),
            ("Date QC", "date_qc.csv"),
            ("Manual YOLO", "manual_yolo_summary.csv"),
            ("Distributions", "verified_feature_distributions.csv"),
            ("Correlations", "verified_feature_yield_correlations.csv"),
            ("Simple model", "simple_model_metrics.csv"),
        ):
            frame = pd.read_csv(OUT / filename)
            frame.to_excel(writer, sheet_name=name, index=False)
    print("Wrote verified distribution and correlation tables/figures")


if __name__ == "__main__":
    main()
