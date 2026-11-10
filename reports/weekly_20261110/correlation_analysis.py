#!/usr/bin/env python3
"""Harmonize field counts and describe future-yield associations, 2324/2526 only.

Run with --counts-only before the model runner, then without it after
analysis/samples.csv exists. No evaluation outcomes select predictors here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / "correlations"
COUNT_MAP = {"FL": "flower", "G": "green", "W": "white", "P": "pink", "R": "red"}
FEATURES = {
    "canopy_size_pixels": "Canopy area",
    "mean_roundness": "Roundness",
    "plant_count": "Plant tracks",
    "flower": "FL*",
    "green": "G*",
    "white": "W*",
    "pink": "P*",
    "red": "R*",
}
STAGES = ["Early", "Middle", "Late"]


def extract_counts():
    """Use manual counts consistently across seasons; preserve every blank."""
    OUT.mkdir(parents=True, exist_ok=True)
    records, cells, paths = [], [], []
    folder = ROOT / "data/real_data/2324_GNV_processed/counting_yield"
    for path in sorted(folder.glob("*.csv")):
        date = pd.to_datetime(path.stem, format="%y%m%d")
        table = pd.read_csv(path, index_col=0)
        table.index = table.index.str.strip()
        paths.append(path)
        for plot in table.columns:
            plot_id = str(plot).strip().upper()
            if not re.fullmatch(r"[BM][1245][1-5]", plot_id):
                continue
            rec = dict(season="2324", date=date, plot=plot_id)
            for code, feature in COUNT_MAP.items():
                value = pd.to_numeric(table.loc[code, plot], errors="coerce")
                rec[feature] = value
                cells.append(dict(season="2324", date=date, plot=plot_id,
                                  feature=feature, value=value,
                                  source_file=str(path.relative_to(ROOT)),
                                  source_sheet="", source_cell=f"row={code};column={plot}"))
            records.append(rec)
    path = ROOT / "check-in & data-col 2526.xlsx"
    paths.append(path)
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for sheet in workbook:
        if not re.fullmatch(r"\d{6}", sheet.title):
            continue
        date = pd.to_datetime(sheet.title, format="%y%m%d")
        rows = list(sheet.iter_rows(min_row=1, max_row=20, values_only=True))
        indices = {str(row[0]).strip(): i for i, row in enumerate(rows) if row}
        assert all(code in indices for code in COUNT_MAP), sheet.title
        columns = {j: str(v).strip().upper() for j, v in enumerate(rows[0])
                   if re.fullmatch(r"[BM][1245][1-5]", str(v).strip().upper())}
        assert len(columns) == 40, sheet.title
        for col, plot in columns.items():
            rec = dict(season="2526", date=date, plot=plot)
            for code, feature in COUNT_MAP.items():
                row = indices[code]
                value = pd.to_numeric(rows[row][col], errors="coerce")
                rec[feature] = value
                cells.append(dict(season="2526", date=date, plot=plot,
                                  feature=feature, value=value,
                                  source_file=str(path.relative_to(ROOT)), source_sheet=sheet.title,
                                  source_cell=f"{openpyxl.utils.get_column_letter(col+1)}{row+1}"))
            records.append(rec)
    workbook.close()
    counts = pd.DataFrame(records).sort_values(["season", "date", "plot"])
    assert set(counts.season) == {"2324", "2526"}
    assert not counts.duplicated(["season", "date", "plot"]).any()
    for col in COUNT_MAP.values():
        valid = counts[col].dropna()
        assert np.isfinite(valid).all() and valid.ge(0).all()
    counts.to_csv(OUT / "counts_harmonized.csv", index=False)
    pd.DataFrame(cells).to_csv(OUT / "counts_cell_audit.csv", index=False)
    audit = dict(
        seasons=["2324", "2526"], excluded_seasons=["2425"],
        source_type="manual field counts in both seasons",
        counts_unit="count per plot per field visit", mapping=COUNT_MAP,
        blank_policy="preserve missing; never replace with zero",
        detector_counts_used=False,
        rows=counts.groupby("season").size().to_dict(),
        date_counts=counts.groupby("season").date.nunique().to_dict(),
        missing_by_season=counts.groupby("season")[list(COUNT_MAP.values())].apply(lambda x: x.isna().sum()).to_dict("index"),
        sources=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
    )
    (OUT / "counts_audit.json").write_text(json.dumps(audit, indent=2))
    return counts


def correlation(x, y, kind="pearson"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float((pearsonr if kind == "pearson" else spearmanr)(x, y).statistic)


def describe(samples):
    rows = []
    for (season, horizon), part in samples.groupby(["season", "horizon_weeks"]):
        for stage in ["All", *STAGES]:
            g = part if stage == "All" else part[part.stage.eq(stage)]
            for feature, label in FEATURES.items():
                valid = g.dropna(subset=[feature, "yield_g"])
                x, y = valid[feature], valid.yield_g
                means = valid.groupby("yield_date")[[feature, "yield_g"]].mean()
                xw = x - valid.groupby("yield_date")[feature].transform("mean")
                yw = y - valid.groupby("yield_date").yield_g.transform("mean")
                rows.append(dict(
                    season=season, horizon_weeks=int(horizon), stage=stage,
                    feature=feature, feature_label=label,
                    n=len(valid), n_origin_dates=valid.canopy_date.nunique(),
                    n_target_dates=valid.yield_date.nunique(), n_plots=valid['plot'].nunique(),
                    pearson_r=correlation(x, y), spearman_rho=correlation(x, y, "spearman"),
                    within_target_date_pearson_r=correlation(xw, yw),
                    date_mean_pearson_r=correlation(means[feature], means.yield_g),
                    x_mean=x.mean(), y_mean=y.mean(),
                ))
    return pd.DataFrame(rows)


def heatmap(ax, frame, columns, xlabel):
    values = frame.pivot(index="feature", columns=xlabel, values="pearson_r").reindex(index=FEATURES, columns=columns)
    ax.imshow(values.to_numpy(), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), [f"{int(v)} week{'s' if int(v)>1 else ''}" if xlabel == "horizon_weeks" else v for v in columns], fontsize=10)
    ax.set_yticks(range(len(FEATURES)), list(FEATURES.values()), fontsize=10)
    for i, feature in enumerate(FEATURES):
        for j, col in enumerate(columns):
            row = frame[(frame.feature == feature) & (frame[xlabel] == col)]
            if row.empty:
                value, n, dates = np.nan, 0, 0
            else:
                row = row.iloc[0]
                value, n, dates = row.pearson_r, row.n, row.n_target_dates
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and abs(value) > .65 else "#19313d"
            ax.text(j, i, f"{label}\nn={n}, d={dates}", ha="center", va="center", fontsize=8.5, color=color)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figures(table):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 5.9), constrained_layout=True)
    for ax, season in zip(axes, ["2324", "2526"]):
        frame = table[(table.season == season) & (table.stage == "All")]
        heatmap(ax, frame, [1, 2, 3], "horizon_weeks")
        ax.set_title(f"Season {season}", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_yticklabels([])
    fig.suptitle("Features and future harvest yield", fontsize=15, fontweight="bold")
    fig.supxlabel("Pearson r: blue = positive; red = negative. n = pairs; d = harvest dates.\n* FL = flower; G = green; W = white; P = pink; R = red. Manual observations.", fontsize=9)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"correlation_overview.{ext}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 5.9), constrained_layout=True)
    for ax, season in zip(axes, ["2324", "2526"]):
        frame = table[(table.season == season) & table.stage.isin(STAGES) & table.horizon_weeks.eq(1)]
        heatmap(ax, frame, STAGES, "stage")
        ax.set_title(f"Season {season}", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_yticklabels([])
    fig.suptitle("One-week relationships across the season", fontsize=14, fontweight="bold")
    fig.supxlabel("Phases = thirds of observed harvest dates. Pearson r; n = pairs; d = dates.\n* FL = flower; G = green; W = white; P = pink; R = red.", fontsize=8)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"correlation_stages_h1.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-only", action="store_true")
    parser.add_argument("--samples", type=Path, default=BASE / "analysis/samples.csv")
    args = parser.parse_args()
    counts = extract_counts()
    if args.counts_only:
        print(f"Wrote {len(counts)} harmonized manual-count rows")
        return
    samples = pd.read_csv(args.samples, dtype={"season": str}, parse_dates=["canopy_date", "yield_date"])
    assert set(samples.season) == {"2324", "2526"}
    assert samples.sample_id.is_unique
    assert samples.yield_date.gt(samples.canopy_date).all()
    samples['plot'] = samples['plot'].str.upper()
    aliases = {"canopy_size_pixels": ["canopy", "canopy_2", "canopy_size_pixels_2"],
               "mean_roundness": ["roundness", "roundness_2", "mean_roundness_2"],
               "plant_count": ["plants", "plants_2", "plant_count_2"]}
    for feature, names in aliases.items():
        if feature not in samples:
            found = [c for c in names if c in samples]
            if found:
                samples[feature] = samples[found[0]]
    # Merge only absent count columns to preserve the runner's exact current inputs.
    needed = [c for c in COUNT_MAP.values() if c not in samples]
    if needed:
        samples = samples.merge(counts[["season", "date", "plot", *needed]].rename(columns={"date": "canopy_date"}),
                                on=["season", "canopy_date", "plot"], how="left", validate="many_to_one")
    assert all(c in samples for c in FEATURES), list(samples.columns)
    table = describe(samples)
    table.to_csv(OUT / "correlations.csv", index=False)
    samples[["sample_id", "season", "canopy_date", "yield_date", "plot", "horizon_weeks", "stage", "yield_g", *FEATURES]].to_csv(OUT / "correlation_samples.csv", index=False)
    figures(table)
    selected = table[table.stage.eq("All") & table.horizon_weeks.eq(1)]
    summary = {season: part.sort_values("pearson_r", key=abs, ascending=False)[["feature", "pearson_r", "spearman_rho", "within_target_date_pearson_r", "n", "n_target_dates"]].to_dict("records")
               for season, part in selected.groupby("season")}
    (OUT / "correlation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    (OUT / "correlation_method.json").write_text(json.dumps(dict(
        sample_source=str(args.samples.relative_to(ROOT)),
        sample_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        feature_timing="Current measurement at forecast origin; no future feature values",
        target="Observed future harvest yield (g/plot), matched within +/-2 days of 1/2/3 calendar weeks",
        phases="Fixed thirds of each season's observed harvest-date schedule; target-harvest phase",
        raw_associations="Pearson and Spearman over plot-harvest pairs; pairwise complete observations",
        within_date_associations="Demean feature and yield within each target harvest date, then calculate Pearson r",
        date_mean_associations="Pearson r over mean feature/yield per target harvest date; NA with fewer than 3 dates",
        inference="Descriptive only. Repeated plots and few dates preclude treating raw row count as independent replication; no p-value claims",
        red_policy="Current red-fruit count included only as a descriptive correlation feature; not a primary predictor",
    ), indent=2))
    print(f"Wrote {len(table)} correlation rows and two figures to {OUT}")


if __name__ == "__main__":
    main()
