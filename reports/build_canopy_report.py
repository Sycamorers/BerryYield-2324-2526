#!/usr/bin/env python3
"""Build the canopy-size versus current/future-yield correlation report.

The analysis uses the final canopy-pipeline pixel area, deliberately excludes
roundness, and matches each canopy observation to yield measured 0, 1, 2, or
3 weeks later.  Future target dates are selected without interpolation using
the nearest recorded harvest within two days of the requested horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
CANOPY_SOURCE = ROOT / "canopy_pipeline" / "final_results" / "gnv_canopy_traits_3years.csv"
YIELD_ROOT = ROOT / "data" / "real_data"
OUTPUT_DIR = ROOT / "detection_only_yield_model" / "canopy_yield_analysis"
REPORT_PATH = ROOT / "reports" / "Canopy_Yield_Correlation_Analysis_Report.docx"
FIGURE_PATH = OUTPUT_DIR / "canopy_yield_lagged_correlations.png"
PAIR_PATH = OUTPUT_DIR / "canopy_yield_lagged_pairs.csv"
OVERALL_PATH = OUTPUT_DIR / "canopy_yield_lagged_correlations_by_season.csv"
STAGE_PATH = OUTPUT_DIR / "canopy_yield_lagged_correlations_by_stage.csv"

SEASONS = ("2324", "2425")
HORIZONS = (0, 1, 2, 3)
FUTURE_HORIZONS = (1, 2, 3)
MATCH_TOLERANCE_DAYS = 2
STAGES = ("Early", "Middle", "Late")


@dataclass(frozen=True)
class Match:
    canopy_date: pd.Timestamp
    expected_yield_date: pd.Timestamp
    yield_date: pd.Timestamp
    date_offset_days: int


def load_canopy() -> pd.DataFrame:
    data = pd.read_csv(CANOPY_SOURCE, dtype={"season": str}, parse_dates=["date"])
    data["season"] = data["season"].str.zfill(4)
    data = data[data["season"].isin(SEASONS)].copy()
    data["plot"] = data["plot"].astype(str).str.strip().str.upper()
    data["canopy_size_pixels"] = pd.to_numeric(data["canopy_size_pixels"], errors="coerce")
    return data[["season", "date", "plot", "canopy_size_pixels"]]


def load_yield() -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for season in SEASONS:
        season_dir = YIELD_ROOT / f"{season}_GNV_processed"
        for path in sorted(season_dir.glob("*/consolidated_summary_with_yield.csv")):
            source = pd.read_csv(path, index_col=0)
            if "Yield (g)" not in source.index:
                continue
            records.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "date": pd.to_datetime(path.parent.name, format="%y%m%d"),
                        "plot": source.columns.astype(str).str.strip().str.upper(),
                        "yield_g": pd.to_numeric(source.loc["Yield (g)"], errors="coerce").to_numpy(),
                    }
                )
            )
    return pd.concat(records, ignore_index=True)


def select_matches(canopy_dates: pd.DatetimeIndex, yield_dates: pd.DatetimeIndex, horizon: int) -> list[Match]:
    matches: list[Match] = []
    for canopy_date in sorted(canopy_dates):
        expected = canopy_date + pd.Timedelta(days=7 * horizon)
        if horizon == 0:
            candidates = yield_dates[yield_dates == expected]
        else:
            candidates = yield_dates[
                (yield_dates > canopy_date)
                & ((yield_dates - expected).days.astype(int).__abs__() <= MATCH_TOLERANCE_DAYS)
            ]
        if len(candidates) == 0:
            continue
        target = min(candidates, key=lambda date: (abs((date - expected).days), date))
        matches.append(
            Match(
                canopy_date=pd.Timestamp(canopy_date),
                expected_yield_date=pd.Timestamp(expected),
                yield_date=pd.Timestamp(target),
                date_offset_days=int((target - expected).days),
            )
        )
    return matches


def assign_stages(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["stage"] = "All season"
    for (season, horizon), subset in result[result["horizon_weeks"] > 0].groupby(
        ["season", "horizon_weeks"], sort=True
    ):
        dates = np.array(sorted(subset["canopy_date"].unique()), dtype="datetime64[ns]")
        date_groups = np.array_split(dates, 3)
        for stage, group in zip(STAGES, date_groups):
            result.loc[
                (result["season"] == season)
                & (result["horizon_weeks"] == horizon)
                & result["canopy_date"].isin(group),
                "stage",
            ] = stage
    return result


def build_pairs(canopy: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for season in SEASONS:
        canopy_season = canopy[canopy["season"] == season]
        yield_season = yields[yields["season"] == season]
        canopy_dates = pd.DatetimeIndex(canopy_season["date"].drop_duplicates())
        yield_dates = pd.DatetimeIndex(yield_season["date"].drop_duplicates())
        for horizon in HORIZONS:
            matches = select_matches(canopy_dates, yield_dates, horizon)
            for match in matches:
                canopy_day = canopy_season[canopy_season["date"] == match.canopy_date].rename(
                    columns={"date": "canopy_date"}
                )
                yield_day = yield_season[yield_season["date"] == match.yield_date][["plot", "yield_g"]]
                paired = canopy_day.merge(yield_day, on="plot", how="inner")
                paired["horizon_weeks"] = horizon
                paired["expected_yield_date"] = match.expected_yield_date
                paired["yield_date"] = match.yield_date
                paired["date_offset_days"] = match.date_offset_days
                pieces.append(paired)
    pairs = pd.concat(pieces, ignore_index=True)
    pairs = pairs.dropna(subset=["canopy_size_pixels", "yield_g"]).copy()
    pairs = assign_stages(pairs)
    columns = [
        "season",
        "horizon_weeks",
        "stage",
        "canopy_date",
        "expected_yield_date",
        "yield_date",
        "date_offset_days",
        "plot",
        "canopy_size_pixels",
        "yield_g",
    ]
    return pairs[columns].sort_values(["season", "horizon_weeks", "canopy_date", "plot"])


def correlation_record(data: pd.DataFrame, season: str, horizon: int, stage: str) -> dict[str, object]:
    pearson = pearsonr(data["canopy_size_pixels"], data["yield_g"])
    spearman = spearmanr(data["canopy_size_pixels"], data["yield_g"])
    return {
        "season": season,
        "horizon_weeks": horizon,
        "stage": stage,
        "n": len(data),
        "canopy_dates": data["canopy_date"].nunique(),
        "yield_dates": data["yield_date"].nunique(),
        "canopy_date_start": data["canopy_date"].min(),
        "canopy_date_end": data["canopy_date"].max(),
        "yield_date_start": data["yield_date"].min(),
        "yield_date_end": data["yield_date"].max(),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def calculate_correlations(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    for (season, horizon), data in pairs.groupby(["season", "horizon_weeks"], sort=True):
        overall_rows.append(correlation_record(data, season, int(horizon), "All season"))
    future = pairs[pairs["horizon_weeks"].isin(FUTURE_HORIZONS)]
    for (season, horizon, stage), data in future.groupby(
        ["season", "horizon_weeks", "stage"], sort=False
    ):
        stage_rows.append(correlation_record(data, season, int(horizon), stage))
    overall = pd.DataFrame(overall_rows).sort_values(["season", "horizon_weeks"])
    stage = pd.DataFrame(stage_rows)
    stage["stage"] = pd.Categorical(stage["stage"], categories=STAGES, ordered=True)
    stage = stage.sort_values(["season", "horizon_weeks", "stage"]).reset_index(drop=True)
    stage["stage"] = stage["stage"].astype(str)
    return overall, stage


def make_figure(overall: pd.DataFrame, stage: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    colors = {"Pearson": "#1F5A7A", "Spearman": "#D68132"}
    for column, season in zip(axes[0], SEASONS):
        data = overall[overall["season"] == season]
        column.plot(data["horizon_weeks"], data["pearson_r"], marker="o", linewidth=2.2,
                    color=colors["Pearson"], label="Pearson r")
        column.plot(data["horizon_weeks"], data["spearman_rho"], marker="s", linewidth=2.2,
                    color=colors["Spearman"], label="Spearman ρ")
        column.axhline(0, color="#777777", linewidth=0.8)
        column.set_xticks(HORIZONS, ["Same day", "1 week", "2 weeks", "3 weeks"])
        column.tick_params(axis="x", rotation=18)
        column.set_ylim(-1, 1)
        column.set_ylabel("Correlation coefficient")
        column.set_title(f"Season {season}: all eligible dates")
        column.grid(axis="y", alpha=0.2)
        column.legend(frameon=False, loc="best")

    for column, season in zip(axes[1], SEASONS):
        data = stage[stage["season"] == season]
        matrix = data.pivot(index="stage", columns="horizon_weeks", values="pearson_r").reindex(
            index=STAGES, columns=FUTURE_HORIZONS
        )
        image = column.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        column.set_xticks(range(3), ["1 week", "2 weeks", "3 weeks"])
        column.set_yticks(range(3), STAGES)
        column.set_title(f"Season {season}: Pearson r by stage")
        for row in range(3):
            for col in range(3):
                value = matrix.iloc[row, col]
                text_color = "white" if abs(value) >= 0.58 else "#222222"
                column.text(col, row, f"{value:.2f}", ha="center", va="center",
                            color=text_color, fontsize=11, fontweight="bold")
    colorbar = fig.colorbar(image, ax=axes[1, :], shrink=0.88, pad=0.025)
    colorbar.set_label("Pearson correlation")
    fig.suptitle("Canopy size versus same-day and future harvest yield", fontsize=16, fontweight="bold")
    fig.savefig(FIGURE_PATH, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def p_text(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def date_range(row: pd.Series) -> str:
    start = pd.Timestamp(row["canopy_date_start"]).strftime("%b %d")
    end = pd.Timestamp(row["canopy_date_end"]).strftime("%b %d, %Y")
    return f"{start}–{end}"


def horizon_label(horizon: int) -> str:
    return "Same day" if horizon == 0 else f"{horizon} week" + ("" if horizon == 1 else "s")


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def embed_report_image(docx_path: Path) -> None:
    """Replace LibreOffice's external image link with an embedded PNG."""
    rels_name = "word/_rels/document.xml.rels"
    document_name = "word/document.xml"
    types_name = "[Content_Types].xml"
    with ZipFile(docx_path) as archive:
        entries = {item.filename: archive.read(item.filename) for item in archive.infolist()}

    rels = etree.fromstring(entries[rels_name])
    relationship_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    image_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    image_relationships = rels.findall(f"{{{relationship_ns}}}Relationship")
    external = next(item for item in image_relationships if item.get("Type") == image_type)
    relationship_id = external.get("Id")
    embedded_image_name = FIGURE_PATH.name
    external.set("Target", f"media/{embedded_image_name}")
    external.attrib.pop("TargetMode", None)
    entries[rels_name] = etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True)
    entries[f"word/media/{embedded_image_name}"] = FIGURE_PATH.read_bytes()

    document = etree.fromstring(entries[document_name])
    drawing_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    office_rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    word_drawing_ns = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
    for blip in document.findall(f".//{{{drawing_ns}}}blip"):
        link_name = f"{{{office_rel_ns}}}link"
        if blip.get(link_name) == relationship_id:
            blip.attrib.pop(link_name)
            blip.set(f"{{{office_rel_ns}}}embed", relationship_id)
    # LibreOffice's HTML importer treats pixel dimensions as physical inches.
    # Override the drawing extent to 6.9 × 4.92 inches, preserving aspect ratio.
    width_emu = str(round(6.9 * 914400))
    height_emu = str(round((6.9 / 11.5 * 8.2) * 914400))
    for extent in document.findall(f".//{{{word_drawing_ns}}}extent"):
        extent.set("cx", width_emu)
        extent.set("cy", height_emu)
    for extent in document.findall(f".//{{{drawing_ns}}}ext"):
        extent.set("cx", width_emu)
        extent.set("cy", height_emu)
    word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    for paragraph in document.findall(f".//{{{word_ns}}}p"):
        paragraph_text = "".join(paragraph.itertext())
        if paragraph_text == "Early-, middle-, and late-stage correlations":
            properties = paragraph.find(f"{{{word_ns}}}pPr")
            if properties is None:
                properties = etree.Element(f"{{{word_ns}}}pPr")
                paragraph.insert(0, properties)
            if properties.find(f"{{{word_ns}}}pageBreakBefore") is None:
                etree.SubElement(properties, f"{{{word_ns}}}pageBreakBefore")
    entries[document_name] = etree.tostring(
        document, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    content_types = etree.fromstring(entries[types_name])
    content_type_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    png_defaults = [
        item
        for item in content_types.findall(f"{{{content_type_ns}}}Default")
        if item.get("Extension", "").lower() == "png"
    ]
    if not png_defaults:
        etree.SubElement(
            content_types,
            f"{{{content_type_ns}}}Default",
            Extension="png",
            ContentType="image/png",
        )
    entries[types_name] = etree.tostring(
        content_types, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    embedded_path = docx_path.with_name("embedded.docx")
    with ZipFile(embedded_path, "w", ZIP_DEFLATED) as archive:
        for name, contents in entries.items():
            archive.writestr(name, contents)
    embedded_path.replace(docx_path)


def build_report(overall: pd.DataFrame, stage: pd.DataFrame) -> None:
    overall_rows = []
    for _, row in overall.iterrows():
        overall_rows.append(
            [
                str(row["season"]),
                horizon_label(int(row["horizon_weeks"])),
                str(int(row["canopy_dates"])),
                str(int(row["n"])),
                f'{row["pearson_r"]:.3f}',
                p_text(row["pearson_p"]),
                f'{row["spearman_rho"]:.3f}',
                p_text(row["spearman_p"]),
            ]
        )
    overall_table = html_table(
        ["Season", "Yield horizon", "Canopy dates", "N", "Pearson r", "p", "Spearman ρ", "p"],
        overall_rows,
    )
    stage_sections: list[str] = []
    for season in SEASONS:
        rows = []
        season_data = stage[stage["season"] == season]
        for _, row in season_data.iterrows():
            rows.append(
                [
                    horizon_label(int(row["horizon_weeks"])),
                    str(row["stage"]),
                    date_range(row),
                    str(int(row["canopy_dates"])),
                    str(int(row["n"])),
                    f'{row["pearson_r"]:.3f} ({p_text(row["pearson_p"])})',
                    f'{row["spearman_rho"]:.3f} ({p_text(row["spearman_p"])})',
                ]
            )
        stage_sections.append(
            f"<h2>Season {season}</h2>"
            + html_table(
                ["Horizon", "Stage", "Canopy-date range", "Dates", "N", "Pearson r (p)", "Spearman ρ (p)"],
                rows,
            )
        )
    interpretations: list[str] = []
    for season in SEASONS:
        season_overall = overall[overall["season"] == season].copy()
        future = season_overall[season_overall["horizon_weeks"] > 0]
        strongest_future = future.iloc[future["pearson_r"].abs().argmax()]
        interpretations.append(
            f"Season {season}: the same-day Pearson correlation is "
            f"{season_overall.loc[season_overall['horizon_weeks'] == 0, 'pearson_r'].iloc[0]:.3f}. "
            f"Among future horizons, the largest absolute Pearson correlation is "
            f"{strongest_future['pearson_r']:.3f} at {horizon_label(int(strongest_future['horizon_weeks']))}. "
            "The stage tables show whether that all-season association is consistent across the "
            "season or concentrated in a particular time period."
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Canopy–Yield Lagged Correlation Analysis</title>
<style>
@page {{ size: 8.5in 11in; margin: 0.65in 0.7in; }}
body {{ font-family: Aptos, Arial, sans-serif; font-size: 9.5pt; line-height: 1.32; color: #2d3439; }}
h1.title {{ color: #1f5a7a; font-size: 24pt; text-align: center; margin: 4pt 0 2pt; }}
.subtitle {{ text-align: center; font-style: italic; color: #56656d; margin: 0 0 20pt; }}
h1 {{ color: #1f5a7a; font-size: 15pt; margin: 15pt 0 6pt; page-break-after: avoid; }}
h2 {{ color: #344955; font-size: 12pt; margin: 11pt 0 5pt; page-break-after: avoid; }}
p {{ margin: 0 0 7pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 5pt 0 12pt; font-size: 8.2pt; page-break-inside: avoid; }}
th {{ background: #1f5a7a; color: white; font-weight: bold; border: 0.6pt solid #8aa0ab; padding: 4pt; }}
td {{ border: 0.6pt solid #a9b7be; padding: 4pt; text-align: center; vertical-align: middle; }}
tbody tr:nth-child(even) td {{ background: #eaf1f5; }}
.figure {{ text-align: center; margin: 8pt 0 2pt; }}
.figure img {{ width: 6.9in; }}
.caption {{ text-align: center; font-style: italic; font-size: 8.5pt; margin-bottom: 8pt; }}
.pagebreak {{ page-break-before: always; }}
.note {{ color: #4d5d65; }}
</style></head><body>
<h1 class="title">Canopy–Yield Lagged Correlation Analysis</h1>
<p class="subtitle">Same-day baseline and 1-, 2-, and 3-week future harvest yield | Seasons 2324 and 2425</p>

<h1>Purpose and analysis design</h1>
<p>This report examines the relationship between plot-level canopy size and harvest yield. The same-day result is used as a baseline. The main analysis compares canopy size with yield measured about 1, 2, or 3 weeks later. The two seasons are analysed separately.</p>
<p>Canopy size is <i>canopy_size_pixels</i> from the final canopy-pipeline output. Roundness is not included. For each future horizon, the nearest recorded harvest date within ±2 days is used. Values are not interpolated, and records without a valid canopy–yield pair are omitted.</p>
<p>For each season and future horizon, the available canopy dates are divided into three time periods: Early, Middle, and Late. The exact dates are shown in the tables. Pearson <i>r</i> describes linear association, while Spearman <i>ρ</i> describes the general increasing or decreasing pattern.</p>

<h1>All eligible dates within each season</h1>
<p>The coefficients below use all complete plot pairs available for each season and horizon. N is the number of plot-level pairs, not the number of dates.</p>
{overall_table}
<div class="figure"><img src="{FIGURE_PATH.as_uri()}" alt="Lagged correlation summary"></div>
<p class="caption">Figure 1. Overall Pearson/Spearman correlations (top) and stage-specific Pearson correlations for future horizons (bottom).</p>

<h1>Interpretation</h1>
{''.join(f'<p>{escape(text)}</p>' for text in interpretations)}
<p>These results describe association only; they do not show that a larger canopy causes higher future yield. Because the same plots were measured more than once, small differences between periods should be interpreted carefully.</p>

<div class="pagebreak"></div>
<h1>Early-, middle-, and late-stage correlations</h1>
<p>Each table reports the three time periods separately for the 1-, 2-, and 3-week horizons. Date range refers to the starting canopy measurement dates.</p>
{''.join(stage_sections)}
</body></html>"""
    with TemporaryDirectory(dir=REPORT_PATH.parent) as temporary_directory:
        temporary = Path(temporary_directory)
        html_path = temporary / "Canopy_Yield_Correlation_Analysis_Report.html"
        html_path.write_text(html, encoding="utf-8")
        completed = subprocess.run(
            [
                "/usr/bin/libreoffice",
                "--headless",
                "--convert-to",
                "docx:Office Open XML Text",
                "--outdir",
                str(temporary),
                str(html_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        converted = temporary / REPORT_PATH.name
        if not converted.exists():
            raise RuntimeError(f"LibreOffice did not create the report: {completed.stdout} {completed.stderr}")
        embed_report_image(converted)
        converted.replace(REPORT_PATH)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canopy = load_canopy()
    yields = load_yield()
    pairs = build_pairs(canopy, yields)
    overall, stage = calculate_correlations(pairs)
    pairs.to_csv(PAIR_PATH, index=False, float_format="%.8g", date_format="%Y-%m-%d")
    overall.to_csv(OVERALL_PATH, index=False, float_format="%.8g", date_format="%Y-%m-%d")
    stage.to_csv(STAGE_PATH, index=False, float_format="%.8g", date_format="%Y-%m-%d")
    make_figure(overall, stage)
    build_report(overall, stage)
    print("Overall correlations:\n" + overall.to_string(index=False))
    print("\nStage correlations:\n" + stage.to_string(index=False))
    print(f"\nWrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
