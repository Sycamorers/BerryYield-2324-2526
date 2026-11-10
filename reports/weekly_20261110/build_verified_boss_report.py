"""Build the supervisor report from the verified paired 12-model comparison."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json

import pandas as pd
from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "analysis/verified_counts"
FULL = DATA / "full_models"
TARGET = ROOT / "Jing-2026-1110-yield prediction new data-revised.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS = {"w": W, "a": A, "r": R, "wp": WP}
METHODS = ("OLS", "Quadratic OLS", "Spline ridge", "Random forest", "SVR",
           "Gradient boosting", "Lag OLS", "Lag ridge", "Lag random forest",
           "RNN", "LSTM", "Transformer")
SCENARIOS = (("2324_to_2526", "2324 → 2526"),
             ("2526_to_2324", "2526 → 2324"),
             ("within_2324", "Within 2324"),
             ("within_2526", "Within 2526"))
STAGES = ("Early", "Middle", "Late")


def text_of(node):
    return "".join(node.xpath(".//w:t/text()", namespaces=NS))


def set_text(node, value):
    texts = node.xpath(".//w:t", namespaces=NS)
    if not texts:
        raise ValueError("Template has no text run")
    pieces = str(value).split(";")
    clean = pieces[0].rstrip()
    for piece in pieces[1:]:
        sentence = piece.strip()
        if sentence:
            clean += ". " + sentence[0].upper() + sentence[1:]
    texts[0].text = clean
    for other in texts[1:]:
        other.text = ""


def table_from(template, matrix):
    table = deepcopy(template)
    old_rows = table.findall(f"{{{W}}}tr")
    for row in old_rows:
        table.remove(row)
    for i, values in enumerate(matrix):
        row = deepcopy(old_rows[0 if i == 0 else 1])
        cells = row.findall(f"{{{W}}}tc")
        if len(cells) != len(values):
            raise ValueError("Incorrect table width")
        for cell, value in zip(cells, values):
            set_text(cell, value)
        table.append(row)
    return table


def fmt_pair(row):
    return f"{row.MAE_manual:.1f} / {row.MAE_yolo:.1f}"


def winner(group, source):
    if group.empty:
        return "no eligible forecast"
    row = group.loc[group[f"MAE_{source}"].idxmin()]
    return f"{row.model} ({row[f'MAE_{source}']:.1f} g/plot)"


def main():
    with ZipFile(TARGET) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    document = etree.fromstring(parts["word/document.xml"])
    body = document.find(f"{{{W}}}body")
    original = list(body)
    assert text_of(original[7]) == "Research activities last week"
    assert "10.28.2026" in text_of(original[4])
    assert text_of(original[19]).startswith("Yield model development:")
    set_text(original[12], text_of(original[12]))
    heading_template = next(n for n in original[25:] if n.tag == f"{{{W}}}p" and text_of(n).startswith("1. "))
    normal_template = original[26]
    picture_template = next(n for n in original[25:] if n.xpath(".//a:blip", namespaces=NS))
    tables = {}
    for node in original[25:]:
        if node.tag == f"{{{W}}}tbl":
            first = node.find(f"{{{W}}}tr")
            tables.setdefault(len(first.findall(f"{{{W}}}tc")), node)
    assert 4 in tables and 5 in tables

    # Retain the author's weekly activity section. Only update the obsolete model summary.
    set_text(original[21], "Compared 12 predictive models and six feature sets using manual and YOLO counts in both season-transfer directions and chronological within-season evaluations. Model performance varied by season, forecast horizon and counting source.")
    set_text(original[24], "The model comparison covers 12 models, three forecast horizons and early, middle and late harvest periods. The final section identifies one model using its average rank across the matched evaluations; individual results remain visible in the tables.")
    for node in original[25:]:
        body.remove(node)

    def add_para(value, heading=False, conclusion=False, highlight=False, note=False):
        node = deepcopy(heading_template if heading else normal_template)
        set_text(node, value)
        if heading:
            ppr = node.find(f"{{{W}}}pPr")
            if ppr is not None:
                page_break = ppr.find(f"{{{W}}}pageBreakBefore")
                if page_break is not None:
                    ppr.remove(page_break)
        if highlight or note:
            run = node.find(f".//{{{W}}}r")
            rpr = run.find(f"{{{W}}}rPr")
            if rpr is None:
                rpr = etree.Element(f"{{{W}}}rPr")
                run.insert(0, rpr)
            if highlight:
                etree.SubElement(rpr, f"{{{W}}}b")
                etree.SubElement(rpr, f"{{{W}}}color", {f"{{{W}}}val": "24577A"})
            if note:
                etree.SubElement(rpr, f"{{{W}}}i")
                etree.SubElement(rpr, f"{{{W}}}color", {f"{{{W}}}val": "606060"})
                etree.SubElement(rpr, f"{{{W}}}sz", {f"{{{W}}}val": "18"})
        body.append(node)

    def add_table(matrix):
        body.append(table_from(tables[len(matrix[0])], matrix))

    image_rels = []
    def add_picture(image_name):
        rid = f"rId{101 + len(image_rels)}"
        node = deepcopy(picture_template)
        blips = node.xpath(".//a:blip", namespaces=NS)
        assert len(blips) == 1
        blips[0].set(f"{{{R}}}embed", rid)
        path = DATA / image_name
        with Image.open(path) as picture:
            width_px, height_px = picture.size
        for extent in node.xpath(".//wp:extent | .//a:xfrm/a:ext", namespaces=NS):
            width_emu = int(extent.get("cx"))
            extent.set("cy", str(round(width_emu * height_px / width_px)))
        body.append(node)
        name = Path(image_name).name
        parts[f"word/media/{name}"] = path.read_bytes()
        image_rels.append((rid, name))

    paired = pd.read_csv(FULL / "paired_source_metrics.csv")
    phases = pd.read_csv(FULL / "paired_phase_metrics.csv")
    matched = pd.read_csv(FULL / "paired_matched_phase_metrics.csv")
    counts = pd.read_csv(DATA / "manual_yolo_summary.csv")
    audit = json.loads((FULL / "comparison_audit.json").read_text())
    assert audit["models"] == 12 and audit["same_sample_ids"] and audit["same_yield_targets"]
    assert set(paired.model) == set(METHODS)

    add_para("1. Data and measurement", True)
    add_para("The analysis uses the 2324 and 2526 seasons. Plant roundness was recalculated as Roundnessᵢ = 4πAᵢ/Pᵢ², where Aᵢ is plant mask area and Pᵢ is the complete contour perimeter, including holes. Plot roundness is the mean across retained plants; canopy size is the sum of their mask areas. The source processing contains 13,353 plant records and 862 plot-dates.")
    add_para("The 2324 YOLO counts came from image-derived detection summaries. The 2526 FL, G, W and P counts were rerun from the original RGB images with the five-class YOLO detector and linked to tracked plant boxes; all 504 available plot-dates completed. Manual counts came from the 2324 field CSV files and the 2526 check-in workbook. FL denotes flower, G green, W white, P pink and R mature red fruit. R is reported descriptively and excluded from every prediction model because it is already mature fruit.")
    add_para("A whole visit date was excluded when fewer than 32 of 40 plots had complete image, manual and YOLO features, or when retained tracks exceeded twice the planted number. Remaining incomplete plot rows were removed. The matched dataset contains 318 plot-dates across eight 2324 dates and 439 across eleven 2526 dates. The 2526 January 12 visit had only 24 image plots; February 23 had a tracking anomaly. Three blank-count visits and two later visits lacked matching images.")
    add_table([["Season", "Matched plot-dates", "Visit dates", "Median retained plants"],
               ["2324", "318", "8", "20"], ["2526", "439", "11", "9"]])
    add_para("Note. A plot-date is one plot on one image visit. Retained plants are tracked masks used for canopy measurement and quality control; they are not prediction inputs.", note=True)
    add_para("Conclusion: Both counting sources and image features are available on the same 757 retained plot-dates. No blank count or yield was replaced with a numerical value.", conclusion=True)

    add_para("2. Feature distributions and count agreement", True)
    add_para("Mean canopy size is 1.76 million pixels per plot in 2324 and 0.15 million in 2526. Planted plants declined from about 20 to 10 per plot, and median retained tracks declined from 20 to 9. Mean mask area per retained track was also about five times larger in 2324. Plant number explains part of the canopy difference; image scale and segmentation also contribute. Pixel area is not a calibrated physical area.")
    add_picture("verified_image_yield_distributions.png")
    add_para("Note. Boxplots display observed distributions; canopy uses millions of image pixels per plot, roundness is dimensionless, plant tracks are counts per plot, and yield is grams per plot per harvest. The boxes are descriptive, not prediction errors.", note=True)
    harvest = pd.read_csv(DATA.parent / "yields.csv", dtype={"season": str})
    harvest_rows = [["Season", "Harvest records", "Mean yield (g/plot)", "Median yield (g/plot)"]]
    for season in ("2324", "2526"):
        values = harvest.loc[harvest.season.eq(season), "yield_g"]
        harvest_rows.append([season, str(len(values)), f"{values.mean():.1f}", f"{values.median():.1f}"])
    add_table(harvest_rows)
    add_para("Note. Each harvest record is one plot on one harvest date; these values are per-plot harvest yield, not total seasonal production or yield per plant.", note=True)
    mean_2324 = harvest.loc[harvest.season.eq("2324"), "yield_g"].mean()
    mean_2526 = harvest.loc[harvest.season.eq("2526"), "yield_g"].mean()
    assert mean_2324 > mean_2526
    add_para(f"Observed harvest yield per plot was higher in 2324 than 2526 ({mean_2324:.1f} vs {mean_2526:.1f} g/plot on average). The 2526 collection represented about half as many plants per plot, approximately 10 rather than 20, which is an important design reason for the lower raw plot yield. The canopy pixel difference is larger than the plant-number change alone and also reflects image scale and segmentation.", conclusion=True, highlight=True)
    add_para("YOLO counted fewer FL, G, W and P than field counting on the matched plot-dates in both seasons. R was closer to manual counts: Pearson r was 0.73 in 2324 and 0.81 in 2526. Observed zero counts were retained as zeros.")
    add_picture("manual_yolo_distributions.png")
    add_para("Note. Each boxplot uses counts per plot-date. Manual is field counting and YOLO is image-derived counting; R denotes mature red fruit and is descriptive only.", note=True)
    rows = [["Season", "Feature", "Manual mean", "YOLO mean", "Pearson r"]]
    for season in ("2324", "2526"):
        for code in ("FL", "G", "W", "P", "R"):
            rec = counts[(counts.season.astype(str) == season) & counts.feature.eq(code)].iloc[0]
            rows.append([season, code, f"{rec.manual_mean:.1f}", f"{rec.yolo_mean:.1f}", f"{rec.pearson_r:.2f}"])
    add_table(rows)
    add_para("Note. Manual and YOLO means are counts per plot-date; Pearson r measures agreement across matched plot-dates, not correlation with yield.", note=True)
    add_para("Conclusion: Manual and YOLO counts are related but differ in scale, particularly for FL, G, W and P. Their forecasting results therefore require a matched comparison.", conclusion=True, highlight=True)

    add_para("3. Association with observed harvest yield", True)
    add_para("Image features were paired with the observed harvest nearest 7, 14 or 21 days later (±2 days). At one week, canopy–yield Pearson r was 0.325 in 2324 and 0.139 in 2526; roundness–yield r was −0.070 and −0.392. Manual W–yield r was 0.274 and 0.425, while YOLO W–yield r was 0.424 and 0.325. The figure shows only eligible prediction features at one week. Separate descriptive tables retain all recorded variables and three horizons.")
    add_picture("verified_feature_yield_correlations.png")
    add_para("Note. Bar height is the one-week Pearson correlation with observed harvest yield; M and Y identify manual and YOLO count sources. R and retained plant tracks are not shown because they are not prediction features.", note=True)
    add_para("Conclusion: Association strength and direction vary between seasons and counting sources. These correlations do not establish causation or guarantee forecast accuracy.", conclusion=True)

    add_para("4. Prediction design and overall results", True)
    add_para("The only prediction features are FL, G, W, P, canopy size and roundness. Retained plant tracks are used for image quality control and are not prediction inputs. The six input sets are canopy, roundness, FL/G/W/P, canopy with roundness, canopy with FL/G/W/P, and all six features. Twelve fixed predictive models cover linear and nonlinear regression, tree models, support vector regression and three neural sequence models. Current-input models use the latest image date. Temporal models use three available observations. Each result predicts one observed harvest yield in grams per plot, not total seasonal yield.")
    add_para("Manual and YOLO runs use identical plot-dates, harvest labels, test IDs and model settings. Cross-season results are reported in both directions. Within-season models are refitted chronologically using only harvest labels available before the forecast date; two earlier completed harvests are required. The 2526 → 2324 transfer is retrospective. Lower MAE is better. All rankings are descriptive because the test results have been inspected.")
    add_para("The fixed random-forest comparison below isolates the six input choices at one week. Canopy-only and roundness-only values are identical between manual and YOLO, and their errors match exactly.")
    add_picture("full_models/feature_set_comparison_h1.png")
    rf = paired[(paired.model == "Random forest") & (paired.horizon_weeks == 1)]
    for scenario, label in SCENARIOS:
        block = rf[rf.scenario.eq(scenario)]
        add_para(f"{label}: the lowest random-forest MAE among six input sets was {block.loc[block.MAE_manual.idxmin(), 'feature_set']} with manual counts ({block.MAE_manual.min():.1f} g/plot), and {block.loc[block.MAE_yolo.idxmin(), 'feature_set']} with YOLO counts ({block.MAE_yolo.min():.1f} g/plot).", conclusion=True)
    example = paired[(paired.feature_set == "All features") & paired.scenario.eq("2324_to_2526") &
                     paired.horizon_weeks.eq(1) & paired.model.eq("Random forest")].iloc[0]
    add_para("All 12 models are compared on the complete feature set in the following figure and tables.")
    add_picture("full_models/model_comparison_h1.png")
    add_para(f"Note. Each paired value is manual / YOLO MAE in g/plot, with lower values indicating smaller prediction errors. For example, {fmt_pair(example)} for the one-week random forest means {example.MAE_manual:.1f} using manual FL/G/W/P and {example.MAE_yolo:.1f} using YOLO FL/G/W/P. A dash means no eligible forecast.", note=True)
    for horizon in (1, 2, 3):
        block = paired[(paired.feature_set == "All features") & paired.horizon_weeks.eq(horizon)]
        add_para(f"{horizon}-week overall prediction: manual / YOLO MAE (g/plot)")
        matrix = [["Model", *(label for _, label in SCENARIOS)]]
        for model in METHODS:
            row = [model]
            for scenario, _ in SCENARIOS:
                sub = block[block.model.eq(model) & block.scenario.eq(scenario)]
                row.append(fmt_pair(sub.iloc[0]) if len(sub) else "—")
            matrix.append(row)
        add_table(matrix)
        for scenario, label in SCENARIOS:
            sub = block[block.scenario.eq(scenario)]
            if len(sub):
                better_m = int((sub.MAE_manual < sub.MAE_yolo).sum())
                better_y = int((sub.MAE_yolo < sub.MAE_manual).sum())
                add_para(f"{label}, {horizon} week(s): lowest manual MAE was {winner(sub, 'manual')}; lowest YOLO MAE was {winner(sub, 'yolo')}. Manual had lower MAE for {better_m} of 12 models and YOLO for {better_y}.", conclusion=True)
            else:
                add_para(f"{label}, {horizon} weeks: no chronological forecast met the two-prior-harvest requirement.", conclusion=True)
    sensitive = paired[paired.feature_set.isin(("Counts", "Canopy + counts", "All features"))]
    manual_lower = int((sensitive.MAE_manual < sensitive.MAE_yolo).sum())
    yolo_lower = int((sensitive.MAE_yolo < sensitive.MAE_manual).sum())
    add_para(f"For the {len(sensitive)} overall model comparisons containing FL/G/W/P, manual had lower MAE in {manual_lower} and YOLO in {yolo_lower}. These are repeated evaluations on shared plots and harvest dates, not independent trials.", conclusion=True)
    add_para("A fixed random forest illustrates the change with forecast lead time; its results do not represent every model.")
    add_picture("full_models/horizon_comparison.png")

    add_para("5. Early, middle and late harvest periods", True)
    add_para("Early, middle and late refer to ordered thirds of observed harvest dates within each season. These are calendar groups, not physiological stages. A phase with one harvest date has limited evidence. Tables below score each overall-fitted model separately by harvest period.")
    add_para("Note. M/Y means manual MAE first and YOLO MAE second, both in g/plot. Overall is the pooled-model result; Early, Middle and Late score that same model on the indicated harvest periods. A dash means no eligible chronological forecast.", note=True)
    for horizon in (1, 2, 3):
        for scenario, label in SCENARIOS:
            pooled = paired[(paired.feature_set == "All features") & paired.horizon_weeks.eq(horizon) & paired.scenario.eq(scenario)]
            if pooled.empty:
                continue
            block = phases[(phases.feature_set == "All features") & phases.horizon_weeks.eq(horizon) & phases.scenario.eq(scenario)]
            add_para(f"{label}, {horizon} week(s): overall and harvest-period MAE")
            matrix = [["Model", "Overall M/Y", "Early M/Y", "Middle M/Y", "Late M/Y"]]
            for model in METHODS:
                row = [model, fmt_pair(pooled[pooled.model.eq(model)].iloc[0])]
                for stage in STAGES:
                    sub = block[block.model.eq(model) & block.stage.eq(stage)]
                    row.append(fmt_pair(sub.iloc[0]) if len(sub) else "—")
                matrix.append(row)
            add_table(matrix)
            claims = []
            for stage in STAGES:
                sub = block[block.stage.eq(stage)]
                if len(sub):
                    n = int(sub.n_manual.iloc[0])
                    claims.append(f"{stage} (n={n}): manual {winner(sub,'manual')}, YOLO {winner(sub,'yolo')}")
            add_para(f"{label}, {horizon} week(s): lowest observed phase MAEs were " + "; ".join(claims) + ". These phase winners are descriptive.", conclusion=True)

    add_para("6. Phase-specific training", True)
    add_para("A separate experiment fitted transfer models using only training samples from the same harvest period. The comparison with overall-fitted models uses exactly the same test plots for each period. Within-season phase-specific fits had too few earlier completed harvest dates and are therefore unavailable.")
    add_para("Note. In each cell, M = manual and Y = YOLO. For each source, the value before / is the phase-specific model MAE and the value after / is the overall-fitted model MAE, both in g/plot. A dash means that no phase-specific fit was available.", note=True)
    for scenario, label in SCENARIOS[:2]:
        for horizon in (1, 2, 3):
            pooled = matched[(matched.feature_set == "All features") & matched.horizon_weeks.eq(horizon) & matched.scenario.eq(scenario)]
            stagefit = matched[(matched.feature_set == "All features") & matched.horizon_weeks.eq(horizon) & matched.scenario.eq("stage_" + scenario)]
            joined = pooled.merge(stagefit, on=["feature_set", "horizon_weeks", "stage", "model"], suffixes=("_overall", "_phase"), validate="one_to_one")
            if joined.empty:
                continue
            add_para(f"{label}, {horizon} week(s): phase-fitted / overall-fitted MAE on matched test plots")
            matrix = [["Model", "Early phase/overall", "Middle phase/overall", "Late phase/overall"]]
            for model in METHODS:
                row = [model]
                for stage in STAGES:
                    sub = joined[joined.model.eq(model) & joined.stage.eq(stage)]
                    if sub.empty:
                        row.append("—")
                    else:
                        z = sub.iloc[0]
                        row.append(f"M {z.MAE_manual_phase:.0f}/{z.MAE_manual_overall:.0f}; Y {z.MAE_yolo_phase:.0f}/{z.MAE_yolo_overall:.0f}")
                matrix.append(row)
            add_table(matrix)
            statements = []
            for source in ("manual", "yolo"):
                lower = int((joined[f"MAE_{source}_phase"] < joined[f"MAE_{source}_overall"]).sum())
                statements.append(f"{source} {lower}/{len(joined)}")
            add_para(f"{label}, {horizon} week(s): phase-specific fitting had lower MAE in " + ", ".join(statements) + " matched model-period comparisons. The result varies by period and model.", conclusion=True)

    add_para("7. Model selection and interpretation", True)
    selection = paired.copy()
    for source in ("manual", "yolo"):
        selection[f"rank_{source}"] = selection.groupby(["scenario", "horizon_weeks"])[f"MAE_{source}"].rank(method="average")
    rank_summary = selection.groupby(["model", "feature_set"])[["rank_manual", "rank_yolo"]].mean()
    rank_summary["mean_rank"] = rank_summary[["rank_manual", "rank_yolo"]].mean(axis=1)
    rank_summary = rank_summary.sort_values("mean_rank")
    selected, selected_features = rank_summary.index[0]
    n_settings = selection[["scenario", "horizon_weeks"]].drop_duplicates().shape[0]
    assert n_settings == 11 and (selected, selected_features) == ("Lag random forest", "All features")
    selected_rank = rank_summary.loc[(selected, selected_features), "mean_rank"]
    add_para(f"Selected model: {selected} with all six features. It had the lowest average MAE rank among the 72 model–input combinations across {n_settings} available setting–horizon combinations and both count sources ({selected_rank:.2f} out of 72). Each setting contributes one manual rank and one YOLO rank. This is a descriptive selection from the present data; the model did not have the lowest error in every individual evaluation.", conclusion=True, highlight=True)
    selected_rows = [["One-week evaluation", "Manual MAE", "YOLO MAE", "Matched plots"]]
    for scenario, label in SCENARIOS:
        row = selection[(selection.scenario.eq(scenario)) & selection.horizon_weeks.eq(1) & selection.model.eq(selected) & selection.feature_set.eq(selected_features)].iloc[0]
        selected_rows.append([label, f"{row.MAE_manual:.1f}", f"{row.MAE_yolo:.1f}", str(int(row.n_manual))])
    add_table(selected_rows)
    add_para("Note. Rank 1 denotes the lowest MAE among 72 model–input combinations in one setting. The table shows the selected model with all six features at a one-week lead time; each MAE is in g/plot, and manual and YOLO use the same matched test plots in each row.", note=True)
    set_text(original[24], f"Across the available setting–horizon comparisons, {selected} with all six features had the lowest average MAE rank among 72 model–input combinations ({selected_rank:.2f} out of 72). Individual results vary by evaluation setting; the full comparison follows.")
    add_para("The earlier PheMuT system combined two YOLO detectors, ByteTrack, SAM, depth, weather encoding, stage weighting and an LSTM. The present prediction step uses FL, G, W, P, canopy and roundness. Canopy pixel scale differs substantially between seasons, and manual and YOLO counts disagree. The small number of distinct harvest dates limits the precision of model comparisons.")

    body.append(deepcopy(original[-1]))
    assert ";" not in "".join(document.xpath(".//w:t/text()", namespaces=NS))
    parts["word/document.xml"] = etree.tostring(document, encoding="UTF-8", xml_declaration=True, standalone=True)
    rels = etree.fromstring(parts["word/_rels/document.xml.rels"])
    for relation in list(rels):
        if relation.get("Id", "").startswith("rId10"):
            rels.remove(relation)
    for rid, name in image_rels:
        etree.SubElement(rels, f"{{{P}}}Relationship", Id=rid,
                         Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                         Target=f"media/{name}")
    parts["word/_rels/document.xml.rels"] = etree.tostring(rels, encoding="UTF-8", xml_declaration=True, standalone=True)
    temp = TARGET.with_suffix(".docx.tmp")
    with ZipFile(temp, "w", ZIP_DEFLATED) as archive:
        for name, contents in parts.items():
            archive.writestr(name, contents)
    with ZipFile(temp) as archive:
        assert archive.testzip() is None
    temp.replace(TARGET)
    print(json.dumps({"report": str(TARGET), "models": 12, "sections": 7, "pictures": len(image_rels)}))


if __name__ == "__main__":
    main()
