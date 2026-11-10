#!/usr/bin/env python3
"""Create the 27 October weekly report by cloning the supplied Word template.

The original file is never modified. Styles, numbering, page setup, headers,
footers and logo are copied byte-for-byte. Results are read from actual runs.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json

from lxml import etree
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "Jing-2026-1013- dissertation.docx"
DEST = ROOT / "Jing-2026-1027- dissertation.docx"
DATA = ROOT / "canopy_temporal_benchmark/artifacts"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"w": W, "r": R, "wp": WP, "a": A}


def q(name): return f"{{{W}}}{name}"


def strip_ids(node):
    for item in node.iter():
        for key in list(item.attrib):
            if key.endswith("}paraId") or key.endswith("}textId"):
                del item.attrib[key]
    return node


def set_text(paragraph, text):
    run_properties = paragraph.find("w:r/w:rPr", NS)
    run_properties = deepcopy(run_properties) if run_properties is not None else None
    for child in list(paragraph):
        if child.tag != q("pPr"): paragraph.remove(child)
    run = etree.SubElement(paragraph, q("r"))
    if run_properties is not None: run.append(run_properties)
    element = etree.SubElement(run, q("t")); element.text = str(text)
    element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return paragraph


def property_node(paragraph):
    prop = paragraph.find("w:pPr", NS)
    if prop is None: prop = etree.Element(q("pPr")); paragraph.insert(0, prop)
    return prop


def flag(prop, name, enabled=True):
    old = prop.find(q(name))
    if old is not None: prop.remove(old)
    if enabled: etree.SubElement(prop, q(name))


class TemplateWriter:
    def __init__(self):
        with ZipFile(TEMPLATE) as archive:
            self.parts = {entry.filename: archive.read(entry.filename) for entry in archive.infolist()}
        self.root = etree.fromstring(self.parts["word/document.xml"])
        self.body = self.root.find("w:body", NS)
        self.original = [deepcopy(e) for e in self.body]
        for child in list(self.body): self.body.remove(child)
        self.nodes = []
        self.next_picture = 100

    def paragraph(self, text, kind="body", new_page=False):
        index = {"body": 21, "heading": 23, "bullet": 8,
                 "label": 12, "title": 19, "caption": 30}[kind]
        p = strip_ids(deepcopy(self.original[index])); set_text(p, text)
        prop = property_node(p)
        if kind in {"heading", "label", "title"}: flag(prop, "keepNext")
        if new_page: flag(prop, "pageBreakBefore")
        self.nodes.append(p)
        return p

    def blank(self):
        p = deepcopy(self.original[18]); self.nodes.append(p)

    def cover(self):
        self.nodes.extend(deepcopy(self.original[:7]))
        cells = self.nodes[4].xpath(".//w:tc", namespaces=NS)
        set_text(cells[3].find("w:p", NS), "From: 10.14.2026    To: 10.27.2026")
        set_text(cells[7].find("w:p", NS), "10.27.2026")

    def table(self, headers, rows, widths=None):
        n = len(headers)
        widths = widths or ([2280]+[1180]*6 if n == 7 else [9360//n]*n)
        widths[-1] += 9360-sum(widths)
        source = self.original[25]
        table = etree.Element(q("tbl")); table.append(deepcopy(source.find("w:tblPr", NS)))
        table.find("w:tblPr/w:tblW", NS).set(q("w"), "9360")
        grid = etree.SubElement(table, q("tblGrid"))
        for width in widths: etree.SubElement(grid, q("gridCol"), {q("w"): str(width)})
        template_rows = source.findall("w:tr", NS)
        for ri, values in enumerate([headers]+rows):
            tr = etree.SubElement(table, q("tr")); trpr = etree.SubElement(tr, q("trPr"))
            etree.SubElement(trpr, q("cantSplit"))
            if ri == 0: etree.SubElement(trpr, q("tblHeader"))
            originals = template_rows[0 if ri == 0 else 1].findall("w:tc", NS)
            for ci, (value, width) in enumerate(zip(values, widths)):
                tc = strip_ids(deepcopy(originals[min(ci, len(originals)-1)]))
                tc.find("w:tcPr/w:tcW", NS).set(q("w"), str(width))
                paras = tc.findall("w:p", NS)
                p = paras[0]
                for extra in paras[1:]: tc.remove(extra)
                set_text(p, value)
                prop = property_node(p)
                flag(prop, "keepNext", False)
                # More methods require denser rows; retain fonts, colors and
                # line spacing while reducing only after-paragraph table gaps.
                spacing = prop.find("w:spacing", NS)
                if spacing is None: spacing = etree.SubElement(prop, q("spacing"))
                spacing.set(q("after"), "40")
                # Preserve template typography and table styles, fit within margins.
                align = prop.find("w:jc", NS)
                if align is None: align = etree.SubElement(prop, q("jc"))
                align.set(q("val"), "left" if ci == 0 and ri else "center")
                tr.append(tc)
        self.nodes.append(table)

    def picture(self, path):
        p = strip_ids(deepcopy(self.original[29])); image_id = self.next_picture
        self.next_picture += 1
        rel_id = f"rId{image_id}"
        p.find(".//a:blip", NS).set(f"{{{R}}}embed", rel_id)
        for extent in p.xpath(".//wp:extent | .//a:xfrm/a:ext", namespaces=NS):
            extent.set("cx", "5943600"); extent.set("cy", "3017520")
        docpr = p.find(".//wp:docPr", NS); docpr.set("id", str(image_id)); docpr.set("name", path.stem)
        relns = "http://schemas.openxmlformats.org/package/2006/relationships"
        rels = etree.fromstring(self.parts["word/_rels/document.xml.rels"])
        filename = f"weekly_{image_id}.png"
        etree.SubElement(rels, f"{{{relns}}}Relationship", Id=rel_id,
            Type=f"{R}/image", Target=f"media/{filename}")
        self.parts["word/_rels/document.xml.rels"] = etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True)
        self.parts[f"word/media/{filename}"] = path.read_bytes()
        self.nodes.append(p)

    def save(self):
        for node in self.nodes: self.body.append(node)
        self.body.append(deepcopy(self.original[-1]))
        self.parts["word/document.xml"] = etree.tostring(self.root, xml_declaration=True, encoding="UTF-8", standalone=True)
        core = etree.fromstring(self.parts["docProps/core.xml"])
        creator = core.find("{http://purl.org/dc/elements/1.1/}creator")
        if creator is not None: creator.text = "Jing"
        modified = core.find("{http://purl.org/dc/terms/}modified")
        if modified is not None: modified.text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.parts["docProps/core.xml"] = etree.tostring(core, xml_declaration=True, encoding="UTF-8", standalone=True)
        app = etree.fromstring(self.parts["docProps/app.xml"])
        appns = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
        for name, value in {"Pages": 6, "Words": len(" ".join(self.root.xpath(".//w:t/text()", namespaces=NS)).split())}.items():
            item = app.find(f"{{{appns}}}{name}")
            if item is not None: item.text = str(value)
        self.parts["docProps/app.xml"] = etree.tostring(app, xml_declaration=True, encoding="UTF-8", standalone=True)
        with ZipFile(DEST, "w", ZIP_DEFLATED) as archive:
            for name, data in self.parts.items(): archive.writestr(name, data)
        return DEST


MODELS = ["Zero", "Train mean", "Train median", "OLS", "Quadratic OLS", "Spline ridge",
          "Random forest", "SVR", "Gradient boosting", "Lag OLS", "Lag ridge", "Lag random forest",
          "RNN", "LSTM", "Transformer"]


def main_table(writer, metrics, scenario):
    frame = metrics[metrics.scenario.eq(scenario)].set_index(["model", "horizon_weeks"])
    rows = []
    for model in MODELS:
        row = [model]
        for h in (1, 2, 3):
            entry = frame.loc[(model, h)]
            row += [f"{entry.MAE:.2f}", f"{entry.R2:.3f}"]
        rows.append(row)
    writer.table(["Model", "1w MAE", "1w R²", "2w MAE", "2w R²", "3w MAE", "3w R²"], rows)


def write_content(w, metrics, stages, seed_metrics, sensitivity):
    w.paragraph("Research activities last week", "heading")
    for text in [
        "Successfully defended my dissertation.",
        "Completed a canopy-only comparison of traditional statistical models, machine learning regression, and RNN, LSTM, and Transformer forecasting at 1-, 2-, and 3-week horizons.",
        "Completed a separate early-, middle-, and late-stage modeling branch and compared stage-specific models with models fitted across the full season.",
        "Continued processing all canopy datasets, including roundness extraction and season 2526. I expect initial results and analysis to be ready for the next report.",
    ]: w.paragraph(text, "bullet")
    w.paragraph("Problems encountered", "label")
    w.paragraph("Large yield differences between seasons and the small number of harvest dates limit model transfer and early-stage evaluation. Season 2526 processing is ongoing, and matching yield labels are not yet available for this benchmark.", "bullet")
    w.paragraph("Plan for next week", "label")
    for text in [
        "Meet with Dr. Kiker this week and with Dr. Li on Wednesday, October 28.",
        "Complete canopy and roundness processing, including season 2526, and analyze the updated results for the next report.",
        "Revisit all yield prediction models after the full dataset has been processed and checked.",
        "Complete post-defense dissertation revisions.",
        "Attend interviews with two companies next week (November 2–6).",
        "Bring Zekun into the yield project and prepare the data, code, analysis, and manuscript handover so he can continue the project and publication after I leave.",
    ]: w.paragraph(text, "bullet")

    # Page 2: the experimental design, preserving the original analysis heading.
    w.paragraph("Canopy-Only Yield Prediction", "title", new_page=True)
    w.paragraph("Regression, temporal models, and stage-specific prediction", "heading")
    w.paragraph("Data. I used the same final canopy-pipeline area measure as the previous report (canopy_size_pixels) and measured harvest yield in g/plot. Seasons 2324 and 2425 contain 8 and 9 harvest dates, respectively, with up to 40 plots per date. Roundness, fruit counts, weather, plant counts, and previous yield values were excluded from model inputs. Season 2526 was excluded because matching yield labels are unavailable.")
    w.paragraph("Inputs and targets. Snapshot regression uses canopy at the forecast origin. Temporal methods use the three most recent canopy observations for the same plot and season, ending at that origin. If fewer than three exist, the earliest observed value is repeated on the left. Observations are irregularly spaced: the median history span is 14 days and the maximum is 35 days. Yield is matched to 1, 2, or 3 calendar weeks later within ±2 days, without interpolating future data.")
    w.table(["Method group", "Models", "Canopy input"], [
        ["Benchmarks", "Zero, training mean, training median", "No canopy"],
        ["Traditional regression", "OLS, quadratic OLS, spline ridge", "Latest observation"],
        ["Machine learning regression", "Random forest, SVR, gradient boosting", "Latest observation"],
        ["Traditional temporal", "Distributed-lag OLS and ridge", "Three observations"],
        ["Machine learning temporal", "Lag random forest", "Three observations"],
        ["Deep temporal", "RNN, LSTM, Transformer", "Three observations"],
    ], widths=[2300,4560,2500])
    w.paragraph("Across-season transfer. Fit each horizon using season 2324 only, then predict 2425. Training counts are 278/239/199 and test counts are 319/239/278 at 1/2/3 weeks. Within-season forward update. Refit using only 2425 target yields harvested strictly before each forecast origin; begin after two distinct target dates are available. All methods share the same forecast pairs within each evaluation.")
    w.paragraph("Stage branch. Split each season’s ordered target harvest dates into fixed thirds, shared across horizons and models. Train a separate 2324 model for each target stage and forecast the corresponding 2425 stage. These are retrospective calendar phases, not physiological growth stages. Pooled-model predictions are also summarized on the same stage samples.")
    w.paragraph("Training and scoring. Settings were fixed without test-based tuning. Neural models use width 16, one recurrent/encoder layer, 80 AdamW epochs, and seeds 42/123/2026; the Transformer has two attention heads and sinusoidal positions. Main neural results are the mean of three seed forecasts. Scaling uses training data only, and all forecasts are clipped at zero. MAE is primary; R² is shown alongside it. Reproducible scripts, full RMSE/bias, seed results, parameters, and date audits are in canopy_temporal_benchmark/ and its artifacts/ folder.")

    # Page 3: all fifteen methods, with the limitations that affect interpretation.
    w.paragraph("Across-season transfer results", "heading", new_page=True)
    w.paragraph("Train on 2324; test on 2425. At 1/2/3 weeks, N = 319/239/278 across 8/6/7 target dates. MAE is in g/plot; R² is dimensionless. RNN, LSTM, and Transformer rows are three-seed ensembles.")
    main_table(w, metrics, "pooled_transfer")
    w.paragraph("The mean measured yield differs sharply between seasons: 264.63 g/plot in 2324 versus 24.24 g/plot in 2425. Several nonlinear regressors transfer poorly, so the present ranking should not be interpreted as general evidence that traditional methods outperform machine learning.")
    w.paragraph("Some low errors reflect forecasts close to zero. At 1 week, lag ridge predicts exactly zero for 318 of 319 plot-date forecasts; at 2 weeks, quadratic OLS, lag OLS, and lag ridge all reduce to the zero benchmark after clipping. These results do not demonstrate useful canopy-based discrimination.")
    seed = seed_metrics[(seed_metrics.scenario == "pooled_transfer") & (seed_metrics.horizon_weeks == 1) & (seed_metrics.model == "LSTM")]
    w.paragraph(f"The 1-week LSTM ensemble has R² = 0.051 and RMSE = 32.03 g/plot, compared with 40.41 for zero. Its MAE advantage over zero is only 0.01 g/plot. Individual-seed MAE ranges from {seed.MAE.min():.2f} to {seed.MAE.max():.2f}; every individual seed has negative R². The positive ensemble R² therefore requires cautious interpretation.")
    l = sensitivity[sensitivity.model.eq("LSTM")].sort_values("horizon_weeks")
    w.paragraph("History sensitivity. Requiring three real canopy observations spanning at most 18 days leaves 239/200/199 test pairs. With unchanged settings, LSTM MAE is " + "/".join(f"{v:.2f}" for v in l.MAE) + " g/plot at 1/2/3 weeks. Target coverage changes, so these values are not direct improvements over the full table. Original-model scores on the identical restricted samples are supplied separately.")

    # Page 4: chronological current-season forecasting.
    w.paragraph("Within-season forward results", "heading", new_page=True)
    w.paragraph("Update within 2425 using only completed harvests strictly before each prediction date. At 1/2/3 weeks, N = 200/80/80 across 5/2/2 target dates. The 2- and 3-week evaluations cover only late-stage yields on February 26 and March 5.")
    main_table(w, metrics, "pooled_forward")
    w.paragraph("At 1 week, random forest has the lowest learned-model MAE (26.97), with OLS (27.12) and LSTM (27.24) close behind. At 2 weeks, LSTM has the lowest MAE (26.69), followed by lag random forest (27.59); OLS is 30.53. This suggests that canopy history can help for this particular horizon and late-season subset.")
    w.paragraph("At 3 weeks, the training-mean benchmark has lower MAE (29.63) than every learned method. LSTM is 33.30 and Transformer is 32.41. All forward-test R² values remain negative. The results do not establish reliable yield prediction from canopy alone or a consistent advantage from temporal architecture.")
    seed_rows = []
    for model in ("RNN", "LSTM", "Transformer"):
        row = [model]
        for horizon in (1,2,3):
            vals = seed_metrics[(seed_metrics.scenario == "pooled_forward") & (seed_metrics.horizon_weeks == horizon) & (seed_metrics.model == model)].MAE
            row.append(f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}")
        seed_rows.append(row)
    w.paragraph("Individual-seed MAE mean ± SD (distinct from ensemble MAE above):")
    w.table(["Model", "1 week", "2 weeks", "3 weeks"], seed_rows, widths=[2280,2360,2360,2360])

    # Page 5: stage-specific training, distinct from merely stratifying a fit.
    w.paragraph("Early-, middle-, and late-stage prediction", "heading", new_page=True)
    w.paragraph("The 2324 target phases are January 8–23 (Early), January 30–February 13 (Middle), and February 20–27 (Late). The 2425 phases are January 7–24, January 29–February 12, and February 20–March 5. Each table entry below comes from a model trained only on the corresponding 2324 phase. MAE is in g/plot; N is shown with the stage/horizon.")
    selected = ["OLS", "Lag ridge", "Lag random forest", "RNN", "LSTM", "Transformer"]
    stage_rows = []
    for stage in ("Early", "Middle", "Late"):
        for horizon in (1,2,3):
            sub = stages[(stages.scenario == "stage_transfer") & (stages.stage == stage) & (stages.horizon_weeks == horizon)]
            if sub.empty:
                n = 39 if stage == "Early" and horizon == 3 else 0
                stage_rows.append([f"{stage} {horizon}w ({n})"] + ["NA"]*6)
            else:
                sub = sub.set_index("model")
                stage_rows.append([f"{stage} {horizon}w ({int(sub.iloc[0]['n'])})"] + [f"{sub.loc[m, 'MAE']:.2f}" for m in selected])
    w.table(["Stage / horizon (N)", "OLS", "Lag ridge", "Lag RF", "RNN", "LSTM", "Transf."], stage_rows, widths=[2160,1200,1200,1200,1200,1200,1200])
    w.paragraph("NA has a data-specific meaning: Early/2w has no eligible 2425 targets under the ±2-day matching rule; Early/3w has 39 test pairs but no corresponding 2324 training pairs. Pure 2425 stage-specific forward fitting is infeasible at every future horizon because two completed same-stage target dates are not yet available before any eligible origin.")
    matched_rows = []
    predictions = pd.read_csv(DATA / "predictions.csv")
    matched_metrics = []
    for horizon in (1,2,3):
        specific = predictions[(predictions.scenario == "stage_transfer") & (predictions.horizon_weeks == horizon) & (predictions.model == "Lag ridge")]
        pooled = predictions[(predictions.scenario == "pooled_transfer") & (predictions.horizon_weeks == horizon) & (predictions.model == "Lag ridge") & predictions.sample_id.isin(set(specific.sample_id))]
        assert set(specific.sample_id) == set(pooled.sample_id)
        v1 = np.mean(np.abs(pooled.prediction-pooled.yield_g)); v2 = np.mean(np.abs(specific.prediction-specific.yield_g))
        matched_rows.append([f"{horizon} week(s)", str(len(specific)), f"{v1:.2f}", f"{v2:.2f}"])
        matched_metrics.append(dict(horizon_weeks=horizon,n=len(specific),pooled_MAE=v1,stage_MAE=v2))
    pd.DataFrame(matched_metrics).to_csv(DATA / "report_stage_matched_lag_ridge.csv", index=False)
    w.paragraph("Lag ridge: pooled versus stage-specific training on identical test pairs")
    w.table(["Horizon", "Matched N", "Pooled MAE", "Stage MAE"], matched_rows, widths=[2280,2360,2360,2360])
    w.paragraph("Stage-specific lag ridge reduces MAE at all three horizons. Its R² is 0.105 at 1 week and 0.218 at 3 weeks; the latter excludes the unavailable early stage. Stage-specific lag OLS has the lowest 3-week MAE (19.84), while lag ridge has lower RMSE (31.55 versus 33.47). In contrast, stage-specific LSTM performs worse than pooled LSTM at every horizon on matched samples. Smaller stage datasets do not support a general benefit from fitting more complex models.")
    w.paragraph("All 15 methods were fitted in every available stage/horizon combination. The full stage tables, including benchmarks, quadratic/spline regressions, SVR, random forest, gradient boosting, RMSE, bias, and R², are retained in stage_metrics.csv.")

    # Page 6: visual comparison and practical implications.
    w.paragraph("Model comparison and next analysis", "heading", new_page=True)
    w.picture(DATA / "main_model_comparison.png")
    w.paragraph("Figure 1. MAE of selected methods at 1–3 weeks. Transfer and forward evaluation use different eligible samples and panel scales; compare methods within each panel.", "caption")
    w.picture(DATA / "stage_model_comparison.png")
    w.paragraph("Figure 2. One-week pooled versus stage-specific transfer on identical Early/Middle/Late samples (N = 80/119/120). Lag ridge improves mainly in Late; neural gains vary by stage.", "caption")
    w.paragraph("The next analysis will repeat the comparison after all canopy data, roundness, and season 2526 have been processed and checked. The current evidence supports keeping simple statistical and lagged-canopy baselines alongside neural models, with particular attention to season shifts, limited target dates, and stage coverage.")


def main():
    metrics = pd.read_csv(DATA / "metrics.csv")
    stages = pd.read_csv(DATA / "stage_metrics.csv")
    seed_metrics = pd.read_csv(DATA / "neural_seed_metrics.csv")
    sensitivity = pd.read_csv(DATA / "sensitivity_metrics.csv")
    writer = TemplateWriter()
    writer.cover()
    write_content(writer, metrics, stages, seed_metrics, sensitivity)
    destination = writer.save()
    # Byte-for-byte preservation of the visual template foundations.
    with ZipFile(TEMPLATE) as old, ZipFile(destination) as new:
        for name in ["word/styles.xml", "word/numbering.xml", "word/header1.xml", "word/footer1.xml", "word/footer2.xml", "word/media/image2.png"]:
            assert old.read(name) == new.read(name), name
        for name in new.namelist():
            if name.endswith(".xml"): etree.fromstring(new.read(name))
    print(destination)


if __name__ == "__main__": main()
