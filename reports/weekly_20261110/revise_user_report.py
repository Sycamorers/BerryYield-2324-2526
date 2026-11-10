#!/usr/bin/env python3
"""Revise the user-edited weekly report without rebuilding its approved opening."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCE = ROOT / 'Jing-2026-1110-yield prediction new data.docx'
DEST = ROOT / 'Jing-2026-1110-yield prediction new data-revised.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}

# Body element numbers in the supplied file. Everything through 24 is preserved.
REWRITE = {
    25: '1. Reprocess data and image features',
    26: 'Recalculate roundness from each retained plant mask. Aggregate canopy, roundness and plant tracks by plot and date: 13,353 plant records yield 862 plot-dates. Check the aggregates against the source table; maximum roundness difference is below 2 × 10⁻¹⁶.',
    30: 'Measure the three image features',
    32: '2. Feature distributions by season',
    36: 'Compare the two seasons. Mean canopy pixel area is 9.35 times higher in 2324; roundness is higher in 2526. Check the last 2526 scan: its median is 46.5 tracks per plot versus about 6–11 on other dates. Treat camera scale and tracking as possible sources of this shift.',
    39: 'Feature distributions',
    40: 'Compare canopy, roundness, plant tracks, FL, G, W, P, R and yield by season. Read each histogram line as the median. Use FL for flower, G for green, W for white, P for pink and R for red. Keep R descriptive only. Do not treat these separate source distributions as matched predictor–yield pairs.',
    45: 'Distribution summary',
    47: 'Calculate each plant’s roundness as 4πA/P² and average across retained plants. Use medians and IQRs to compare the skewed features and yield. Keep valid zeros separate from missing values.',
    48: 'Use date-matched observations in the correlation analysis below. Do not infer an association from marginal distributions alone.',
    49: '3. Feature–yield correlations',
    50: 'Compare each feature at the image date with yield 1, 2 and 3 weeks later. Read r as Pearson correlation, n as matched plot observations and d as harvest dates. Treat repeated plots and the small number of dates as limits on interpretation.',
    53: 'At one week, compare canopy and roundness: r=0.325 and −0.070 in 2324; r=0.113 and −0.339 in 2526. Compare W and P: r=0.274 and 0.374 in 2324; r=0.447 and 0.437 in 2526.',
    54: 'Center observations within each harvest date. In 2526, canopy r becomes 0.305 and roundness r becomes −0.196. Read Spearman and date-centered results in correlations.csv; do not infer causation.',
    55: 'Check correlations by harvest phase',
    58: 'For 2526 canopy, compare r=0.384 early, 0.515 middle and −0.044 late. For W, compare r=0.591, 0.258 and 0.543. Report phase-specific model errors as well as pooled errors.',
    59: 'Treat phase correlations as descriptive: each phase has only two or three matched harvest dates. Do not select features from evaluation-season correlations.',
    60: '4. Prediction task and comparison',
    61: 'Predict g/plot at the observed harvest nearest to 7, 14 or 21 days after each image date (±2 days). Require a future harvest; do not interpolate yield or turn blanks into zero. Compare models on identical complete-input samples. For within-season fits, require two earlier completed harvest dates.',
    65: 'Give current-input models the latest features. Give history models the last three observations per plot, oldest first; repeat the earliest value when needed. Fit scaling on training data only. Keep dates, plot IDs, phase and future yield out of predictors. Use fixed settings; average neural forecasts from seeds 42, 123 and 2026 after 80 epochs (hidden width 16; AdamW learning rate 0.003).',
    66: 'Use 318 complete plot-dates in 2324 and 504 in 2526. Exclude 40 image-only 2324 rows lacking FL/G/W/P. Use 320 and 480 observed yields, respectively. Compare canopy, roundness, FL/G/W/P + tracks, canopy + roundness, canopy + FL/G/W/P + tracks, and all features. Exclude R and detector-derived fruit values from predictors; clip negative forecasts to zero.',
    68: 'Read all-feature errors in g/plot per matched harvest. Use canopy, roundness, FL/G/W/P and plant tracks. Compare each horizon with its own training-mean reference.',
    69: 'Use 304/304/320 test observations from eight harvest dates at 1/2/3 weeks.',
    71: 'Compare the lowest observed learned-model MAE: LSTM 80.0 g (R²=0.03) at 1 week; RNN 81.7 g (0.02) at 2 weeks; LSTM 80.8 g (−0.02) at 3 weeks. Treat these rankings as descriptive.',
    75: 'Read all-feature errors in g/plot per matched harvest. Compare each horizon with its own training-mean reference.',
    76: 'Use 278/239/199 test observations from 7/6/5 harvest dates at 1/2/3 weeks.',
    78: 'Compare the lowest observed learned-model MAE: LSTM 140.1 g (R²=0.19) at 1 week; LSTM 154.2 g (0.04) at 2 weeks; gradient boosting 148.9 g (−0.17) at 3 weeks. Treat these rankings as descriptive.',
    82: 'Read all-feature errors in g/plot per matched harvest. Compare each horizon with its own training-mean reference.',
    83: 'Use 159/80 test observations from 4/2 harvest dates at 1/2 weeks. Mark 3 weeks unavailable.',
    85: 'Compare the lowest observed learned-model MAE: LSTM 185.8 g (R²=−0.90) at 1 week; random forest 107.1 g (0.20) at 2 weeks. Treat these rankings as descriptive.',
    88: 'Mark 2324 within-season 3-week accuracy unavailable: no origin has two earlier completed training harvest dates.',
    90: 'Read all-feature errors in g/plot per matched harvest. Compare each horizon with its own training-mean reference.',
    91: 'Use 184/184/200 test observations from five harvest dates at 1/2/3 weeks.',
    93: 'Compare the lowest observed learned-model MAE: Transformer 75.4 g (R²=0.17) at 1 week; LSTM 79.9 g (0.08) at 2 weeks; LSTM 78.4 g (0.01) at 3 weeks. Treat these rankings as descriptive.',
    96: 'Check forecasts against observed harvests',
    97: 'Compare observed and predicted mean g/plot by harvest date at one week. Use the lowest-MAE learned model in each panel for illustration only; it was selected after viewing these test results.',
    101: 'Read bias as predicted minus observed yield. Check MAE, RMSE and R² together; a low MAE can still accompany a negative R² or missed yield peaks.',
    102: 'Compare feature sets',
    103: 'Hold test samples, settings and protocol fixed. Compare one-week MAE across canopy, roundness, FL/G/W/P + tracks and their combinations. Read each panel on its own scale.',
    107: 'Add roundness to canopy + FL/G/W/P + tracks. Count lower-MAE comparisons among learned models and horizons: 22/36 for 2324→2526, 21/36 for 2526→2324, 9/24 within 2324 and 23/36 within 2526. Treat these as descriptive comparisons.',
    108: 'Compare early, middle and late harvests across seasons',
    109: 'Score each full-season model within each harvest phase. Compare separate phase fits on the same test samples; use the target phase only as a retrospective grouping.',
    115: 'Read 23→25 as 2324→2526 and 25→23 as 2526→2324. Compare pool (one full-season model) with fit (one phase model) on identical test IDs. Treat any phase gain as exploratory.',
    116: 'Compare early, middle and late harvests within seasons',
    122: 'Require two earlier completed harvest dates in the same phase before a phase-forward fit. Mark unsupported results unavailable; do not use future labels.',
    123: 'Main results and next steps',
    124: 'Compare 15 methods, six feature sets, three horizons and four evaluation settings. Note that 110 of 132 available all-feature learned-model results have negative R². Check simple references and measurement shifts before choosing a model.',
    125: 'Check the completed analysis',
    126: 'Use the audited 13,353 plant records, 862 plot-dates and matched yields. Check the independent review (25 passed checks) and saved-model reload (147,960 predictions reproduced within 1.82 × 10⁻¹²). Keep the same test IDs within each comparison.',
    127: 'Interpret season differences',
    128: 'Account for camera scale and tracking differences when comparing seasons. Keep the recorded observations in evaluation; do not remove difficult test rows.',
    129: 'Treat model rankings as exploratory because the test results were inspected. Use neural seeds to assess training variation and target-date bootstrap intervals to describe uncertainty across the limited observed dates. Collect independent dates or seasons before deployment claims.',
    130: 'Compare shared test samples',
    132: 'Compare transfer and within-season results only on the shared test IDs in this table. Do not compare broader protocol tables as if they used the same observations.',
    133: 'Use the recorded harvest phases',
    135: 'Reproduce and hand over',
    136: 'Use the delivered Word/PDF, source data, figures, sample IDs, predictions, metrics, fit audits, source hashes, saved models and scripts. Follow reports/weekly_20261110/README.txt. Exclude 2425 throughout.',
}

REMOVE_CAPTIONS = {29, 34, 38, 42, 44, 52, 57, 63, 73, 80, 87, 95, 99, 105, 111, 118}
IMAGES = {
    'image2.png': BASE/'figures/image_to_trait_workflow.png',
    'image3.png': BASE/'pipeline/traits_season_comparison.png',
    'image5.png': BASE/'correlations/image_yield_distributions.png',
    'image6.png': BASE/'correlations/field_count_distributions.png',
    'image7.png': BASE/'correlations/correlation_overview.png',
    'image8.png': BASE/'correlations/correlation_stages_h1.png',
    'image9.png': BASE/'figures/evaluation_design.png',
    'image15.png': BASE/'figures/feature_comparison.png',
}


def get_text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=NS))


def set_text(node, value):
    texts = node.xpath('.//w:t', namespaces=NS)
    assert texts, value
    texts[0].text = value
    for t in texts[1:]:
        t.text = ''


def revise_table(table):
    substitutions = {
        'Trait': 'Feature', 'Plant count': 'Plant tracks', 'Mean track count': 'Mean tracks',
        'Flower count': 'FL', 'Green count': 'G', 'White count': 'W',
        'Pink count': 'P', 'Red count': 'R', 'Counts': 'FL/G/W/P + tracks',
        'Canopy + counts': 'Canopy + FL/G/W/P + tracks', 'All traits': 'All features',
    }
    for cell in table.xpath('.//w:tc', namespaces=NS):
        old = get_text(cell)
        if old in substitutions:
            set_text(cell, substitutions[old])
    # Make the feature definition table use direct instructions.
    rows = table.xpath('./w:tr', namespaces=NS)
    if rows and get_text(rows[0]).startswith('FeatureDefinition'):
        for row, value in zip(rows[1:], [
            'Sum retained plant-mask pixel areas by plot; treat pixels as uncalibrated area.',
            'For each plant, calculate Rᵢ = 4πAᵢ/Pᵢ² (mask area Aᵢ; full contour perimeter Pᵢ, including holes). Average Rᵢ across retained plants for each plot-date.',
            'Count retained plant tracks; do not treat this as a verified plant census.',
        ]):
            set_text(row.xpath('./w:tc', namespaces=NS)[1], value)


def main():
    with ZipFile(SOURCE) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    root = etree.fromstring(parts['word/document.xml'])
    body = root.find(f'{{{W}}}body')
    nodes = list(body)
    assert get_text(nodes[7]) == 'Research activities last week'
    assert get_text(nodes[19]).startswith('Yield model development:')
    assert get_text(nodes[24]).startswith('One-week all-trait learned-model rankings:')
    approved = [etree.tostring(node) for node in nodes[:25]]
    for index, node in enumerate(nodes):
        if index <= 24:
            continue
        if index in REMOVE_CAPTIONS:
            body.remove(node)
        elif index in REWRITE:
            set_text(node, REWRITE[index])
        elif node.tag == f'{{{W}}}tbl':
            revise_table(node)
        elif index in {67, 74, 81, 89}:
            pass
        elif node.tag == f'{{{W}}}p' and get_text(node).strip():
            raise AssertionError(f'Unreviewed paragraph {index}: {get_text(node)[:90]}')
    assert approved == [etree.tostring(node) for node in list(body)[:25]]
    parts['word/document.xml'] = etree.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    for name, path in IMAGES.items():
        assert path.exists(), path
        parts[f'word/media/{name}'] = path.read_bytes()
    with ZipFile(DEST, 'w', ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    print(DEST)


if __name__ == '__main__':
    main()
