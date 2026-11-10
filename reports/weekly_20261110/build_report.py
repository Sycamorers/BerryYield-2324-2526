#!/usr/bin/env python3
"""Build the next weekly report from the reprocessed 2324/2526 experiment."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from report_layout import ReportWriter
from report_plots import MODEL_ORDER, FEATURE_ORDER, SCENARIOS, LABELS
from report_diagrams import overview,evaluation,image_to_trait_workflow
from distribution_analysis import main as build_distributions
DEST=ROOT/'Jing-2026-1110-yield prediction-unified.docx'
ANALYSIS=HERE/'analysis';PIPELINE=HERE/'pipeline';CORR=HERE/'correlations';FIG=HERE/'figures'

def f(value,d=2):
    return '—' if pd.isna(value) else f'{value:.{d}f}'

def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}

def read_csv(path):
    return pd.read_csv(path,dtype={'season':str})

def choose_learned(a):
    a=a[~a.model.isin(['Zero','Train mean','Train median'])]
    return a.loc[a.MAE.idxmin()] if len(a) else None

def find_figure(directory,preferred,patterns=()):
    for name in preferred:
        path=directory/name
        if path.exists():return path
    for pattern in patterns:
        paths=sorted(directory.glob(pattern))
        if paths:return paths[0]
    return None

def pipeline_pages(w):
    audit=load_json(PIPELINE/'pipeline_audit.json');cov=read_csv(PIPELINE/'coverage_by_season.csv')
    w.paragraph('1. Data reprocessing and image traits',kind='heading',new_page=True)
    w.paragraph('Roundness was recalculated from per-plant area and perimeter records. Canopy area, mean roundness and plant-track counts were aggregated by plot and date, covering 13,353 plant records and 862 plot-dates. All aggregate records agree with the reference table; the largest roundness difference is below 2 × 10⁻¹⁶.')
    w.table(['Season','Plot-dates','Image dates','Plots','Plant records'],[[str(r.season),str(int(r.plot_dates)),str(int(r.dates)),str(int(r.plots)),f'{int(r.plant_tracks):,}'] for r in cov.itertuples()],[1300,2000,1900,1300,2860])
    p=FIG/'image_to_trait_workflow.png'
    if p.exists():
        w.picture(p)
        w.paragraph('Figure 1. Image-to-trait workflow: plant detection and tracking, representative-frame selection, segmentation and plot-level trait aggregation.',kind='caption')
    w.paragraph('How the three image traits are measured',kind='label')
    w.table(['Trait','Definition'],[
        ['Canopy','Sum of retained plant-mask pixel areas in each plot; not a calibrated ground area or a single-frame union.'],
        ['Roundness','For each plant i: Rᵢ = 4πAᵢ/Pᵢ², where Aᵢ is mask area and Pᵢ is the full contour perimeter, including hole boundaries. Plot-date mean: R̄ = (1/n)ΣᵢRᵢ across n retained plants.'],
        ['Plant count','Number of retained tracks/masks; a processing count rather than an independently verified plant census.']],[1900,7460])
    w.paragraph('2. Trait distributions and measurement comparability',kind='heading',new_page=True)
    p=PIPELINE/'traits_season_comparison.png'
    if p.exists():
        w.picture(p)
        w.paragraph('Figure 2. The same trait definitions applied to all retained plot-date records in the two included seasons. These are descriptive distributions before yield matching.',kind='caption')
    w.table(['Season','Mean canopy (pixels)','Mean roundness ± SD','Mean track count'],[[str(r.season),f'{r.canopy_mean_pixels:,.0f}',f'{r.roundness_mean:.3f} ± {r.roundness_sd:.3f}',f'{r.plant_count_mean:.1f}'] for r in cov.itertuples()],[1300,2900,3060,2100])
    w.paragraph('Mean canopy pixel area is 9.35 times larger in 2324 than in 2526. Roundness is higher on average in 2526. The last 2526 scan has a median of 46.5 retained tracks per plot, compared with about 6–11 on other dates. Camera geometry, scale and tracking can contribute to these differences; the current data cannot separate measurement effects from plant biology. Cross-season errors must be interpreted in this context.')
    p=find_figure(PIPELINE,['existing_qc_2526_20260119_b11.jpg','existing_qc_overview.png','existing_qc_overview.jpg','qc_example.png','qc_example.jpg'])
    if p:
        w.picture(p,width=4.0)
        w.paragraph('Canopy-mask example: 2526, 2026-01-19, B11.',kind='caption')

def methodology_page(w,m,samples,bounds,summary):
    w.paragraph('4. Prediction task and fair comparison',kind='heading',new_page=True)
    w.paragraph('Predict observed harvest weight in g/plot approximately 1, 2 or 3 weeks after an image date. Match the nearest recorded harvest within ±2 days and require it to occur after the image. Do not interpolate harvest weight or fill missing yield with zero. The same complete-input sample IDs are used for every model and input set within each scenario and horizon. Within-season fits require two completed earlier training harvest dates.')
    w.picture(FIG/'evaluation_design.png')
    w.paragraph('Figure 4. Forward validation within a season uses only earlier completed harvest labels. The two transfer experiments keep the evaluation season out of model fitting; 2526 → 2324 is a retrospective transfer test.',kind='caption')
    w.table(['Family','Methods'],[
        ['Simple references','Zero; training mean; training median'],
        ['Statistical models','OLS; quadratic OLS; spline ridge'],
        ['Machine learning','Random forest; SVR; gradient boosting'],
        ['History-based regression','Lag OLS; lag ridge; lag random forest'],
        ['Neural sequence models','RNN; LSTM; Transformer']],[2550,6810])
    w.paragraph('Current-input models receive the latest selected traits. History models receive the last three observations from the same plot, oldest first; repeat the earliest observation when fewer than three exist. Visits are irregular. Dates, phase labels, plot IDs and future harvests are not predictors. Scale features using training data only. Use fixed settings and average the neural predictions from seeds 42, 123 and 2026; no evaluation-season tuning is performed. Neural models use 80 epochs, hidden width 16 and AdamW learning rate 0.003.')
    w.paragraph('Complete image/count plot-dates: 318 in 2324 and 504 in 2526. Exclude 40 image-only 2324 rows with missing counts. Measured yield: 320 labels in 2324 and 480 in 2526. Preserve recorded zeros and missing cells. Six input sets: canopy; roundness; counts; canopy + roundness; canopy + counts; all traits. Counts = manual flowers/green/white/pink fruit + plant tracks; all traits add canopy and roundness. No YOLO fruit counts enter the models. Clip negative forecasts to zero.')

def model_page(w,m,scenario):
    a=m[(m.scenario==scenario)&(m.feature_set=='All traits')]
    w.paragraph(LABELS[scenario],kind='heading',new_page=True)
    w.paragraph('All-trait results: canopy, roundness, manual flower/green/white/pink counts and plant-track count. Each error is per plot per matched harvest, not a full-season yield total.')
    support=[]
    statements=[]
    for h in [1,2,3]:
        b=a[a.horizon_weeks==h]
        if len(b):
            n=int(b.n.iloc[0]);dates=int(b.target_dates.iloc[0]) if 'target_dates' in b else None
            support.append(f'{h}w: n={n}'+(f', {dates} harvest dates' if dates is not None else ''))
            best=choose_learned(b)
            statements.append(f'{h}w {best.model} ({best.MAE:.1f} g, R²={best.R2:.2f})')
        else:support.append(f'{h}w: unavailable')
    w.paragraph('Evaluation support: '+ '; '.join(support)+'.')
    rows=[]
    for model in MODEL_ORDER:
        vals=[]
        for h in [1,2,3]:
            b=a[(a.model==model)&(a.horizon_weeks==h)]
            vals += [f(b.MAE.iloc[0]) if len(b) else '—',f(b.R2.iloc[0],2) if len(b) else '—']
        rows.append([model]+vals)
    w.table(['Method','1w MAE','1w R²','2w MAE','2w R²','3w MAE','3w R²'],rows)
    w.paragraph('Lowest observed learned-model MAE: '+ '; '.join(statements)+'. Rankings describe these held-out data and do not establish a stable best model.')
    p=FIG/('horizons_'+scenario+'.png')
    if p.exists():
        w.picture(p)
        w.paragraph('Figure. Fixed representatives from each learned family versus the training-mean reference. The complete table includes all 15 methods; panels and horizons can have different eligible observations.',kind='caption')
    if scenario=='within_2324' and not a.horizon_weeks.eq(3).any():w.paragraph('No three-week forecast origin has two completed earlier training harvest dates in 2324. Its within-season accuracy is unavailable; no value is estimated from future labels.')

def ablation_page(w,m,ablation):
    w.paragraph('Input comparison: what do the traits add?',kind='heading',new_page=True)
    w.paragraph('Hold the evaluation observations, model settings and validation protocol fixed while changing only the input set. The figures show fixed representative methods; the table gives the lowest observed learned-model MAE for each input set at one week.')
    w.picture(FIG/'feature_comparison.png')
    w.paragraph('Figure. One-week MAE for canopy, roundness, counts, canopy + roundness, canopy + counts and all traits. Each panel has its own scale. Counts combine manual developmental counts and retained plant-track count.',kind='caption')
    rows=[]
    for feat in FEATURE_ORDER:
        vals=[]
        for s in SCENARIOS:
            b=m[(m.scenario==s)&(m.horizon_weeks==1)&(m.feature_set==feat)]
            best=choose_learned(b)
            vals.append(f'{best.MAE:.1f}' if best is not None else '—')
        rows.append([feat]+vals)
    w.table(['Inputs','2324 → 2526','2526 → 2324','Within 2324','Within 2526'],rows,[2600,1690,1690,1690,1690])
    parts=[]
    for sc in SCENARIOS:
        a=ablation[(ablation.scenario==sc)&(~ablation.model.isin(MODEL_ORDER[:3]))]
        parts.append(f"{LABELS[sc].replace(': forward validation','')}: {int(a.roundness_added_to_canopy_counts.lt(0).sum())}/{len(a)}")
    w.paragraph('After canopy and counts are already supplied, adding roundness lowers MAE in these learned-model / horizon comparisons: '+ '; '.join(parts)+'. These fractions describe fixed comparisons; they are not statistical success rates.')

def support_table(w,stages,scenarios):
    rows=[]
    for sc in scenarios:
        a=stages[(stages.scenario==sc)&(stages.feature_set=='All traits')&(stages.horizon_weeks==1)&(stages.model=='Train mean')]
        vals=[]
        for stage in ['Early','Middle','Late']:
            b=a[a.stage==stage]
            if len(b):
                r=b.iloc[0];vals.append(f'n={int(r.n)}'+(f'; {int(r.target_dates)} dates' if 'target_dates' in b else ''))
            else:vals.append('Unavailable')
        rows.append([LABELS[sc].replace(': forward validation','')]+vals)
    w.table(['Evaluation','Early','Middle','Late'],rows,[3660,1900,1900,1900])

def stage_pages(w,m,stages,bounds,unavailable):
    w.paragraph('Early, middle and late harvest phases: across seasons',kind='heading',new_page=True)
    w.paragraph('First evaluate the same full-season model separately within each harvest phase. This shows where errors occur without changing the fitted model. Separately fitted phase models are an additional exploratory comparison; a target phase must be known to select one.')
    w.picture(FIG/'stage_cross.png')
    w.paragraph('Figure. One-week, all-trait MAE by target harvest phase for pooled models. The two transfer directions have different evaluation seasons and supports.',kind='caption')
    support_table(w,stages,SCENARIOS[:2])
    rows=[]
    for sc in SCENARIOS[:2]:
        for model in ['Train mean','OLS','Random forest','Lag ridge','LSTM','Transformer']:
            vals=[]
            for stage in ['Early','Middle','Late']:
                for scenario in [sc,'stage_'+sc]:
                    b=stages[(stages.scenario==scenario)&(stages.feature_set=='All traits')&(stages.horizon_weeks==1)&(stages.model==model)&(stages.stage==stage)]
                    vals.append(f(b.MAE.iloc[0],1) if len(b) else '—')
            rows.append([('23→25' if sc==SCENARIOS[0] else '25→23')+' '+model]+vals)
    w.table(['Direction / model','Early pool','Early fit','Mid pool','Mid fit','Late pool','Late fit'],rows)
    w.paragraph('23→25 = 2324→2526; 25→23 = 2526→2324. Pool = one model trained using all source-season phases; fit = model trained using the matching source-season phase only. Compare each cell pair on identical evaluation sample IDs. All methods, horizons and input sets are preserved in CSV. A phase-specific improvement is descriptive, not a separately validated model-selection rule.')
    w.paragraph('Early, middle and late harvest phases: within each season',kind='heading',new_page=True)
    w.picture(FIG/'stage_within.png')
    w.paragraph('Figure. One-week, all-trait forward errors split by target harvest phase. Missing early-phase results mean insufficient earlier completed harvests; they are not zero errors.',kind='caption')
    support_table(w,stages,SCENARIOS[2:])
    rows=[]
    for model in MODEL_ORDER:
        vals=[]
        for sc in SCENARIOS[2:]:
            for stage in ['Early','Middle','Late']:
                b=stages[(stages.scenario==sc)&(stages.feature_set=='All traits')&(stages.horizon_weeks==1)&(stages.model==model)&(stages.stage==stage)]
                vals.append(f(b.MAE.iloc[0],1) if len(b) else '—')
        rows.append([model]+vals)
    w.table(['Method','23 early','23 middle','23 late','25 early','25 middle','25 late'],rows)
    w.paragraph('Separate within-phase forward models require at least two completed earlier harvest dates in that same phase. Unsupported fits are recorded as unavailable, rather than using future labels. Phase support is small, and repeated plots do not replace independent harvest dates.')



def correlation_pages(w):
    c=read_csv(CORR/'correlations.csv')
    w.paragraph('3. Trait–yield correlations',kind='heading',new_page=True)
    w.paragraph('Pearson correlations link traits at the image date to measured harvest weight 1, 2 or 3 weeks later. Each heatmap cell shows r, the number of plot-harvest observations n and the number of distinct target harvest dates d. All correlations are descriptive; repeated plots and few dates limit evidence for a general relationship.')
    w.picture(CORR/'correlation_overview.png')
    w.paragraph('Figure 3a. Trait–yield relationships differ by season and lead time. Red fruit is included only in this descriptive study and is excluded from predictive model inputs.',kind='caption')
    w.paragraph('At one week, 2324 canopy has r=0.325 and roundness r=−0.070; white and pink fruit counts have r=0.274 and 0.374. In 2526, canopy has r=0.113, roundness r=−0.339, white fruit r=0.447 and pink fruit r=0.437. Counts and image traits therefore do not behave identically across seasons.')
    w.paragraph('Centering observations within each target harvest date changes 2526 canopy r to 0.305 and roundness r to −0.196. Part of the pooled correlation reflects changes over the season rather than differences between plots on the same harvest date. Spearman correlations and date-centered results are retained in correlations.csv. No causal effect is inferred.')
    w.paragraph('Correlations change across harvest phases',kind='heading',new_page=True)
    w.picture(CORR/'correlation_stages_h1.png')
    w.paragraph('Figure 3b. One-week correlations split by early, middle and late target harvest phases. Each cell includes its observation and harvest-date support. Phases are defined separately within each season.',kind='caption')
    w.paragraph('For 2526, canopy r is 0.384 early, 0.515 in the middle and −0.044 late. White fruit r is 0.591, 0.258 and 0.543. A relationship that appears useful in one phase can weaken in another; this motivates phase-specific error reporting as well as pooled model comparisons.')
    w.paragraph('Each phase usually contains only two or three matched harvest dates. These coefficients should guide hypotheses and follow-up data collection, not be treated as stable significance tests or used to choose predictors from the evaluation season.')


def distribution_pages(w):
    d=read_csv(CORR/'distributions.csv')
    w.paragraph('Measured-variable distributions',kind='heading',new_page=True)
    w.paragraph('Figure 3a shows all three image traits and observed yield by season; Figure 3b shows every manual flower and fruit count, including red fruit used only for descriptive analysis. Histograms use observed values before matching predictor dates to future harvests. The vertical lines are medians. The image traits, field counts and yield therefore have different source populations and sample sizes; their panels must not be interpreted as matched pairs.')
    w.picture(CORR/'image_yield_distributions.png')
    w.paragraph('Figure 3a. Distribution of canopy area, plot-mean roundness, retained plant tracks and recorded harvest yield. Pixel areas are uncalibrated. Yield is grams per plot per recorded harvest, including valid zero values.',kind='caption')
    w.picture(CORR/'field_count_distributions.png')
    w.paragraph('Figure 3b. Distributions of manual flower, green, white, pink and red fruit counts per plot per visit. Missing field cells are excluded from histograms and are never replaced by zero.',kind='caption')
    w.paragraph('Distribution summary',kind='heading',new_page=True)
    rows=[]
    for r in d.itertuples():
        rows.append([r.label,str(r.season),str(int(r.n)),str(int(r.missing)),str(int(r.zeros)),f'{r.median:,.2f}',f'{r.q25:,.2f}–{r.q75:,.2f}'])
    w.table(['Variable','Season','n','Missing','Zeros','Median','IQR (Q1–Q3)'],rows,[2700,820,720,900,800,1250,2170])
    w.paragraph('Roundness is a dimensionless 4πA/P² measure, averaged across retained plant masks in each plot. Its two seasonal distributions overlap but have different centers and spread. Canopy area differs strongly in scale across seasons; the median and interquartile range make this visible without treating camera pixels as physical area. Counts are nonnegative and often skewed, so medians and quartiles accompany the mean and standard deviation in the exported distributions.csv. The recorded yield distribution is also skewed. Zeros and missing cells are reported separately because they have different meanings.')
    w.paragraph('The table is descriptive. Inference about future yield uses only the date-matched pairs in the next section; comparison of these marginal distributions alone does not establish association, model value or a biological change between seasons.')


def diagnostic_page(w,m):
    from report_plots import prediction_diagnostics
    pred=read_csv(ANALYSIS/'predictions.csv')
    prediction_diagnostics(m,pred)
    w.paragraph('How forecasts differ from measured harvests',kind='heading',new_page=True)
    w.paragraph('Show the measured and predicted mean g/plot for each target harvest date at one week. For illustration, each panel uses the learned model with the lowest observed MAE in that scenario. This selection uses the evaluation results and is exploratory; it is not an independent assessment of a model chosen in advance.')
    w.picture(FIG/'observed_predicted.png')
    w.paragraph('Figure. Date means reveal missed yield peaks and systematic bias. R² shown above each panel is calculated from individual plot observations; averaging by date in this figure does not change the score or create additional observations.',kind='caption')
    rows=[]
    for sc in SCENARIOS:
        b=m[(m.scenario==sc)&(m.feature_set=='All traits')&(m.horizon_weeks==1)]
        r=choose_learned(b)
        rows.append([LABELS[sc].replace(': forward validation',''),r.model,f(r.MAE,1),f(r.RMSE,1),f(r.R2,2),f(r['Bias'],1)])
    w.table(['Evaluation','Illustrated model','MAE','RMSE','R²','Bias'],rows,[2800,2060,1100,1100,1100,1200])
    w.paragraph('Bias = predicted − observed yield: positive values indicate overprediction. A low MAE can coexist with a negative R² if large peaks are missed; negative R² means worse squared error than the evaluation group’s mean. Comparing only the lowest error hides this distinction.')


def conclusion_page(w,m,stages,samples,bounds,summary):
    w.paragraph('Main results and interpretation',kind='heading',new_page=True)
    a=m[(m.feature_set=='All traits')&(~m.scenario.str.startswith('stage_'))&(~m.model.isin(MODEL_ORDER[:3]))]
    w.paragraph(f'The comparison includes 15 methods, six input sets, three forecast horizons and four main evaluation settings. With all traits, {int(a.R2.lt(0).sum())} of {len(a)} available learned-model / horizon results have negative R². Any useful improvement must be judged against simple references, measurement differences and the small number of harvest dates.')
    w.paragraph('Completed analysis',kind='label')
    w.paragraph('The numeric image-trait outputs were recalculated, aggregated and verified. An independent review passed all 25 numeric and evaluation-protocol checks; saved cross-season model reloads reproduced 147,960 prediction values within 1.82 × 10⁻¹². A single sample table links image traits and harmonized manual counts to observed yield. Every model uses the same held-out sample IDs within each setting. Both transfer directions, both within-season forward tests, input comparisons and harvest-phase errors are included.')
    w.paragraph('Interpreting seasonal differences',kind='label')
    w.paragraph('Pixel area depends on camera scale, and plant-track counts depend on tracking behavior. These measurement differences can contribute to cross-season changes alongside plant biology. The analysis retains the observations without removing rows because they predict poorly in the evaluation season.')
    w.paragraph('Model rankings were inspected on these evaluation data. They are descriptive comparisons, not a final untouched validation after model selection. Three neural seeds measure training variation; target-date bootstrap intervals reflect the small observed date set, not new seasons. Additional independent dates or seasons are needed before claiming reliable deployment.')
    w.paragraph('Same-population comparisons',kind='label')
    matched_path=ANALYSIS/'matched_transfer_within_metrics.csv'
    if matched_path.exists():
        matched=read_csv(matched_path)
        rows=[]
        for sc in SCENARIOS:
            b=matched[(matched.feature_set=='All traits')&(matched.horizon_weeks==1)&(matched.scenario==sc)&(matched.model=='LSTM')]
            if len(b):
                r=b.iloc[0];rows.append([LABELS[sc].replace(': forward validation',''),str(int(r.n)),f(r.MAE,1),f(r.R2,2)])
        w.table(['LSTM on shared test sample IDs','n','MAE','R²'],rows,[5160,1200,1500,1500])
        w.paragraph('For each evaluation season, transfer and within-season rows above use exactly the sample IDs eligible for forward validation. The broader tables earlier in the report use every eligible sample for each protocol and should not be compared as if their populations were identical.')
    w.paragraph('Harvest phase definitions',kind='label')
    w.table(['Season','Early harvests','Middle harvests','Late harvests'],[[season]+[f"{bounds[(bounds.season==season)&(bounds.stage==stage)].iloc[0].start} to {bounds[(bounds.season==season)&(bounds.stage==stage)].iloc[0].end}" for stage in ['Early','Middle','Late']] for season in ['2324','2526']],[1000,2780,2780,2800])
    w.paragraph('Reproducibility and packaged outputs',kind='label')
    w.paragraph('The delivery folder contains this Word/PDF report, figure PNG/PDF files, numeric result tables, exact sample IDs, fit audit records, source hashes, seed results and reproducible Python scripts. Run reprocess_pipeline.py, correlation_analysis.py, run_analysis.py and build_report.py from reports/weekly_20261110/ as documented in the package README. All numerical results come from saved observations; 2425 is excluded.')


def main():
    from report_plots import main as plot_results
    overview();evaluation();image_to_trait_workflow();plot_results();build_distributions()
    m=read_csv(ANALYSIS/'metrics.csv');stages=read_csv(ANALYSIS/'stage_metrics.csv')
    samples=read_csv(ANALYSIS/'samples.csv');bounds=read_csv(ANALYSIS/'stage_boundaries.csv')
    summary=load_json(ANALYSIS/'summary.json')
    unavailable=read_csv(ANALYSIS/'unavailable.csv')
    ablation=read_csv(ANALYSIS/'ablation.csv')
    assert set(samples.season.astype(str))=={'2324','2526'}
    w=ReportWriter(DEST);w.cover()
    w.paragraph('Weekly research activities and yield model development',kind='title')
    w.paragraph('Weekly report | 10 November 2026')
    w.paragraph('Research activities last week',kind='label')
    w.paragraph('Revised and submitted the dissertation. Completed processing the canopy data, calculated plant roundness, and redeveloped the yield-prediction models using the 2324 and 2526 seasons. The detailed methods and results follow the weekly update.')
    w.paragraph('Handed over the use of ASTIR to Zekun and completed functional testing. The tested ASTIR workflow showed no issues. Checked the field and scheduled the date and travel for a visit to the Tampa commercial field. Tested ABV; no issues were observed in the current test.')
    w.paragraph('Problems encountered',kind='label')
    w.paragraph('No problems were identified in the ASTIR handover and functional test or the current ABV test. Performance in the Tampa commercial field has not yet been tested.')
    w.paragraph('Plan for next week',kind='label')
    w.paragraph('Test ABV in the Tampa commercial field, collect field data, and analyze the resulting measurements and system performance.')
    w.paragraph('Yield model development: canopy, roundness and field counts',kind='heading',new_page=True)
    w.paragraph('Detailed analysis | Seasons 2324 and 2526 only')
    w.paragraph('Abstract. This study reprocessed canopy mask measurements and plant roundness in the 2324 and 2526 strawberry seasons, aligned them with manual flower and fruit counts and observed harvest yield, and evaluated future-yield prediction at one, two and three weeks. It compared 15 fixed methods and six feature sets under cross-season and chronological within-season protocols. Roundness and other traits showed season-dependent associations with yield; forecast rankings varied by evaluation setting. Pixel-scale differences, few harvest dates and repeated plots limit generalization. Season 2425 is excluded because its data are unreliable.')
    w.paragraph('Research questions. Can a consistent set of canopy, roundness and count measurements be rebuilt? How well do statistical, machine learning and sequence models predict future harvest weight within and across seasons?')
    w.picture(FIG/'study_overview.png')
    w.paragraph('Study design. Rebuild image-derived traits, link them to observed future harvests, then compare all model families on shared evaluation observations and by harvest phase.',kind='caption')
    w.paragraph('Main findings',kind='label')
    w.paragraph('Roundness was recalculated from area and perimeter records, and canopy and plant-track counts were aggregated across 13,353 plant records into 862 plot-date records. Differences in pixel area and tracking between seasons help explain the cross-season measurement shift.')
    winners=[]
    for sc in SCENARIOS:
        b=m[(m.scenario==sc)&(m.feature_set=='All traits')&(m.horizon_weeks==1)]
        best=choose_learned(b)
        if best is not None:winners.append(f'{LABELS[sc].replace(": forward validation","")}: {best.model}, MAE {best.MAE:.1f} g/plot, R² {best.R2:.2f}')
    w.paragraph('One-week all-trait learned-model rankings: '+ '; '.join(winners)+'. These are descriptive results; use the full tables and simple references to judge the size and consistency of any improvement.')
    pipeline_pages(w)
    distribution_pages(w)
    correlation_pages(w)
    methodology_page(w,m,samples,bounds,summary)
    for sc in SCENARIOS:model_page(w,m,sc)
    diagnostic_page(w,m)
    ablation_page(w,m,ablation)
    stage_pages(w,m,stages,bounds,unavailable)
    conclusion_page(w,m,stages,samples,bounds,summary)
    result=w.save()
    qa={'report':str(result),'included_seasons':['2324','2526'],'excluded_season':'2425','model_count':len(MODEL_ORDER),'scenarios':SCENARIOS,'feature_sets':FEATURE_ORDER,'source_directory':str(ANALYSIS),'contains_only_saved_results':True,'raw_image_pipeline_rerun':False}
    (HERE/'report_qa.json').write_text(json.dumps(qa,indent=2))
    print(result)

if __name__=='__main__':main()
