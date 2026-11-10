#!/usr/bin/env python3
"""Create a reversible date-level QC candidate; keep raw inputs unchanged.

The 2526 FL/G/W/P/R provenance remains unresolved until the image disk is
available, so these files are not final model inputs.
"""
from pathlib import Path
import json
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT=BASE/'analysis/healthy_candidate'
EXPECTED={'2324':20,'2526':10}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    traits=pd.read_csv(BASE/'pipeline/traits_reprocessed.csv',dtype={'season':str})
    counts=pd.read_csv(BASE/'correlations/counts_harmonized.csv',dtype={'season':str})
    yields=pd.read_csv(BASE/'analysis/yields.csv',dtype={'season':str})
    audit=[]
    for (season,date),g in traits.groupby(['season','date']):
        coverage=len(g)/40
        median_tracks=g.plant_count.median()
        reasons=[]
        if coverage<.8:reasons.append('fewer than 80% of expected plots have image features')
        if median_tracks>2*EXPECTED[season]:reasons.append('median retained tracks exceeds twice planted count')
        audit.append(dict(season=season,date=date,expected_plots=40,observed_plots=len(g),
                          coverage=coverage,median_tracks=median_tracks,planted_count=EXPECTED[season],
                          keep=not reasons,reason='; '.join(reasons)))
    table=pd.DataFrame(audit).sort_values(['season','date'])
    table.to_csv(OUT/'date_qc.csv',index=False)
    rejected=table[~table.keep][['season','date']]
    for name,frame in [('traits',traits),('counts',counts),('yields',yields)]:
        retained=frame.merge(rejected.assign(reject=True),on=['season','date'],how='left')
        retained=retained[retained.reject.isna()].drop(columns='reject')
        if name=='counts':
            retained=retained[retained[['flower','green','white','pink']].notna().all(axis=1)]
        retained.to_csv(OUT/f'{name}.csv',index=False)
    info=dict(status='candidate_only',count_source_verified=False,
              rule='drop whole dates with <80% plot image coverage or >2x expected median plant tracks; preserve raw files',
              rejected_dates=table.loc[~table.keep,['season','date','reason']].to_dict('records'),
              kept_plot_dates=table.loc[table.keep].groupby('season').observed_plots.sum().to_dict())
    (OUT/'README.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
