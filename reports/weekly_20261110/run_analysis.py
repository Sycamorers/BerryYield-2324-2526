#!/usr/bin/env python3
"""Two-season fixed-setting yield benchmark; no 2425 data are loaded.

Run from the project root: python reports/weekly_20261110/run_analysis.py
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/berry_weekly_20261110_mpl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from threadpoolctl import threadpool_limits
from canopy_temporal_benchmark.load_2526_workbook import load_workbook_yields
from canopy_temporal_benchmark.run_benchmark import MODEL_INFO, NEURAL, SEEDS, make_estimator, metric_table, cluster_differences
from canopy_temporal_benchmark.neural import predict_neural, get_config
from reports.build_canopy_report import select_matches

BASE = Path(__file__).resolve().parent
OUT = BASE / 'analysis'
COUNTS = ['flower', 'green', 'white', 'pink']
FEATURE_SETS = {
    'Canopy': ['canopy_size_pixels'],
    'Roundness': ['mean_roundness'],
    'Counts': COUNTS,
    'Canopy + roundness': ['canopy_size_pixels', 'mean_roundness'],
    'Canopy + counts': ['canopy_size_pixels'] + COUNTS,
    'All features': ['canopy_size_pixels', 'mean_roundness'] + COUNTS,
}
ALL = FEATURE_SETS['All features']
STAGES = ['Early', 'Middle', 'Late']


def load_inputs():
    traits_path = BASE/'pipeline/traits_reprocessed.csv'
    count_path = BASE/'correlations/counts_harmonized.csv'
    traits = pd.read_csv(traits_path, dtype={'season': str}, parse_dates=['date'])
    counts = pd.read_csv(count_path, dtype={'season': str}, parse_dates=['date'])
    assert set(traits.season) == set(counts.season) == {'2324', '2526'}
    merged = traits.merge(counts, on=['season','date','plot'], how='left', validate='one_to_one')
    valid = np.isfinite(merged[ALL]).all(axis=1) & merged.canopy_size_pixels.gt(0) & merged.mean_roundness.between(0,1) & merged[COUNTS].ge(0).all(axis=1)
    excluded = merged[~valid].copy()
    observations = merged[valid].sort_values(['season','plot','date']).copy()
    labels, sources = [], [traits_path, count_path]
    for path in sorted((ROOT/'data/real_data/2324_GNV_processed').glob('*/consolidated_summary_with_yield.csv')):
        frame = pd.read_csv(path,index_col=0)
        if 'Yield (g)' not in frame.index: continue
        labels.append(pd.DataFrame({'season':'2324','date':pd.to_datetime(path.parent.name,format='%y%m%d'),
            'plot':frame.columns.str.strip().str.upper(), 'yield_g':pd.to_numeric(frame.loc['Yield (g)'],errors='coerce').to_numpy()}))
        sources.append(path)
    workbook = ROOT/'check-in & data-col 2526.xlsx'
    yields_2526, audit = load_workbook_yields(workbook)
    yields_2526.to_csv(OUT/'yield_2526_extracted.csv', index=False)
    audit.to_csv(OUT/'workbook_yield_audit.csv',index=False)
    labels.append(yields_2526[['season','date','plot','yield_g']]); sources.append(workbook)
    yields = pd.concat(labels,ignore_index=True).dropna(subset=['yield_g'])
    assert np.isfinite(yields.yield_g).all() and yields.yield_g.ge(0).all()
    assert not yields.duplicated(['season','date','plot']).any()
    return observations, yields, excluded, sources


def build_samples(observations, yields):
    histories = []
    for (season, plot), group in observations.groupby(['season','plot']):
        group = group.sort_values('date')
        for i, row in enumerate(group.to_dict('records')):
            hist = group.iloc[max(0,i-2):i+1]
            values = np.pad(hist[ALL].to_numpy(), ((3-len(hist),0),(0,0)), mode='edge')
            rec = {k:row[k] for k in ['season','plot']+ALL}
            rec.update(canopy_date=row['date'], history_count=len(hist), history_dates='|'.join(hist.date.dt.strftime('%Y-%m-%d')),
                history_span_days=(row['date']-hist.date.min()).days)
            for step in range(3):
                rec.update({f'{feature}_{step}': values[step,j] for j,feature in enumerate(ALL)})
            histories.append(rec)
    origins = pd.DataFrame(histories)
    samples, boundaries = [], []
    for season, part in origins.groupby('season'):
        sy = yields[yields.season.eq(season)]
        dates = pd.DatetimeIndex(sorted(sy.date.unique()))
        mapping = {}
        for stage, block in zip(STAGES, np.array_split(dates,3)):
            if not len(block): continue
            for day in block: mapping[pd.Timestamp(day)] = stage
            boundaries.append(dict(season=season,stage=stage,start=block[0],end=block[-1],dates=len(block)))
        for horizon in (1,2,3):
            for match in select_matches(pd.DatetimeIndex(part.canopy_date.unique()), dates, horizon):
                day = part[part.canopy_date.eq(match.canopy_date)]
                target = sy[sy.date.eq(match.yield_date)][['plot','yield_g']]
                joined = day.merge(target,on='plot',validate='one_to_one')
                joined['yield_date'] = match.yield_date
                joined['expected_yield_date'] = match.expected_yield_date
                joined['date_offset_days'] = match.date_offset_days
                joined['horizon_weeks'] = horizon
                joined['stage'] = mapping[match.yield_date]
                joined['sample_id'] = joined.apply(lambda r: f'{season}_h{horizon}_{r.canopy_date:%Y%m%d}_{r["plot"]}',axis=1)
                samples.append(joined)
    samples = pd.concat(samples,ignore_index=True)
    assert samples.sample_id.is_unique
    assert samples.yield_date.gt(samples.canopy_date).all()
    return samples, origins, pd.DataFrame(boundaries)


def arrays(frame, features):
    return np.stack([frame[[f'{c}_{i}' for c in features]].to_numpy() for i in range(3)], axis=1)


def fit_predict(train, test, feature_set, scenario, fit_id, seed_rows, fit_rows, save=False):
    assert len(train) and len(test)
    assert set(train.sample_id).isdisjoint(test.sample_id)
    same_season = 'within_' in scenario
    chronological = bool(train.yield_date.max() < test.canopy_date.min())
    if same_season:
        assert chronological and set(train.season) == set(test.season)
    else:
        assert set(train.season).isdisjoint(set(test.season))
    features = FEATURE_SETS[feature_set]
    tx, vx, ty = arrays(train,features), arrays(test,features), train.yield_g.to_numpy()
    fit_rows.append(dict(fit_id=fit_id,feature_set=feature_set,scenario=scenario,horizon_weeks=int(test.horizon_weeks.iloc[0]),
        n_train=len(train),n_test=len(test),train_target_dates=train.yield_date.nunique(),test_target_dates=test.yield_date.nunique(),
        train_latest_target=train.yield_date.max(),test_first_origin=test.canopy_date.min(),
        label_available_before_forecast=chronological,evaluation_mode='chronological' if same_season else 'season_holdout',
        training_seasons='|'.join(sorted(set(train.season))),test_seasons='|'.join(sorted(set(test.season))),
        train_sample_ids='|'.join(train.sample_id),test_sample_ids='|'.join(test.sample_id)))
    metadata=['sample_id','season','plot','canopy_date','yield_date','horizon_weeks','stage','yield_g','history_count','history_span_days']
    frames=[]
    folder=OUT/'models'/feature_set.lower().replace(' + ','_').replace(' ','_')/scenario/f'h{int(test.horizon_weeks.iloc[0])}'
    if save: folder.mkdir(parents=True,exist_ok=True)
    for name,(family,info) in MODEL_INFO.items():
        if name in ('Zero','Train mean','Train median'):
            constant={'Zero':0.,'Train mean':float(ty.mean()),'Train median':float(np.median(ty))}[name]
            pred=np.full(len(test),constant)
            if save: joblib.dump(dict(model=name,constant=constant,features=features),folder/f'{name}.joblib')
        elif name in NEURAL:
            values=[]
            for seed in SEEDS:
                ps=np.maximum(0,predict_neural(name,tx,ty,vx,seed=seed,bundle_path=folder/f'{name}_{seed}.pt' if save else None))
                values.append(ps)
                sf=test[metadata].copy();sf['prediction']=ps;sf['model']=name;sf['seed']=seed
                sf['scenario']=scenario;sf['feature_set']=feature_set;sf['fit_id']=fit_id
                seed_rows.append(sf)
            pred=np.mean(values,axis=0)
        else:
            x,xt=(tx[:,-1,:],vx[:,-1,:]) if info=='current' else (tx.reshape(len(tx),-1),vx.reshape(len(vx),-1))
            estimator=make_estimator(name);estimator.fit(x,ty);pred=estimator.predict(xt)
            if save: joblib.dump(dict(estimator=estimator,features=features,input=info),folder/f'{name}.joblib')
        pred=np.maximum(np.asarray(pred).reshape(-1),0)
        assert np.isfinite(pred).all()
        frame=test[metadata].copy();frame['prediction']=pred;frame['model']=name;frame['family']=family
        frame['feature_set']=feature_set;frame['scenario']=scenario;frame['fit_id']=fit_id
        frames.append(frame)
    return frames


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    observations,yields,excluded,sources=load_inputs()
    samples,origins,bounds=build_samples(observations,yields)
    for name,frame in [('observations',observations),('yields',yields),('excluded_observations',excluded),('samples',samples),('origins',origins),('stage_boundaries',bounds)]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    coverage=samples.groupby(['season','horizon_weeks','stage']).agg(n=('sample_id','size'),origin_dates=('canopy_date','nunique'),target_dates=('yield_date','nunique')).reset_index()
    coverage.to_csv(OUT/'sample_coverage.csv',index=False)
    predictions,seed_rows,fit_rows,unavailable=[],[],[],[]
    with threadpool_limits(limits=1):
        for feature_set in FEATURE_SETS:
            print(f'FEATURE SET: {feature_set}',flush=True)
            for h in (1,2,3):
                hs=samples[samples.horizon_weeks.eq(h)]
                for source,target in [('2324','2526'),('2526','2324')]:
                    tr,te=hs[hs.season.eq(source)],hs[hs.season.eq(target)]
                    scenario=f'{source}_to_{target}'
                    print(f'  h{h} {scenario}: {len(tr)} -> {len(te)}',flush=True)
                    predictions.extend(fit_predict(tr,te,feature_set,scenario,f'{feature_set}_{scenario}_h{h}',seed_rows,fit_rows,save=True))
                    for stage in STAGES:
                        a,b=tr[tr.stage.eq(stage)],te[te.stage.eq(stage)]
                        if len(a) and len(b):
                            predictions.extend(fit_predict(a,b,feature_set,'stage_'+scenario,f'{feature_set}_stage_{scenario}_h{h}_{stage}',seed_rows,fit_rows))
                        else:
                            unavailable.append(dict(feature_set=feature_set,scenario='stage_'+scenario,horizon_weeks=h,stage=stage,n_train=len(a),n_test=len(b),reason='No training pairs in phase' if not len(a) else 'No test pairs in phase'))
                for season in ('2324','2526'):
                    season_samples=hs[hs.season.eq(season)]
                    for origin,te in season_samples.groupby('canopy_date',sort=True):
                        tr=season_samples[season_samples.yield_date.lt(origin)]
                        scenario='within_'+season
                        if tr.yield_date.nunique()<2:
                            for stage,b in te.groupby('stage'):
                                unavailable.append(dict(feature_set=feature_set,scenario=scenario,horizon_weeks=h,origin=origin,stage=stage,n_train=len(tr),n_test=len(b),reason='Fewer than two completed training harvest dates'))
                            continue
                        predictions.extend(fit_predict(tr,te,feature_set,scenario,f'{feature_set}_{scenario}_h{h}_{origin:%Y%m%d}',seed_rows,fit_rows))
                        for stage,b in te.groupby('stage'):
                            a=tr[tr.stage.eq(stage)]
                            if a.yield_date.nunique()>=2:
                                predictions.extend(fit_predict(a,b,feature_set,'stage_'+scenario,f'{feature_set}_stage_{scenario}_h{h}_{origin:%Y%m%d}_{stage}',seed_rows,fit_rows))
                            else:
                                unavailable.append(dict(feature_set=feature_set,scenario='stage_'+scenario,horizon_weeks=h,origin=origin,stage=stage,n_train=len(a),n_test=len(b),reason='Fewer than two completed same-phase training harvest dates'))
            print(f'  completed {feature_set}; elapsed {time.time()-started:.1f} s',flush=True)
    pred=pd.concat(predictions,ignore_index=True);seeds=pd.concat(seed_rows,ignore_index=True)
    metrics=metric_table(pred,['feature_set','scenario','horizon_weeks','model','family'])
    stages=metric_table(pred,['feature_set','scenario','horizon_weeks','stage','model'])
    seed_metrics=metric_table(seeds,['feature_set','scenario','horizon_weeks','model','seed'])
    assert not pred.duplicated(['feature_set','scenario','model','sample_id']).any()
    for keys,g in pred.groupby(['scenario','horizon_weeks']):
        ids=[set(p.sample_id) for _,p in g.groupby(['feature_set','model'])]
        assert len(ids)==len(FEATURE_SETS)*len(MODEL_INFO) and all(s==ids[0] for s in ids)
    assert set(pred.season)=={'2324','2526'}
    fits=pd.DataFrame(fit_rows)
    assert fits.loc[fits.evaluation_mode.eq('chronological'),'label_available_before_forecast'].all()
    main_metrics=metrics[~metrics.scenario.str.startswith('stage_')]
    learned=main_metrics[~main_metrics.model.isin(['Zero','Train mean','Train median'])]
    winners=learned.loc[learned.groupby(['feature_set','scenario','horizon_weeks']).MAE.idxmin()]
    ablation=main_metrics.pivot(index=['scenario','horizon_weeks','model'],columns='feature_set',values='MAE').reset_index()
    ablation['roundness_added_to_canopy']=ablation['Canopy + roundness']-ablation['Canopy']
    ablation['roundness_added_to_canopy_counts']=ablation['All features']-ablation['Canopy + counts']
    ablation['counts_added_to_canopy_roundness']=ablation['All features']-ablation['Canopy + roundness']
    ablation['all_minus_canopy']=ablation['All features']-ablation['Canopy']
    bootstrap=[]
    if 'Train mean' in MODEL_INFO:
        for feature_set,g in pred[~pred.scenario.str.startswith('stage_')].groupby('feature_set'):
            b=cluster_differences(g);b['feature_set']=feature_set;bootstrap.append(b)
    ranges=[]
    for source,target in [('2324','2526'),('2526','2324')]:
        for c in ALL:
            a,b=[observations[observations.season.eq(s)][c] for s in (source,target)]
            ranges.append(dict(scenario=f'{source}_to_{target}',feature=c,train_min=a.min(),train_max=a.max(),test_min=b.min(),test_max=b.max(),test_fraction_outside_train_range=((b<a.min())|(b>a.max())).mean()))
    matched=[]
    for target in ('2324','2526'):
        within='within_'+target;transfer=('2526' if target=='2324' else '2324')+'_to_'+target
        ids=set(pred[pred.scenario.eq(within)].sample_id)
        part=pred[pred.scenario.isin([within,transfer]) & pred.sample_id.isin(ids)]
        matched.append(metric_table(part,['feature_set','scenario','horizon_weeks','model']))
    # Compare phase-specific training to pooled training on identical test pairs.
    matched_stage=[]
    for stage_scenario,part in pred[pred.scenario.str.startswith('stage_')].groupby('scenario'):
        pooled_scenario=stage_scenario[len('stage_'):]
        ids=set(part.sample_id)
        comparison=pd.concat([part,pred[pred.scenario.eq(pooled_scenario)&pred.sample_id.isin(ids)]])
        matched_stage.append(metric_table(comparison,['feature_set','scenario','horizon_weeks','stage','model']))
    for name,frame in [('predictions',pred),('metrics',metrics),('stage_metrics',stages),('seed_predictions',seeds),('seed_metrics',seed_metrics),('fit_audit',fits),('unavailable',pd.DataFrame(unavailable)),('ablation',ablation),('winners',winners),('paired_mae_bootstrap',pd.concat(bootstrap) if bootstrap else pd.DataFrame()),('input_ranges',pd.DataFrame(ranges)),('matched_transfer_within_metrics',pd.concat(matched)),('matched_phase_training_metrics',pd.concat(matched_stage))]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    summary=dict(seasons=['2324','2526'],excluded_seasons=['2425'],feature_sets=FEATURE_SETS,models={k:list(v) for k,v in MODEL_INFO.items()},
        target_unit='grams per plot per observed harvest; not total seasonal yield',horizons_weeks=[1,2,3],match_tolerance_days=2,
        history='Three available joint complete observations oldest first, left-pad earliest observation; irregular intervals; ends at origin',
        phase_definition='Fixed thirds of the ordered observed harvest dates in each season, assigned using target harvest date; retrospective calendar phases, not physiological stages',
        within_season='Expanding-window refit; every training target strictly before test origin; require at least two completed training target dates',
        cross_season='Both disjoint-season folds. Reverse 2526 to 2324 is retrospective generalization, not an operational chronological forecast',
        count_source='Manual flower, green, white and pink counts in both seasons, plus tracked segmented plant count; no mixing with YOLO fruit counts',
        input_policy='Predictors are canopy size, mean roundness, FL, G, W and P only; mature red fruit and retained plant-track count are excluded from every predictive feature set',
        missing_policy='Retain observed numerical zeros; never fill blank yields or counts; all feature sets evaluated on identical complete-case observations and histories',
        hyperparameters='Fixed prior benchmark settings, no test-set tuning; all input/target scaling training-only; prediction clipped at zero',
        neural=get_config()|{'seeds':SEEDS,'input_normalization':'Per-channel mean and SD fitted only to training histories','input_features_per_step':'Depends on feature set'},
        observation_counts=observations.groupby('season').size().to_dict(),excluded_observation_counts=excluded.groupby('season').size().to_dict(),
        yield_counts=yields.groupby('season').size().to_dict(),prediction_rows=len(pred),seed_prediction_rows=len(seeds),fit_batches=len(fits),
        checks=dict(no_2425=True,equal_test_samples_all_models_and_feature_sets=True,chronological_within_fits=True,disjoint_cross_season_fits=True,no_duplicate_predictions=True,no_fabricated_labels=True),
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,torch=str(torch.__version__)),
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        code_sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [Path(__file__),ROOT/'canopy_temporal_benchmark/run_benchmark.py',ROOT/'canopy_temporal_benchmark/neural.py']],
        elapsed_seconds=time.time()-started)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps({k:summary[k] for k in ['observation_counts','yield_counts','prediction_rows','fit_batches','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__': main()
