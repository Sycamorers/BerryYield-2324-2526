#!/usr/bin/env python3
"""Reload every pooled cross-season bundle and reconcile its test predictions."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
import torch
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT))
from run_analysis import FEATURE_SETS, MODEL_INFO, NEURAL, SEEDS, arrays
from canopy_temporal_benchmark.neural import RecurrentRegressor, TransformerRegressor


def main():
    torch.set_num_threads(1)
    out=BASE/'analysis'
    samples=pd.read_csv(out/'samples.csv',dtype={'season':str})
    stored=pd.read_csv(out/'predictions.csv',dtype={'season':str})
    results=[]
    for mode,features in FEATURE_SETS.items():
        for source,target in [('2324','2526'),('2526','2324')]:
            scenario=f'{source}_to_{target}'
            for h in (1,2,3):
                test=samples[samples.season.eq(target)&samples.horizon_weeks.eq(h)]
                x=arrays(test,features)
                folder=out/'models'/mode.lower().replace(' + ','_').replace(' ','_')/scenario/f'h{h}'
                for name in MODEL_INFO:
                    if name in NEURAL:
                        predictions=[]
                        for seed in SEEDS:
                            bundle=torch.load(folder/f'{name}_{seed}.pt',map_location='cpu',weights_only=True)
                            model=TransformerRegressor(bundle['input_size']) if name=='Transformer' else RecurrentRegressor(name,bundle['input_size'])
                            model.load_state_dict(bundle['state_dict']);model.eval()
                            xt=torch.tensor((x-np.asarray(bundle['x_mean']))/np.asarray(bundle['x_scale']),dtype=torch.float32)
                            with torch.inference_mode():
                                ps=model(xt).numpy().astype(float)*bundle['y_scale']+bundle['y_mean']
                            predictions.append(np.maximum(ps,0))
                        prediction=np.mean(predictions,axis=0)
                    else:
                        bundle=joblib.load(folder/f'{name}.joblib')
                        if 'constant' in bundle: prediction=np.full(len(x),bundle['constant'])
                        else:
                            xt=x[:,-1,:] if bundle['input']=='current' else x.reshape(len(x),-1)
                            prediction=np.maximum(bundle['estimator'].predict(xt),0)
                    ref=stored[stored.feature_set.eq(mode)&stored.scenario.eq(scenario)&stored.horizon_weeks.eq(h)&stored.model.eq(name)].set_index('sample_id').loc[test.sample_id,'prediction'].to_numpy()
                    error=float(np.max(np.abs(ref-prediction)))
                    assert np.allclose(ref,prediction,rtol=1e-10,atol=1e-8),(mode,scenario,h,name,error)
                    results.append(dict(feature_set=mode,scenario=scenario,horizon_weeks=h,model=name,n=len(test),max_absolute_error=error))
    pd.DataFrame(results).to_csv(out/'reload_details.csv',index=False)
    result=dict(all_saved_pooled_models_reproduce_predictions=True,model_comparisons=len(results),prediction_values=sum(r['n'] for r in results),maximum_absolute_error=max(r['max_absolute_error'] for r in results))
    (out/'reload_qa.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
