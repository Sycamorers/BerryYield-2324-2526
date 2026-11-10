#!/usr/bin/env python3
"""Independent numerical/protocol review of the two-season benchmark exports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
OUT = BASE / "analysis"
DEST = Path(__file__).resolve().parent / "analysis_review.json"
FEATURES = ["canopy_size_pixels", "mean_roundness", "plant_count", "flower", "green", "white", "pink"]
MODELS = {"Zero", "Train mean", "Train median", "OLS", "Quadratic OLS", "Spline ridge", "Random forest", "SVR", "Gradient boosting", "Lag OLS", "Lag ridge", "Lag random forest", "RNN", "LSTM", "Transformer"}


def main():
    required = ["summary.json", "fit_audit.csv", "predictions.csv", "metrics.csv", "stage_metrics.csv", "unavailable.csv", "seed_predictions.csv"]
    missing = [x for x in required if not (OUT/x).exists()]
    if missing:
        print("Pending analysis files:", ", ".join(missing))
        return 2
    checks, issues = {}, []
    def check(name, value, details=None):
        checks[name] = {"passed": bool(value)}
        if details is not None: checks[name]["details"] = details
        if not value: issues.append(name)

    summary = json.loads((OUT / "summary.json").read_text())
    samples = pd.read_csv(OUT/"samples.csv", dtype={"season": str}, parse_dates=["canopy_date", "yield_date"])
    observations = pd.read_csv(OUT/"observations.csv", dtype={"season": str}, parse_dates=["date"])
    yields = pd.read_csv(OUT/"yields.csv", dtype={"season": str}, parse_dates=["date"])
    pred = pd.read_csv(OUT/"predictions.csv", dtype={"season": str}, parse_dates=["canopy_date", "yield_date"])
    fits = pd.read_csv(OUT/"fit_audit.csv", dtype={"training_seasons":str,"test_seasons":str}, parse_dates=["train_latest_target", "test_first_origin"])
    unavailable = pd.read_csv(OUT/"unavailable.csv")
    sample_by_id = samples.set_index("sample_id")
    check("only_2324_2526_data", all(set(t.season) <= {"2324", "2526"} for t in [samples, observations, yields, pred]))
    check("exactly_15_expected_models", set(pred.model) == MODELS)
    check("exactly_six_feature_sets", set(pred.feature_set) == set(summary["feature_sets"]) and len(summary["feature_sets"]) == 6)
    check("no_duplicate_prediction_keys", not pred.duplicated(["feature_set", "scenario", "model", "sample_id"]).any())
    check("predictions_finite_nonnegative", np.isfinite(pred.prediction).all() and pred.prediction.ge(0).all())
    check("yield_targets_unchanged", np.array_equal(pred.yield_g.to_numpy(), sample_by_id.loc[pred.sample_id].yield_g.to_numpy()))
    check("strictly_future_targets", samples.yield_date.gt(samples.canopy_date).all())

    source_hashes = []
    for source in summary["sources"] + summary["code_sources"]:
        path = ROOT/source["path"]
        source_hashes.append(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"])
    check("source_and_code_hashes_match", all(source_hashes), len(source_hashes))
    check("no_2425_source_paths", all("2425" not in x["path"] for x in summary["sources"]))

    # Match source yield cells directly, independently of the experiment loader.
    source_yields, source_counts = [], []
    for path in sorted((ROOT/"data/real_data/2324_GNV_processed").glob("*/consolidated_summary_with_yield.csv")):
        table = pd.read_csv(path, index_col=0)
        if "Yield (g)" not in table.index: continue
        for plot, value in table.loc["Yield (g)"].items():
            if pd.notna(value): source_yields.append(("2324", pd.to_datetime(path.parent.name,format="%y%m%d"), str(plot).strip().upper(), float(value)))
    count_mapping={"FL":"flower","G":"green","W":"white","P":"pink","R":"red"}
    for path in sorted((ROOT/"data/real_data/2324_GNV_processed/counting_yield").glob("*.csv")):
        table=pd.read_csv(path,index_col=0)
        for plot in table.columns:
            rec={"season":"2324","date":pd.to_datetime(path.stem,format="%y%m%d"),"plot":str(plot).strip().upper()}
            rec.update({new:pd.to_numeric(table.loc[old,plot],errors="coerce") for old,new in count_mapping.items()})
            source_counts.append(rec)
    import openpyxl
    workbook = openpyxl.load_workbook(ROOT/"check-in & data-col 2526.xlsx", data_only=True, read_only=True)
    for sheet in workbook:
        if len(sheet.title) != 6 or not sheet.title.isdigit(): continue
        rows = list(sheet.iter_rows(min_row=1,max_row=7,values_only=True))
        for j, plot in enumerate(rows[0]):
            if plot is not None and str(plot).strip().upper() in {f"{p}{r}{c}" for p in "BM" for r in [1,2,4,5] for c in range(1,6)}:
                val=rows[6][j]
                if val is not None: source_yields.append(("2526",pd.to_datetime(sheet.title,format="%y%m%d"),str(plot).strip().upper(),float(val)))
                rec={"season":"2526","date":pd.to_datetime(sheet.title,format="%y%m%d"),"plot":str(plot).strip().upper()}
                rec.update({new:float(rows[i][j]) if rows[i][j] is not None else np.nan for i,new in enumerate(count_mapping.values(),1)})
                source_counts.append(rec)
    workbook.close()
    source_yields = pd.DataFrame(source_yields, columns=["season","date","plot","yield_g"])
    matched_yields = yields.merge(source_yields,on=["season","date","plot"],suffixes=("_output","_source"),how="outer",indicator=True)
    check("yields_match_original_csv_and_workbook_cells", matched_yields["_merge"].eq("both").all() and matched_yields.yield_g_output.eq(matched_yields.yield_g_source).all(), len(matched_yields))
    counts_output=pd.read_csv(BASE/"correlations/counts_harmonized.csv",dtype={"season":str},parse_dates=["date"])
    compared_counts=counts_output.merge(pd.DataFrame(source_counts),on=["season","date","plot"],how="outer",suffixes=("_output","_source"),indicator=True,validate="one_to_one")
    counts_equal=compared_counts["_merge"].eq("both").all()
    for f in count_mapping.values():
        counts_equal=counts_equal and np.allclose(compared_counts[f+"_output"],compared_counts[f+"_source"],equal_nan=True,rtol=0,atol=0)
    check("manual_counts_match_original_csv_and_workbook_cells_including_blanks",counts_equal,len(compared_counts))

    history_ok, stages_ok, match_ok = [], [], []
    for (season, plot), group in observations.groupby(["season", "plot"]):
        group=group.sort_values("date")
        for row in samples[(samples.season==season)&(samples["plot"]==plot)].to_dict("records"):
            hist=group[group.date.le(row["canopy_date"])].tail(3)
            arr=np.pad(hist[FEATURES].to_numpy(), ((3-len(hist),0),(0,0)),mode="edge")
            actual=np.array([[row[f"{f}_{i}"] for f in FEATURES] for i in range(3)])
            history_ok.append(np.allclose(arr,actual,atol=1e-9,rtol=1e-12) and hist.date.max()==row["canopy_date"])
    for season, part in samples.groupby("season"):
        dates=sorted(yields[yields.season==season].date.unique())
        mapping={pd.Timestamp(d):phase for phase,block in zip(["Early","Middle","Late"],np.array_split(dates,3)) for d in block}
        for row in part.to_dict("records"):
            stages_ok.append(mapping[row["yield_date"]]==row["stage"])
            expected=row["canopy_date"]+pd.Timedelta(days=7*row["horizon_weeks"])
            candidates=[pd.Timestamp(d) for d in dates if pd.Timestamp(d)>row["canopy_date"] and abs((pd.Timestamp(d)-expected).days)<=2]
            nearest=min(candidates,key=lambda d:(abs((d-expected).days),d))
            match_ok.append(nearest==row["yield_date"])
    check("all_three_step_histories_use_only_available_observations",all(history_ok),len(history_ok))
    check("target_dates_are_nearest_within_two_days",all(match_ok),len(match_ok))
    check("phase_assignment_matches_fixed_harvest_date_thirds",all(stages_ok),len(stages_ok))

    within_ok, cross_ok, fit_predictions_ok, baseline_ok, id_lengths_ok, phase_fit_ok = [], [], [], [], [], []
    pred_by_fit = {key:g for key,g in pred.groupby("fit_id")}
    for row in fits.to_dict("records"):
        a=row["train_sample_ids"].split("|");b=row["test_sample_ids"].split("|")
        tr=sample_by_id.loc[a];te=sample_by_id.loc[b]
        id_lengths_ok.append(len(a)==row["n_train"] and len(b)==row["n_test"] and set(a).isdisjoint(b))
        is_within="within_" in row["scenario"]
        if is_within:
            within_ok.append(tr.yield_date.max()<te.canopy_date.min() and tr.yield_date.nunique()>=2 and set(tr.season)==set(te.season))
        else:
            direction=row["scenario"].replace("stage_","")
            source,target=direction.split("_to_")
            cross_ok.append(set(tr.season)=={source} and set(te.season)=={target} and set(tr.season).isdisjoint(set(te.season)))
        if row["scenario"].startswith("stage_"):
            phase_fit_ok.append(te.stage.nunique()==tr.stage.nunique()==1 and set(te.stage)==set(tr.stage))
        p=pred_by_fit[row["fit_id"]]
        sets=[set(x.sample_id) for _,x in p.groupby("model")]
        fit_predictions_ok.append(len(sets)==15 and all(s==set(b) for s in sets))
        for name,expected in [("Zero",0.0),("Train mean",tr.yield_g.mean()),("Train median",tr.yield_g.median())]:
            baseline_ok.append(np.allclose(p[p.model==name].prediction,expected,rtol=1e-12,atol=1e-9))
    check("fit_ids_counts_and_disjoint_samples",all(id_lengths_ok),len(id_lengths_ok))
    check("within_training_labels_strictly_precede_origin",all(within_ok),len(within_ok))
    check("cross_training_uses_correct_held_out_season_in_both_directions",all(cross_ok),len(cross_ok))
    check("phase_specific_training_uses_only_matching_phase",all(phase_fit_ok),len(phase_fit_ok))
    check("all_fit_predictions_match_audited_test_ids",all(fit_predictions_ok),len(fit_predictions_ok))
    check("all_baselines_equal_training_only_statistics",all(baseline_ok),len(baseline_ok))

    shared=[]
    for _,g in pred.groupby(["scenario","horizon_weeks"]):
        sets=[set(p.sample_id) for _,p in g.groupby(["feature_set","model"])]
        shared.append(len(sets)==90 and all(s==sets[0] for s in sets))
    check("all_models_and_feature_sets_share_identical_test_samples",all(shared),len(shared))

    metric_results=[]
    for filename,keys in [("metrics.csv",["feature_set","scenario","horizon_weeks","model","family"]),
            ("stage_metrics.csv",["feature_set","scenario","horizon_weeks","stage","model"])]:
        exported=pd.read_csv(OUT/filename).set_index(keys)
        valid=[]; maxdiff={k:0. for k in ["MAE","RMSE","R2","Bias"]}
        for key,g in pred.groupby(keys):
            record=exported.loc[key]
            y=g.yield_g.to_numpy();yhat=g.prediction.to_numpy();err=yhat-y
            denom=np.square(y-y.mean()).sum()
            calculated={"MAE":np.abs(err).mean(),"RMSE":np.sqrt(np.square(err).mean()),
                "R2":1-np.square(err).sum()/denom if denom>0 else np.nan,"Bias":err.mean()}
            valid.append(int(record["n"])==len(g) and int(record["target_dates"])==g.yield_date.nunique() and int(record["origin_dates"])==g.canopy_date.nunique())
            for field,value in calculated.items():
                valid.append(np.isclose(record[field],value,rtol=1e-10,atol=1e-8,equal_nan=True))
                if np.isfinite(value):maxdiff[field]=max(maxdiff[field],float(abs(record[field]-value)))
        result={"file":filename,"groups":len(exported),"passed":all(valid),"max_absolute_differences":maxdiff}
        metric_results.append(result)
    check("MAE_RMSE_R2_bias_independently_recomputed",all(x["passed"] for x in metric_results),metric_results)

    seeds=pd.read_csv(OUT/"seed_predictions.csv")
    keys=["feature_set","scenario","model","sample_id"]
    ensemble=seeds.groupby(keys).prediction.agg(["mean","size"]).reset_index()
    matched=ensemble.merge(pred[pred.model.isin(["RNN","LSTM","Transformer"])][keys+["prediction"]],on=keys,validate="one_to_one")
    check("neural_predictions_equal_three_clipped_seed_mean",matched["size"].eq(3).all() and np.allclose(matched["mean"],matched.prediction,rtol=1e-12,atol=1e-9),len(matched))

    feasibility=[]
    for (season,h),g in samples.groupby(["season","horizon_weeks"]):
        allowed=set()
        for origin,te in g.groupby("canopy_date"):
            if g[g.yield_date.lt(origin)].yield_date.nunique()>=2:allowed.update(te.sample_id)
        actual=set(pred[(pred.scenario==f"within_{season}")&(pred.horizon_weeks==h)].sample_id)
        feasibility.append({"season":season,"horizon_weeks":int(h),"eligible_test_pairs":len(allowed),"exported_test_pairs":len(actual),"equal":allowed==actual})
    check("within_coverage_exactly_matches_chronological_feasibility",all(x["equal"] for x in feasibility),feasibility)
    unsupported=unavailable[(unavailable.scenario=="within_2324")&(unavailable.horizon_weeks==3)]
    check("within_2324_h3_explicitly_unavailable",len(unsupported)>0 and not len(pred[(pred.scenario=="within_2324")&(pred.horizon_weeks==3)]),len(unsupported))

    result={"status":"passed" if not issues else "issues_found", "review_method":"Independent NumPy recomputation and sample/fit audit; does not call runner metric or sample-building helpers",
        "prediction_rows":len(pred),"fit_batches":len(fits),"checks":checks,"issues":issues,
        "interpretation_notes":["Reverse 2526 to 2324 is retrospective season generalization, not chronological deployment",
            "Calendar phases use thirds of observed target harvest dates and are retrospective, not measured phenology",
            "Manual fruit/flower counts are harmonized across both seasons; tracked plant count is not ground-truth census",
            "Current experiment ranks are descriptive test outcomes; they are not untouched validation after selecting the best method",
            "Complete cases provide equal input comparisons; canopy-only coverage in this benchmark is intentionally restricted to shared available count dates",
            "Irregular histories use the latest three observations, not three equally spaced weeks",
            "The original images and masks are unavailable; per-plant metadata consistency does not establish segmentation accuracy"]}
    DEST.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":result["status"],"checks":len(checks),"issues":issues,"prediction_rows":len(pred),"fit_batches":len(fits)},indent=2))
    return 0 if not issues else 1


if __name__=="__main__":
    raise SystemExit(main())
