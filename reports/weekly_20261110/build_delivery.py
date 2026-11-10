#!/usr/bin/env python3
"""Create the results workbook, short briefing notes and self-contained ZIP."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import hashlib
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
STEM='Jing-2026-1110-yield prediction new data-revised'
ARCHIVE=ROOT/'BerryYield_2324_2526_20261110.zip'


def main():
    tables={
        'Model metrics':BASE/'analysis/metrics.csv',
        'Early middle late':BASE/'analysis/stage_metrics.csv',
        'Feature ablation':BASE/'analysis/ablation.csv',
        'Correlations':BASE/'correlations/correlations.csv',
        'Distributions':BASE/'correlations/distributions.csv',
        'Same sample transfer within':BASE/'analysis/matched_transfer_within_metrics.csv',
        'Same sample phase training':BASE/'analysis/matched_phase_training_metrics.csv',
        'Model winners descriptive':BASE/'analysis/winners.csv',
        'Neural seeds':BASE/'analysis/seed_metrics.csv',
        'Date cluster bootstrap':BASE/'analysis/paired_mae_bootstrap.csv',
        'Coverage by season':BASE/'pipeline/coverage_by_season.csv',
        'Coverage by date':BASE/'pipeline/coverage_by_date.csv',
        'Sample coverage':BASE/'analysis/sample_coverage.csv',
        'Harvest phase boundaries':BASE/'analysis/stage_boundaries.csv',
        'Feature range shifts':BASE/'analysis/input_ranges.csv',
        'Unavailable evaluations':BASE/'analysis/unavailable.csv',
        'Workbook yield audit':BASE/'analysis/workbook_yield_audit.csv',
    }
    with pd.ExcelWriter(BASE/'results.xlsx',engine='openpyxl') as writer:
        for name,path in tables.items():
            frame=pd.read_csv(path,dtype={'season':str})
            frame.to_excel(writer,sheet_name=name,index=False)
            sheet=writer.book[name]
            sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
            sheet.row_dimensions[1].height=32
            for cell in sheet[1]:
                cell.font=Font(name='Calibri',bold=True,color='FFFFFF')
                cell.fill=PatternFill('solid',fgColor='315F74')
                cell.alignment=Alignment(wrap_text=True,vertical='center')
            for j,col in enumerate(frame.columns,1):
                width=min(36,max(13,len(str(col))+2,int(frame[col].astype(str).str.len().quantile(.8))+2 if len(frame) else 13))
                sheet.column_dimensions[get_column_letter(j)].width=width
                if pd.api.types.is_numeric_dtype(frame[col]):
                    for row in sheet.iter_rows(min_row=2,min_col=j,max_col=j):
                        row[0].number_format='0.000' if pd.api.types.is_float_dtype(frame[col]) else '0'

    summary=json.loads((BASE/'analysis/summary.json').read_text())
    metrics=pd.read_csv(BASE/'analysis/metrics.csv')
    scen=[('2324_to_2526','2324训练 → 2526验证'),('2526_to_2324','2526训练 → 2324验证'),('within_2324','2324季内向前验证'),('within_2526','2526季内向前验证')]
    findings=[]
    for scenario,label in scen:
        part=metrics[(metrics.scenario==scenario)&(metrics.feature_set=='All traits')&(metrics.horizon_weeks==1)&(~metrics.model.isin(['Zero','Train mean','Train median']))]
        best=part.loc[part.MAE.idxmin()]
        findings.append(f'{label}：{best.model}，MAE={best.MAE:.1f} g/plot，R²={best.R2:.2f}，n={int(best.n)}，{int(best.target_dates)}个收获日。')
    notes='''2026-11-10 汇报提纲（对应英文完整报告）

本周完成
修改并提交 dissertation；处理数据得到 roundness，并重新开发产量预测模型。向 Zekun 移交 ASTIR 使用方式并完成测试，测试中未发现问题。检查 field，预约 Tampa commercial field 的行程与时间；完成 ABV 当前测试，未发现问题。

下周计划
到 Tampa commercial field 测试 ABV，采集数据并分析结果。

Roundness 公式
单株 Rᵢ = 4πAᵢ/Pᵢ²；Aᵢ 是植株 mask 面积，Pᵢ 是完整轮廓周长（含孔洞边界）。小区日期圆度是 n 株保留植株圆度的等权平均：(1/n)ΣᵢRᵢ。

1. 研究范围
只使用2324和2526。2425不参与训练、相关性或验证。预测目标是某次实际收获的小区产量（g/plot），不是全季总产量；提前1、2、3周。

2. 数据重处理成果
对2324、2526两季共13,353条逐株记录统一重算圆度，并通过pipeline汇总得到862条plot-date记录。canopy为逐株mask像素面积之和；圆度为4πA/P²后逐株等权平均；plant_count是保留轨迹数。

3. 怎样统一counting
两季花、绿果、白果、粉果都使用同口径人工实测计数，并加入pipeline植株轨迹数。计数来源和单位按季统一。红果用于相关性描述。

4. 相关性讲什么
一周提前量下，2324 canopy r=0.325、roundness r=-0.070；2526 canopy r=0.113、roundness r=-0.339。2526白果/粉果计数r=0.447/0.437。去掉同一收获日的共同趋势后，2526 roundness r=-0.196，因此不能把整体相关性当因果效应。

4a. 全部测量值的分布
报告提供 canopy、roundness、植株轨迹数、人工花/绿/白/粉/红果计数与实测产量的两季分布图。附表给出每季有效数、缺失数、零值数、中位数和四分位区间。分布图使用各数据源的有效记录，相关性使用日期匹配后的样本，两者样本数不同。

5. 模型怎样公平比较
15种方法覆盖常数基线、传统统计、机器学习、历史回归、RNN/LSTM/Transformer。6种特征组合使用相同测试样本与相同历史。神经模型固定3个种子、80轮；预处理只拟合训练数据，不利用测试集调参。
季内使用严格向前验证：所有训练产量日期早于预测时点，至少已有2个完成收获日。跨季做2324→2526与2526→2324；反向是回顾性迁移测试。

6. 一周、全部特征的描述性最低MAE模型
'''+ '\n'.join(findings)+'''
这些是查看评估结果后的排名，不代表已经选好一个可部署且独立验证过的最佳模型。负R²照实保留；应同时看基线、MAE、RMSE和偏差。

7. 前中后季怎样解释
按每季已观察的收获日期三等分，分别报告同一个全季模型在各阶段的误差，也对比单独按阶段训练。阶段是回顾性日历分段。季内早季和2324三周提前量没有足够历史标签，明确标不可评估，不用未来标签补结果。

8. 跨季节结果怎样解读
2324平均canopy像素面积约为2526的9.35倍；相机尺度和追踪差异可能影响迁移。2526最后一次扫描轨迹数偏高，解释跨季节结果时应结合这些测量差异。当前结果识别了值得继续验证的性状关系与模型。
'''
    (BASE/'汇报提纲.txt').write_text(notes,encoding='utf-8')
    readme=f'''BerryYield: unified 2324 / 2526 report, 10 November 2026

先阅读 / Start here
- {STEM}.pdf: complete readable report, following the earlier weekly-report order (completed work, problems, next-week plan, then detailed model development).
- {STEM}.docx: editable report.
- reports/weekly_20261110/汇报提纲.txt: concise Chinese speaking notes.
- reports/weekly_20261110/results.xlsx: all comparison tables in one workbook.
- reports/weekly_20261110/analysis/: sample-level predictions, metrics, model bundles and fit audits.
- reports/weekly_20261110/correlations/: harmonized manual counts, source-cell audit and correlation figures/tables.
- reports/weekly_20261110/correlations/distributions.csv and distribution figures: all measured predictors and harvest-yield distributions by season, including missing and zero counts.
- reports/weekly_20261110/pipeline/: recomputed plant/plot traits, source hashes, coverage and processing QA.
- reports/weekly_20261110/figures/: PNG and exportable PDF figures.

范围 / Scope
Only 2324 and 2526 are used. No 2425 measurement or label files are included in this package.
Per-plant roundness recalculation and plot-level trait aggregation cover 862 plot-dates and 13,353 plant records. The report presents the image-to-trait workflow, trait definitions and resulting measurements.
Flower/green/white/pink counts use harmonized manual field observations in both seasons. Image-derived inputs comprise canopy, roundness and retained plant-track count.
Yields are grams per plot per recorded harvest; blanks are missing, numeric zero is valid. Models predict 1/2/3 weeks ahead using nearest observed harvest within two days.
Both transfer directions and both chronological within-season protocols are included. The reverse fold is retrospective. Unsupported early/phase fits and within-2324 three-week forecasts are explicitly unavailable.
Per-model rankings and displayed winners are exploratory; the test results were inspected. Date clusters are few. Full-season totals and deployment reliability are not established.

复现 / Reproduction
Extract this ZIP and enter its BerryYield_2324_2526_20261110 directory. Python dependencies: numpy, pandas, scipy, scikit-learn, torch, matplotlib, openpyxl, Pillow, lxml, joblib, threadpoolctl. Exact recorded versions are in analysis/summary.json.

To refit directly from the included normalized inputs without image files or Excel:
  python reports/weekly_20261110/reproduce_from_export.py
This writes analysis_reproduced/ and protects the delivered original analysis/.

To reproduce trait processing and model comparisons from the included per-plant records and field measurements:
  python reports/weekly_20261110/reprocess_pipeline.py --reference-csv reports/weekly_20261110/pipeline/source_reference_2324_2526.csv
  python reports/weekly_20261110/correlation_analysis.py --counts-only
  python reports/weekly_20261110/run_analysis.py
  python reports/weekly_20261110/correlation_analysis.py
  python reports/weekly_20261110/distribution_analysis.py
  python reports/weekly_20261110/verify_saved_models.py
The delivered revised Word/PDF report is the final presentation copy. Its opening preserves the author's edits. The model tables and figures are reproducible from the exported analysis files; build_report.py is an older analytical draft builder, not the final report generator.
The historical shared Python helpers are included as dependencies. Run the designated 20261110 entry points above for this two-season study.

审计 / Audits
analysis/summary.json: parameters, versions, source hashes and core assertions.
pipeline/analysis_review.json: independent metric, split, baseline, source and seed-ensemble checks.
analysis/reload_qa.json: saved model predictions reconciled against the delivered evaluation outputs.
report_qa.json and final_qa.json: report scope and final delivery checks.
SHA256SUMS.txt inside the ZIP authenticates its included files; SOURCE_SNAPSHOT.json lists the included source files.

Original run: {summary['prediction_rows']:,} ensemble/reference prediction rows; {summary['fit_batches']} feature/scenario fit batches; {summary['elapsed_seconds']:.1f} seconds recorded.
'''
    (BASE/'README.txt').write_text(readme,encoding='utf-8')
    sources=[]
    for season in ['2324','2526']:
        sources.extend(sorted((ROOT/'canopy_pipeline/final_results/per_plot'/season).glob('*/*.json')))
    sources.extend(sorted((ROOT/'data/real_data/2324_GNV_processed').glob('*/consolidated_summary_with_yield.csv')))
    sources.extend(sorted((ROOT/'data/real_data/2324_GNV_processed/counting_yield').glob('*.csv')))
    sources.append(ROOT/'check-in & data-col 2526.xlsx')
    dependencies=[ROOT/p for p in ['README.md','canopy_pipeline/run_canopy_pipeline.py','canopy_temporal_benchmark/__init__.py','canopy_temporal_benchmark/run_benchmark.py','canopy_temporal_benchmark/neural.py','canopy_temporal_benchmark/load_2526_workbook.py','reports/build_canopy_report.py','reports/build_weekly_20261027.py']]
    files=[ROOT/(STEM+'.docx'),ROOT/(STEM+'.pdf')]+dependencies+sources
    for path in BASE.rglob('*'):
        if path.parent==BASE/'pipeline' and path.stem=='pipeline_scope':
            continue  # The report uses the current image_to_trait_workflow figures.
        if path.is_file() and '__pycache__' not in path.parts and 'qa_pages' not in path.parts and not path.name.endswith('.pyc'):
            files.append(path)
    files=sorted(set(files))
    assert all(p.exists() for p in files)
    source_snapshot=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]
    prefix='BerryYield_2324_2526_20261110/'
    checksums=[]
    with ZipFile(ARCHIVE,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            name=str(path.relative_to(ROOT))
            archive.write(path,prefix+name)
            checksums.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+name)
        archive.writestr(prefix+'README.txt',readme)
        archive.writestr(prefix+'SOURCE_SNAPSHOT.json',json.dumps(source_snapshot,indent=2))
        archive.writestr(prefix+'SHA256SUMS.txt','\n'.join(checksums)+'\n')
    with ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
    print(json.dumps(dict(archive=str(ARCHIVE),files=len(files),bytes=ARCHIVE.stat().st_size,results_workbook=str(BASE/'results.xlsx')),indent=2))


if __name__=='__main__':main()
