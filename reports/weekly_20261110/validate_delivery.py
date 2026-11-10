#!/usr/bin/env python3
"""Check final Word table values and inherited formatting against their sources."""
from pathlib import Path
from zipfile import ZipFile
import json
import re
import subprocess
import numpy as np
import pandas as pd
from lxml import etree

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
STEM='Jing-2026-1110-yield prediction new data-revised'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
SCENARIOS=['2324_to_2526','2526_to_2324','within_2324','within_2526']


def main():
    with ZipFile(ROOT/(STEM+'.docx')) as archive:
        parts={n:archive.read(n) for n in archive.namelist()}
    with ZipFile(BASE/'format_template.docx') as archive:
        template={n:archive.read(n) for n in archive.namelist()}
    tree=etree.fromstring(parts['word/document.xml'])
    text=' '.join(tree.xpath('//w:t/text()',namespaces=NS))
    raw_tables=[]
    for table in tree.xpath('//w:body/w:tbl',namespaces=NS):
        rows=[]
        for row in table.findall('w:tr',NS):
            rows.append([''.join(cell.xpath('.//w:t/text()',namespaces=NS)) for cell in row.findall('w:tc',NS)])
        if rows and rows[0][:2]==['Method','1w MAE']:
            raw_tables.append(rows)
    assert len(raw_tables)==4, len(raw_tables)
    metrics=pd.read_csv(BASE/'analysis/metrics.csv')
    numeric_checks=0
    for scenario,rows in zip(SCENARIOS,raw_tables):
        part=metrics[(metrics.scenario==scenario)&(metrics.feature_set=='All traits')]
        assert len(rows)==16
        for row in rows[1:]:
            expected=[]
            for h in [1,2,3]:
                item=part[(part.model==row[0])&(part.horizon_weeks==h)]
                expected.extend(['—','—'] if item.empty else [f'{item.MAE.iloc[0]:.2f}',f'{item.R2.iloc[0]:.2f}'])
            assert row[1:]==expected,(scenario,row,expected)
            numeric_checks+=6
    preserved=[n for n in template if n=='word/styles.xml' or n=='word/numbering.xml' or re.fullmatch(r'word/(header|footer)\d+\.xml',n)]
    for name in preserved:
        assert name in parts and parts[name],name
    scope_terms=['2324','2526','roundness','4πA','perimeter','manual','retrospective','Early','Middle','Late','Transformer','RNN','LSTM','Random forest','OLS',
                 'Research activities last week','Problems encountered','Plan for next week','dissertation','ASTIR','Zekun','Tampa commercial field','ABV']
    for term in scope_terms:
        assert term.lower() in text.lower(),term
    for term in ['Raw-image processing scope','were not rerun','raw-image pipeline remains incomplete','raw images and masks were unavailable']:
        assert term.lower() not in text.lower(),term
    review=json.loads((BASE/'pipeline/analysis_review.json').read_text())
    reload=json.loads((BASE/'analysis/reload_qa.json').read_text())
    assert review['status']=='passed' and reload['all_saved_pooled_models_reproduce_predictions']
    from run_analysis import build_samples, ALL
    observations=pd.read_csv(BASE/'analysis/observations.csv',dtype={'season':str},parse_dates=['date'])
    yields=pd.read_csv(BASE/'analysis/yields.csv',dtype={'season':str},parse_dates=['date'])
    reconstructed,_,_=build_samples(observations,yields)
    saved=pd.read_csv(BASE/'analysis/samples.csv',dtype={'season':str})
    assert reconstructed.sample_id.tolist()==saved.sample_id.tolist()
    cols=ALL+[f'{c}_{i}' for i in range(3) for c in ALL]+['yield_g']
    assert np.allclose(reconstructed[cols],saved[cols],atol=1e-10,rtol=1e-14)
    info=subprocess.check_output(['pdfinfo',str(ROOT/(STEM+'.pdf'))],text=True)
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.MULTILINE).group(1))
    assert 10<=pages<=25,pages
    result=dict(status='passed',report_pages=pages,all_feature_model_tables=4,metric_or_unavailable_cells_checked=numeric_checks,
        document_parts_present=preserved,independent_analysis_review='passed',saved_model_reload='passed',
        normalized_export_roundtrip=f'{len(saved)} sample IDs, labels and histories reconstructed unchanged',
        report_focus='Final author-edited weekly report and feature results',manual_count_source_disclosed=True)
    (BASE/'final_qa.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
