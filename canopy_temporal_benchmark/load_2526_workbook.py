"""Extract measured Yield (g) from date tabs; preserve blanks as missing."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl


def load_workbook_yields(path):
    workbook=openpyxl.load_workbook(path,data_only=True,read_only=True)
    formula_book=openpyxl.load_workbook(path,data_only=False,read_only=True)
    records,audit=[],[]
    for sheet in workbook:
        if not re.fullmatch(r'\d{6}',sheet.title): continue
        date=pd.to_datetime(sheet.title,format='%y%m%d')
        rows=list(sheet.iter_rows(min_row=1,max_row=20,values_only=True))
        headers=rows[0]
        columns={j:str(v).strip().upper() for j,v in enumerate(headers) if re.fullmatch(r'[BM][1245][1-5]',str(v).strip().upper())}
        assert len(columns)==40,(sheet.title,len(columns))
        yr=[i for i,r in enumerate(rows) if str(r[0]).strip().lower()=='yield (g)']
        assert len(yr)==1,(sheet.title,yr)
        values=rows[yr[0]]; missing=zeros=valid=0
        formulas=list(formula_book[sheet.title].iter_rows(min_row=yr[0]+1,max_row=yr[0]+1,values_only=True))[0]
        for j,plot in columns.items():
            value=values[j]
            if value is None or str(value).strip()=='':
                if isinstance(formulas[j],str) and formulas[j].startswith('='):
                    raise ValueError(f'Uncached yield formula: {sheet.title} {plot}')
                missing+=1;continue
            value=float(value)
            assert np.isfinite(value) and value>=0,(sheet.title,plot,value)
            valid+=1;zeros+=int(value==0)
            records.append(dict(season='2526',date=date,plot=plot,yield_g=value,source_sheet=sheet.title,source_cell=f'{openpyxl.utils.get_column_letter(j+1)}{yr[0]+1}'))
        audit.append(dict(sheet=sheet.title,date=date,valid_yields=valid,explicit_zeros=zeros,missing_yields=missing,unit='g/plot'))
    workbook.close();formula_book.close()
    result=pd.DataFrame(records)
    assert len(result) and not result.duplicated(['season','date','plot']).any()
    return result,pd.DataFrame(audit)
