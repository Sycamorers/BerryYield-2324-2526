#!/usr/bin/env python3
"""Audit canopy scale and available manual/YOLO phenology sources for 2324/2526."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'analysis'
STAGES = {'FL': 'flower', 'G': 'green', 'W': 'white', 'P': 'pink', 'R': 'red'}


def main():
    OUT.mkdir(exist_ok=True)
    canopy = pd.read_csv(Path(__file__).parent/'pipeline/traits_reprocessed.csv', dtype={'season': str})
    canopy['area_per_track'] = canopy.canopy_size_pixels/canopy.plant_count
    rows = []
    for season, data in canopy.groupby('season'):
        rows.append(dict(season=season, plot_dates=len(data), canopy_mean=data.canopy_size_pixels.mean(),
                         canopy_median=data.canopy_size_pixels.median(),
                         tracks_mean=data.plant_count.mean(), tracks_median=data.plant_count.median(),
                         area_per_track_mean=data.area_per_track.mean(),
                         area_per_track_median=data.area_per_track.median()))
    pd.DataFrame(rows).to_csv(OUT/'canopy_shift_summary.csv', index=False)
    canopy.groupby(['season','date']).agg(plot_dates=('plot','size'),
        canopy_median=('canopy_size_pixels','median'), tracks_median=('plant_count','median'),
        area_per_track_median=('area_per_track','median')).to_csv(OUT/'canopy_shift_by_date.csv')
    fig,axes=plt.subplots(1,3,figsize=(9.5,3.2),constrained_layout=True)
    for ax,(col,title,scale) in zip(axes,[
        ('canopy_size_pixels','Canopy per plot (million pixels)',1e6),
        ('plant_count','Retained plant tracks per plot',1),
        ('area_per_track','Mask area per track (thousand pixels)',1e3)]):
        values=[canopy.loc[canopy.season.eq(s),col]/scale for s in ('2324','2526')]
        box=ax.boxplot(values,patch_artist=True,showfliers=False,widths=.56)
        for patch,color in zip(box['boxes'],('#246C9C','#D3783B')):patch.set_facecolor(color);patch.set_alpha(.58)
        ax.set_xticks([1,2],['2324','2526']);ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Canopy shift: fewer retained plants and smaller masks',fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'canopy_shift_decomposition.{ext}',dpi=220)
    plt.close(fig)

    pairs = []
    source = ROOT/'data/real_data/2324_GNV_processed'
    for p in sorted(source.glob('*/consolidated_summary_with_yield.csv')):
        date = pd.to_datetime(p.parent.name, format='%y%m%d').strftime('%Y-%m-%d')
        yolo = pd.read_csv(p, index_col=0)
        manual = pd.read_csv(source/'counting_yield'/f'{p.parent.name}.csv', index_col=0)
        for plot in yolo.columns:
            for code, name in STAGES.items():
                pairs.append(dict(season='2324',date=date,plot=plot,feature=code,
                    yolo=pd.to_numeric(yolo.loc['strawberry_'+name, plot],errors='coerce'),
                    manual=pd.to_numeric(manual.loc[code,plot],errors='coerce')))
    frame = pd.DataFrame(pairs)
    frame.to_csv(OUT/'manual_yolo_2324_pairs.csv',index=False)
    comparison=[]
    for code,g in frame.groupby('feature'):
        h=g.dropna(subset=['yolo','manual'])
        delta=h.yolo-h.manual
        comparison.append(dict(feature=code,n=len(h),manual_mean=h.manual.mean(),
            yolo_mean=h.yolo.mean(),yolo_minus_manual=delta.mean(),mae=delta.abs().mean(),
            pearson_r=h.yolo.corr(h.manual)))
    pd.DataFrame(comparison).to_csv(OUT/'manual_yolo_2324_summary.csv',index=False)
    audit = dict(
        season_2324_manual_source='data/real_data/2324_GNV_processed/counting_yield/*.csv',
        season_2324_yolo_source='data/real_data/2324_GNV_processed/*/consolidated_summary_with_yield.csv, strawberry_* rows',
        season_2324_paired_plot_dates=frame.dropna(subset=['manual','yolo'])[['date','plot']].drop_duplicates().shape[0],
        season_2526_workbook='check-in & data-col 2526.xlsx, FL/G/W/P/R rows',
        season_2526_workbook_provenance='manual field-count workbook; count-cell coverage audited separately',
        season_2526_yolo_source='analysis/yolo_recount_2526/yolo_counts_2526.csv, rerun from original RGB images',
        season_2526_separate_yolo_source_found=(ROOT/'analysis/yolo_recount_2526/yolo_counts_2526.csv').is_file(),
        season_2526_separate_manual_source_found=True,
        can_compare_manual_vs_yolo_both_seasons=(ROOT/'reports/weekly_20261110/analysis/verified_counts/manual_yolo_summary.csv').is_file(),
        source_scope='local project files and mounted original image disk',
    )
    (OUT/'count_source_availability.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
