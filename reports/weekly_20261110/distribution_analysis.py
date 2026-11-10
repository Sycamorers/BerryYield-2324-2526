#!/usr/bin/env python3
"""Describe every measured input and the harvest label in the two-season study."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'correlations'
SOURCES = [
    ('Image features', BASE / 'pipeline/traits_reprocessed.csv',
     [('canopy_size_pixels', 'Canopy mask area (pixels)'),
      ('mean_roundness', 'Mean plant roundness'),
      ('plant_count', 'Plant tracks')]),
    ('Field counts', OUT / 'counts_harmonized.csv',
     [(c, {'flower':'FL', 'green':'G', 'white':'W', 'pink':'P', 'red':'R'}[c]) for c in ('flower', 'green', 'white', 'pink', 'red')]),
    ('Harvest yield', BASE / 'analysis/yields.csv', [('yield_g', 'Yield (g/plot/harvest)')]),
]
COLORS = {'2324': '#246C9C', '2526': '#D3783B'}


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    frames = {}
    for source, path, variables in SOURCES:
        frame = pd.read_csv(path, dtype={'season': str})
        assert set(frame.season) == {'2324', '2526'}, path
        frames[source] = frame
        for col, label in variables:
            for season in ('2324', '2526'):
                values = pd.to_numeric(frame.loc[frame.season.eq(season), col], errors='coerce')
                good = values.dropna()
                assert len(good) and np.isfinite(good).all() and good.ge(0).all()
                rows.append(dict(source=source, variable=col, label=label, season=season,
                                 total_rows=len(values), n=len(good), missing=int(values.isna().sum()),
                                 zeros=int(good.eq(0).sum()), mean=good.mean(), sd=good.std(),
                                 min=good.min(), q25=good.quantile(.25), median=good.median(),
                                 q75=good.quantile(.75), max=good.max()))
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / 'distributions.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    groups = [
        ('image_yield_distributions', SOURCES[0:1] + SOURCES[2:3], 2),
        ('field_count_distributions', SOURCES[1:2], 3),
    ]
    for filename, sources, ncols in groups:
        pairs = [(source, col, label) for source, _, variables in sources for col, label in variables]
        nrows = int(np.ceil(len(pairs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.2*nrows), squeeze=False,
                                 constrained_layout=True)
        for ax, (source, col, label) in zip(axes.flat, pairs):
            frame = frames[source]
            for season in ('2324', '2526'):
                values = pd.to_numeric(frame.loc[frame.season.eq(season), col], errors='coerce').dropna()
                ax.hist(values, bins=22, density=True, alpha=.46, color=COLORS[season],
                        label=f'{season} (n={len(values)})')
                ax.axvline(values.median(), color=COLORS[season], lw=1.4)
            ax.set_title(label, loc='left', fontweight='bold')
            ax.set_ylabel('Density')
            ax.legend(frameon=False, fontsize=8)
            ax.grid(axis='y', alpha=.15)
        for ax in list(axes.flat)[len(pairs):]:
            ax.axis('off')
        fig.suptitle('Feature distributions by season (lines: medians)', fontsize=12)
        for ext in ('png', 'pdf'):
            fig.savefig(OUT / f'{filename}.{ext}', dpi=200)
        plt.close(fig)
    print(f'Wrote {len(stats)} distribution summaries and two figures')


if __name__ == '__main__':
    main()
