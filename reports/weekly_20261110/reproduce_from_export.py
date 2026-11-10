#!/usr/bin/env python3
"""Reproduce models from the normalized observations and yields in this package.

The independent output directory protects the delivered original results.
Raw RGB, masks, Excel workbooks, and old-season data are not needed here.
"""
import argparse
from pathlib import Path
import pandas as pd
import run_analysis as benchmark


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(Path(__file__).resolve().parent/'analysis_reproduced'))
    args=parser.parse_args()
    frozen=Path(__file__).resolve().parent/'analysis'
    destination=Path(args.out).resolve()
    if destination==frozen:
        raise ValueError('Choose a separate output directory to protect the delivered results.')

    def load_exported_inputs():
        observations=pd.read_csv(frozen/'observations.csv',dtype={'season':str},parse_dates=['date'])
        yields=pd.read_csv(frozen/'yields.csv',dtype={'season':str},parse_dates=['date'])
        excluded=pd.read_csv(frozen/'excluded_observations.csv',dtype={'season':str},parse_dates=['date'])
        assert set(observations.season)==set(yields.season)=={'2324','2526'}
        return observations,yields,excluded,[frozen/'observations.csv',frozen/'yields.csv']

    benchmark.load_inputs=load_exported_inputs
    benchmark.OUT=destination
    benchmark.main()


if __name__=='__main__':main()
