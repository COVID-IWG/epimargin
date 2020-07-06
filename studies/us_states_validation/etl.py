#!python3 
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import requests

def import_and_clean_cases(save_path: Path) -> pd.DataFrame:
    '''
    Import and clean case data from covidtracking.com. 
    '''
    # Parameters for filtering raw df
    kept_columns   = ['date','state','positive','death']
    excluded_areas = set(['PR','MP','AS','GU','VI'])

    # Import and save result
    res = requests.get("https://covidtracking.com/api/v1/states/daily.json")
    df  = pd.read_json(res.text)
    df.to_csv(save_path/"covidtracking_cases.csv", index=False)
    
    # Exclude specific territories and features
    df = df[~df['state'].isin(excluded_areas)][kept_columns]

    # Format date properly
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format='%Y%m%d')

    # Calculate state change in positives/deaths
    df = df.sort_values(['state','date'])
    df['delta_positive'] = df.groupby(['state'])['positive'].transform(lambda x: x.diff()) 
    df['delta_death']    = df.groupby(['state'])['death'].transform(lambda x: x.diff()) 
    
    return df


def get_rt_live_data(save_path: Path) -> pd.DataFrame:
    '''
    Gets Rt estimates from Rt.live.
    '''
    # Parameters for filtering raw df
    kept_columns   = ['date','region','mean','lower_80','upper_80',
                      'infections','test_adjusted_positive']

    # Import and save as csv
    res = requests.get("https://d14wlfuexuxgcm.cloudfront.net/covid/rt.csv")
    df = pd.read_csv(StringIO(res.text))
    df.to_csv(save_path/"rtlive_estimates.csv", index=False)
    
    # Filter to just necessary features
    df = df[kept_columns]
    
    # Format date properly and rename columns
    df.loc[:,'date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.rename(columns={'region':'state','mean':'RR_pred_rtlive',
                            'lower_80':'RR_lower_rtlive', 'upper_80':'RR_upper_rtlive',
                            'test_adjusted_positive':'adj_positive_rtlive',
                            'infections':'infections_rtlive'}, inplace=True)
    return df
