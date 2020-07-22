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


def get_adaptive_estimates(path: Path) -> pd.DataFrame:
    
    # Parameters for filtering raw df
    kept_columns   = ['date','state','RR_pred','RR_CI_lower','RR_CI_upper','T_pred',
                      'T_CI_lower','T_CI_upper','new_cases_ts','anamoly']

    # Import and subset columns
    df = pd.read_csv(path/"adaptive_estimates.csv")
    df = df[kept_columns]
    
    # Format date properly and return
    df.loc[:,'date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    return df


def get_new_rt_live_estimates(path: Path) -> pd.DataFrame:
    
    # Parameters for filtering raw df
    kept_columns   = ['date','region','mean','lower_80','upper_80',
                      'infections','test_adjusted_positive']

    # Import and save as csv
    res = requests.get("https://d14wlfuexuxgcm.cloudfront.net/covid/rt.csv")
    df = pd.read_csv(StringIO(res.text))
    df.to_csv(path/"rtlive_new_estimates.csv", index=False)
    
    # Filter to just necessary features
    df = df[kept_columns]
    
    # Format date properly and rename columns
    df.loc[:,'date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.rename(columns={'region':'state','mean':'RR_pred_rtlivenew',
                        'lower_80':'RR_lower_rtlivenew', 'upper_80':'RR_upper_rtlivenew',
                        'test_adjusted_positive':'adj_positive_rtlivenew',
                        'infections':'infections_rtlivenew'}, inplace=True)
    return df


def get_old_rt_live_estimates(path: Path) -> pd.DataFrame:
    
    # Parameters for filtering raw df
    kept_columns   = ['date','state','mean','lower_95','upper_95']

    # Import and save as csv
    df = pd.read_csv(path/"rtlive_old_estimates.csv")
    
    # Filter to just necessary features
    df = df[kept_columns]
    
    # Format date properly and rename columns
    df.loc[:,'date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.rename(columns={'region':'state','mean':'RR_pred_rtliveold',
                       'lower_95':'RR_lower_rtliveold', 
                       'upper_95':'RR_upper_rtliveold'}, inplace=True)
    return df


def get_cori_estimates(path: Path) -> pd.DataFrame:
    
    # Import and save as csv
    df = pd.read_csv(path/"cori_estimates.csv")
        
    # Format date properly and rename columns
    df.loc[:,'date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    return df

def get_luis_estimates(path: Path) -> pd.DataFrame:
    
    # Import and save as csv
    df = pd.read_csv(path/"luis_code_estimates.csv")
        
    # Format date properly and rename columns
    df.loc[:,'date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    return df
