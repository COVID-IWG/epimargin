from pathlib import Path
from io import StringIO
from typing import Callable
import numpy as np
import pandas as pd
import requests


def import_clean_smooth_cases(save_path: Path, smoothing: Callable) -> pd.DataFrame:
    '''
    Imports and cleans and smooths case data from covidtracking.com. 
    '''
    # Parameters for filtering raw df
    kept_columns   = ['date','state','positive','negative','death','hospitalizedCumulative']
    excluded_areas = set(['PR','MP','AS','GU','VI'])

    # Import and save raw result
    res = requests.get("https://covidtracking.com/api/v1/states/daily.json")
    df  = pd.read_json(res.text)
    df.to_csv(save_path/"covidtracking_cases_raw.csv", index=False)

    # Exclude specific territories and features
    df = df[~df['state'].isin(excluded_areas)][kept_columns]

    # Format date properly and rename hospitalized variable
    df.loc[:,'date'] = pd.to_datetime(df.loc[:,'date'], format='%Y%m%d')
    df.rename(columns={'hospitalizedCumulative':'hospitalized'}, inplace=True)

    # Calculate state daily changes in each variable
    df = df.sort_values(['state','date'])
    for var in ['positive','negative','death','hospitalized']:
        
        # Reorder columns and make new variables
        df = df[[col for col in list(df.columns) if col != var]+[var]]
        df[f'{var}_orig']       = df[var]
        df[f'{var}_diff']       = df.groupby(['state']).diff()[var]
        df[f'{var}_diff_orig']  = df[f'{var}_diff']
        
        # Keep looping while there are negative values for delta var
        while any(df[f'{var}_diff'] < 0.0):
            
            # Reset previous day to be equal to next day, to force delta variable to zero
            df[f'{var}_shifted']       = df[var].shift(-1)
            df[f'{var}_shifted_diff']  = df[f'{var}_diff'].shift(-1)
            df.loc[df[f'{var}_shifted_diff'] < 0.0, var] = df.loc[df[f'{var}_shifted_diff'] < 0.0, f'{var}_shifted']
            df.drop(columns=[f'{var}_shifted',f'{var}_shifted_diff'], inplace=True)        
            
            # Re-generate delta variable
            df.loc[:,f'{var}_diff'] = df.groupby(['state']).diff()[var]

    # Calculate total test data as sum of positives and negatives
    df['tests']            = df['positive']+df['negative']
    df['tests_orig']       = df['positive_orig']+df['negative_orig']
    df['tests_diff']       = df['positive_diff']+df['negative_diff']
    df['tests_diff_orig']  = df['positive_diff_orig']+df['negative_diff_orig']

    # Do smoothing for each variable in each state
    fulldf = pd.DataFrame()
    for state in df['state'].unique():
        
        # Subset to just state
        smoothingdf = df[df.state == state]
        for var in ['positive','negative','death','hospitalized']:
                
            # Get just smoothable values
            statedf = smoothingdf[smoothingdf[f'{var}_diff'].notnull()] 
            statedf = statedf.sort_values('date')
            
            # Stop if we have too many null values
            if statedf.shape[0] > 15:
                
                # Smooth over first difference
                statedf[f'{var}_diff_smooth'] = smoothing(statedf[f'{var}_diff'])
                statedf[f'{var}_smooth']      = statedf[f'{var}_diff_smooth'].cumsum()

                # Merge results onto smoothing df
                smoothingdf = smoothingdf.merge(statedf[['state','date',f'{var}_diff_smooth',f'{var}_smooth']], on=['date','state'], how='outer')

        # Smooth results for tests values
        smoothingdf['tests_diff_smooth']    = smoothingdf['positive_diff_smooth']+smoothingdf['negative_diff_smooth']
        smoothingdf['tests_smooth']         = smoothingdf['tests_diff_smooth'].cumsum()
        
        # Append to full data frame
        fulldf = pd.concat((fulldf, smoothingdf), axis=0)
    df = fulldf.reset_index(drop=True)

    # Save out clean data frame
    df.to_csv(save_path/"covidtracking_cases_clean.csv", index=False)

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
    df.rename(columns={'region':'state','mean':'RR_pred_rtlivenew',
                        'lower_80':'RR_lower_rtlivenew', 'upper_80':'RR_upper_rtlivenew',
                        'test_adjusted_positive':'adj_positive_rtlivenew',
                        'infections':'infections_rtlivenew'}, inplace=True)
    return df