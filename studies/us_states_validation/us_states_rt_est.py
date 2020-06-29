from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Callable
from tqdm import tqdm
from io import StringIO

from adaptive.utils import cwd
from adaptive.estimators import box_filter, gamma_prior
from etl import import_and_clean_cases, get_rt_live_data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


# PARAMETERS
CI        = 0.95
smoothing = lambda ts: box_filter(ts, 15, None)


def estimate_state_rts(df:pd.DataFrame, CI:float = 0.95,
                       smoothing:Callable = lambda ts: box_filter(ts, 15, None)) -> pd.DataFrame:
    '''
    Gets our estimate of Rt and smoothed case counts based on what is currently in the 
    gamma_prior module. Takes in dataframe of cases and returns dataframe of results.
    '''
    # Initialize results df
    res_full = pd.DataFrame()
    
    # Loop through each state
    print("Estimating state Rt values...")
    for state in tqdm(df['state'].unique()):
        
        # Calculate Rt for that state
        state_df = df[df['state'] == state].set_index('date')
        (
        dates, RR_pred, RR_CI_upper, RR_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        _, anomaly_dates
        ) = gamma_prior(state_df[state_df['positive'] > 0]['positive'], 
                        CI=CI, smoothing=smoothing)
        assert(len(dates) == len(RR_pred))
        
        # Save results
        res = pd.DataFrame({'state':state,
                            'date':dates,
                            'RR_pred':RR_pred,
                            'RR_CI_upper':RR_CI_upper,
                            'RR_CI_lower':RR_CI_lower,
                            'T_pred':T_pred,
                            'T_CI_upper':T_CI_upper,
                            'T_CI_lower':T_CI_lower,
                            'new_cases_ts':new_cases_ts,
                            'total_cases':total_cases[2:],
                            'anamoly':dates.isin(set(anomaly_dates))})
        res_full = pd.concat([res_full,res], axis=0)
    
    # Merge results back onto input df and return
    return df.merge(res_full, how='outer', on=['state','date'])    


def make_state_plots(df: pd.DataFrame, plotspath: Path) -> None:
    '''
    Saves comparison plots of our Rt estimates vs. Rt.live estimates
    into plotspath folder.
    '''
    print("Plotting results...")
    for state in tqdm(df['state'].unique()):
        
        # Get state data
        state_res = merged_df[merged_df['state']==state].sort_values('date')
        daterange = np.arange(np.datetime64(min(state_res['date'])), 
                              np.datetime64(max(state_res['date'])+np.timedelta64(2,'D')), 
                              np.timedelta64(4,'D')) 
        
        # Set up plot
        fig,ax = plt.subplots(2, 1, figsize=(15,15))

        # Top plot
        ax[0].plot(state_res['date'], state_res['RR_pred'], linewidth=2.5)
        ax[0].plot(state_res['date'], state_res['RR_pred_rtlive'], linewidth=2.5)
        ax[0].set_title(f"{state} - Comparing Rt Estimates", fontsize=22)
        ax[0].set_ylabel("Rt Estimate", fontsize=15)
        ax[0].set_xticks(daterange)
        ax[0].set_xticklabels(pd.to_datetime(daterange).strftime("%b %d"), rotation=70)
        ax[0].legend(['Our Rt Estimate', 'rt.live Rt Estimate'],
                     fontsize=15)

        # Bottom plot
        ax[1].plot(state_res['date'], state_res['new_cases_ts'], linewidth=2.5)
        ax[1].plot(state_res['date'], state_res['adj_positive_rtlive'], linewidth=2.5)
        ax[1].plot(state_res['date'], state_res['infections_rtlive'], linewidth=2.5)

        ax[1].set_title(f"{state} - Comparing Estimated Daily New Case Count", fontsize=22)
        ax[1].set_ylabel("Estimated Daily New Case Count", fontsize=15)
        ax[1].set_xticks(daterange)
        ax[1].set_xticklabels(pd.to_datetime(daterange).strftime("%b %d"), rotation=70)
        ax[1].legend(['Our Smoothed Case Count', 
                      'rt.live Test-Adjusted Case Estimate', 
                      'rt.live Infections Estimate'],
                     fontsize=15)

        plt.savefig(plots/f"{state} - Rt and Case Count Comparison")
        plt.close()


if __name__ == "__main__":

    # Folder structures and file names
    root    = cwd()
    data     = root/"data"
    plots    = root/"plots"
    if not data.exists():
        data.mkdir()
    if not plots.exists():
        plots.mkdir()

    # Get data and Rt estimates for us and rt.live
    df         = import_and_clean_cases(data)
    our_rt_df  = estimate_state_rts(df, CI=CI, smoothing=smoothing)
    rt_live_df = get_rt_live_data(data)
    merged_df  = our_rt_df.merge(rt_live_df, how='outer', on=['state','date'])
    ### Note - ABOVE IS NOT A PERFECT MERGE
    ### 5777 full match on state-date
    ###  346 only in rt.live data (mostly early dates, < March 5th)
    ###    2 only in our data (West Virginia, 0 observed cases, doesn't matter)

    # Save CSV and plots
    merged_df.to_csv(data/f"+rt_estimates_comparison.csv")
    make_state_plots(merged_df, plots)