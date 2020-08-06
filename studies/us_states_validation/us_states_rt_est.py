from pathlib import Path
from typing  import Dict, Optional, Sequence, Tuple, Callable
from tqdm    import tqdm
from io      import StringIO

from adaptive.utils      import cwd
from adaptive.estimators import gamma_prior
from adaptive.smoothing  import notched_smoothing

from etl import import_clean_smooth_cases
from etl import get_adaptive_estimates, get_new_rt_live_estimates, get_old_rt_live_estimates, get_cori_estimates, get_luis_estimates
from rtlive_old_model import run_rtlive_old_model
from luis_model import run_luis_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import subprocess


# PARAMETERS
CI               = 0.95
smoothing_window = 7
rexepath         = 'C:\\Program Files\\R\\R-3.6.1\\bin\\'


def run_adaptive_model(df:pd.DataFrame, locationvar:str, CI:float, filepath:Path) -> None:
    '''
    Runs adaptive control model of Rt and smoothed case counts based on what is currently in the 
    gamma_prior module. Takes in dataframe of cases and saves to csv a dataframe of results.
    '''
    # Initialize results df
    res_full = pd.DataFrame()

    # Null smoother to pass to gamma_prior (since smoothing was already done)
    def null_smoother(data: Sequence[float]):
        return data

    # Loop through each location
    print(f"Estimating Adaptive Rt values for each {locationvar}...")
    for loc in tqdm(df[locationvar].unique()):
                
        # Calculate Rt for that location
        loc_df = df[df[locationvar] == loc].set_index('date')
        (
        dates, RR_pred, RR_CI_upper, RR_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        _, anomaly_dates
        ) = gamma_prior(loc_df[loc_df['positive_smooth'] > 0]['positive_smooth'], 
                        CI=CI, smoothing=null_smoother)
        assert(len(dates) == len(RR_pred))
        
        # Save results
        res = pd.DataFrame({locationvar:loc,
                            'date':dates,
                            'RR_pred':RR_pred,
                            'RR_CI_upper':RR_CI_upper,
                            'RR_CI_lower':RR_CI_lower,
                            'T_pred':T_pred,
                            'T_CI_upper':T_CI_upper,
                            'T_CI_lower':T_CI_lower,
                            'new_cases_ts':new_cases_ts,
                            'total_cases':total_cases[2:],
                            'anomaly':dates.isin(set(anomaly_dates))})
        res_full = pd.concat([res_full,res], axis=0)
    
    # Merge results back onto input df and return
    merged_df = df.merge(res_full, how='outer', on=[locationvar,'date'])

    # Parameters for filtering raw df
    kept_columns   = ['date',locationvar,'RR_pred','RR_CI_lower','RR_CI_upper','T_pred',
                      'T_CI_lower','T_CI_upper','new_cases_ts','anomaly']
    merged_df      = merged_df[kept_columns]
    
    # Format date properly and return
    merged_df.loc[:,'date'] = pd.to_datetime(merged_df['date'], format='%Y-%m-%d')

    # Save out result
    merged_df.to_csv(filepath/"adaptive_estimates.csv")


def run_cori_model(filepath:Path, rexepath:Path) -> None:
    '''
    Runs R script that runs Cori model estimates. Saves results in
    a CSV file.
    '''
    subprocess.call([rexepath/"Rscript.exe", filepath/"cori_model.R"], shell=True)


def make_state_plots(df:pd.DataFrame, plotspath:Path) -> None:
    '''
    Saves comparison plots of our Rt estimates vs. Rt.live estimates
    into plotspath folder.
    '''
    print("Plotting results...")
    for state in tqdm(df['state'].unique()):
                
        # Get state data
        state_res = df[df['state']==state].sort_values('date')

        # Filter to after 3/15/2020 (earlier estimates vary wildly)
        state_res = state_res[state_res['date'] >= '2020-04-01'] 
        daterange = np.arange(np.datetime64(min(state_res['date'])), 
                              np.datetime64(max(state_res['date'])+np.timedelta64(2,'D')), 
                              np.timedelta64(4,'D')) 
        
        # Set up plot
        fig,ax = plt.subplots(2, 1, figsize=(15,15))

        # Top plot
        ax[0].plot(state_res['date'], state_res['RR_pred'], linewidth=2.5)
        ax[0].plot(state_res['date'], state_res['RR_pred_rtlivenew'], linewidth=2.5)
        ax[0].plot(state_res['date'], state_res['RR_pred_rtliveold'], linewidth=2.5)
        ax[0].plot(state_res['date'], state_res['RR_pred_cori'], linewidth=2.5)
        ax[0].plot(state_res['date'], state_res['RR_pred_luis'], linewidth=2.5)
        ax[0].set_title(f"{state} - Comparing Rt Estimates", fontsize=22)
        ax[0].set_ylabel("Rt Estimate", fontsize=15)
        ax[0].set_xticks(daterange)
        ax[0].set_xticklabels(pd.to_datetime(daterange).strftime("%b %d"), rotation=70)
        ax[0].legend(['Adaptive Control Estimate', 'New rt.live Estimate', 'Old rt.live Estimate', 'Cori Method Estimate', 'Luis Code Estimate'],
                     fontsize=15)

        # Bottom plot
        ax[1].plot(state_res['date'], state_res['new_cases_ts'], linewidth=2.5)
        ax[1].plot(state_res['date'], state_res['adj_positive_rtlivenew'], linewidth=2.5)
        ax[1].plot(state_res['date'], state_res['infections_rtlivenew'], linewidth=2.5)

        ax[1].set_title(f"{state} - Comparing Estimated Daily New Case Count", fontsize=22)
        ax[1].set_ylabel("Estimated Daily New Case Count", fontsize=15)
        ax[1].set_xticks(daterange)
        ax[1].set_xticklabels(pd.to_datetime(daterange).strftime("%b %d"), rotation=70)
        ax[1].legend(['Adaptive Control Smoothed Case Count', 
                      'New rt.live Test-Adjusted Case Estimate', 
                      'New rt.live Infections Estimate'],
                     fontsize=15)

        plt.savefig(plotspath/f"{state} - Rt and Case Count Comparison")
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

    # Get data case data
    df = import_clean_smooth_cases(data, notched_smoothing(window=smoothing_window))

    # Run models for adaptive and rt.live old version
    run_adaptive_model(df=df, locationvar='state', CI=CI, filepath=data)
    run_luis_model(df=df, locationvar='state', CI=CI, filepath=data)
    run_rtlive_old_model(df=df, locationvar='state', CI=CI, filepath=data)
    # run_cori_model(filepath=root, rexepath=rexepath) # Have to change R file parameters separately

    # Pull CSVs of results
    adaptive_df    = pd.read_csv(data/"adaptive_estimates.csv")
    rt_live_new_df = get_new_rt_live_estimates(data)
    rt_live_old_df = pd.read_csv(data/"rtlive_old_estimates.csv")
    cori_df        = pd.read_csv(data/"cori_estimates.csv")
    luis_df        = pd.read_csv(data/"luis_code_estimates.csv")

    # Merge all results together
    merged_df      = adaptive_df.merge(rt_live_new_df, how='outer', on=['state','date'])
    merged_df      = merged_df.merge(rt_live_old_df, how='outer', on=['state','date'])
    merged_df      = merged_df.merge(cori_df, how='outer', on=['state','date'])
    merged_df      = merged_df.merge(luis_df, how='outer', on=['state','date'])

    # Fix date formatting   
    merged_df.loc[:,'date'] = pd.to_datetime(merged_df['date'], format='%Y-%m-%d')

    # Save CSV and plots
    merged_df.to_csv(data/"+rt_estimates_comparison.csv")
    make_state_plots(merged_df, plots)