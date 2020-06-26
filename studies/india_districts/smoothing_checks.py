from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt 

import numpy as np
import pandas as pd
from tqdm import tqdm

from adaptive.estimators import box_filter, gamma_prior
from adaptive.utils import cwd, days, weeks
from adaptive.plots import plot_RR_est, plot_T_anomalies
from etl import download_data, get_time_series, load_all_data

if __name__ == "__main__":
    root = cwd()
    data = root/"data"

    # model details 
    gamma      = 0.2
    prevalence = 1

    states = ["Maharashtra", "Andhra Pradesh", "Tamil Nadu", "Madhya Pradesh", "Punjab", "Gujarat", "Kerala", "Bihar"]

    # define data versions for api files
    paths = { "v3": ["raw_data1.csv", "raw_data2.csv"],
              "v4": ["raw_data3.csv", "raw_data4.csv",
                     "raw_data5.csv", "raw_data6.csv"] } 

    # download data from india covid 19 api
    for target in paths['v3'] + paths['v4']:
        download_data(data, target)

    # run rolling regressions on historical national case data 
    dfn = load_all_data(
        v3_paths = [data/filepath for filepath in paths['v3']], 
        v4_paths = [data/filepath for filepath in paths['v4']]
    )

    # first, check reimplementation against known figures 
    CI = 0.99
    for state in states: 
        ts = get_time_series(dfn[
            (dfn.date_announced <= "2020-05-08") & 
            (dfn.detected_state == state)
        ])
        (dates, 
        RR_pred, RR_CI_upper, RR_CI_lower, 
        T_pred, T_CI_upper, T_CI_lower, 
        total_cases, new_cases_ts, 
        anomalies, anomaly_dates) = gamma_prior(ts.Hospitalized, CI = CI)
        plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI)
        plt.title(f"{state}: Estimated $R_t$", loc="left", fontsize=20)
        plt.figure()
        plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
        plt.title(f"{state}: Predicted Cases", loc="left", fontsize=20)

        plt.show()

    # generate smoothing checks for national time series 
    ts = get_time_series(dfn)

    CI = 0.95
    # 1: box filter 
    RRfig, RRax = plt.subplots(3, 1, sharex=True)
    T_fig, T_ax = plt.subplots(3, 1, sharex=True)
    for (i, n) in enumerate((5, 10, 15)): 
        (
            dates, 
            RR_pred, RR_CI_upper, RR_CI_lower, 
            T_pred, T_CI_upper, T_CI_lower, 
            total_cases, new_cases_ts, 
            anomalies, anomaly_dates
        ) = gamma_prior(ts.Hospitalized, CI = CI, smoothing = lambda ts: box_filter(ts, n, None))
        
        plt.sca(RRax[i])
        plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI)
        plt.title(f"smoothing window: {n}, no local smoothing", loc = "left", fontsize = 16)
        
        plt.sca(T_ax[i])
        plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
        plt.title(f"smoothing window: {n}, no local smoothing", loc = "left", fontsize = 16)
    
    plt.figure(RRfig.number)
    RRfig.tight_layout()
    plt.suptitle("Estimated Rt (box filter convolution)", fontsize = 20)
    plt.figure(T_fig.number)
    T_fig.tight_layout()
    plt.suptitle("Estimated Daily Cases (box filter convolution)", fontsize = 20)
    plt.show()


    # 2: box filter, local smooothing 
    RRfig, RRax = plt.subplots(3, 1, sharex=True)
    T_fig, T_ax = plt.subplots(3, 1, sharex=True)
    for (i, n) in enumerate((5, 10, 15)): 
        s = n//2 + 1
        (
            dates, 
            RR_pred, RR_CI_upper, RR_CI_lower, 
            T_pred, T_CI_upper, T_CI_lower, 
            total_cases, new_cases_ts, 
            anomalies, anomaly_dates
        ) = gamma_prior(ts.Hospitalized, CI = CI, smoothing = lambda ts: box_filter(ts, n, s))
        
        plt.sca(RRax[i])
        plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI)
        plt.title(f"smoothing window: {n}, local smoothing: last {s} points", loc = "left", fontsize = 16)
        
        plt.sca(T_ax[i])
        plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
        plt.title(f"smoothing window: {n}, local smoothing: last {s} points", loc = "left", fontsize = 16)
    
    plt.figure(RRfig.number)
    RRfig.tight_layout()
    plt.suptitle("Estimated Rt (box filter convolution, local smoothing)", fontsize = 20)
    plt.subplots_adjust(right = 0.9)

    plt.figure(T_fig.number)
    T_fig.tight_layout()
    plt.suptitle("Estimated Daily Cases (box filter convolution, local smoothing)", fontsize = 20)
    plt.subplots_adjust(right = 0.9)
    
    plt.show()