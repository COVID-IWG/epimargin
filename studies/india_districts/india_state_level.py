import sys
from pathlib import Path
from typing import Dict, Optional, Sequence
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

from adaptive.estimators import analytical_MPVS
from adaptive.etl.covid19india import (download_data, get_time_series,
                                       load_statewise_data, state_name_lookup)
from adaptive.plots import plot_RR_est
from adaptive.smoothing import convolution
from adaptive.utils import cwd, days

simplefilter("ignore")

def project(dates, R_values, smoothing, period = 7*days):
    julian_dates = [_.to_julian_date() for _ in dates[-smoothing//2:None]]
    return OLS(
        RR_pred[-smoothing//2:None], 
        add_constant(julian_dates)
    )\
    .fit()\
    .predict([1, julian_dates[-1] + period])[0]

# set to cloud temp directory if not explicitly told to run locally 
root = cwd() if len(sys.argv) > 1 and sys.argv[1] == "--local" else Path("/tmp")
data = root/"data"

# model details 
gamma     = 0.2
smoothing = 21
CI        = 0.95

download_data(data, 'state_wise_daily.csv')

state_df = load_statewise_data(data/"state_wise_daily.csv")
country_time_series = get_time_series(state_df)

estimates  = []
timeseries = []

# country level
(
    dates,
    RR_pred, RR_CI_upper, RR_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = analytical_MPVS(country_time_series["Hospitalized"].iloc[:-1], CI = CI, smoothing = convolution(window = smoothing)) 

country_code = state_name_lookup["India"]
for row in zip(dates, RR_pred, RR_CI_upper, RR_CI_lower):
    timeseries.append((country_code, *row))

# state level rt estimates
state_time_series = get_time_series(state_df, 'state')
state_names = list(state_time_series.index.get_level_values(level=0).unique())
max_len = 1 + max(map(len, state_names))
with tqdm(state_names) as states:
    for state in states:
        state_code = state_name_lookup[state]
        states.set_description(f"{state :<{max_len}}")
        try: 
            (dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates) = analytical_MPVS(state_time_series.loc[state]['Hospitalized'], CI = CI, smoothing = convolution(window = smoothing))
            for row in zip(dates, RR_pred, RR_CI_upper, RR_CI_lower):
                timeseries.append((state_code, *row))
            estimates.append((state_code, RR_pred[-1], RR_CI_lower[-1], RR_CI_upper[-1], project(dates, RR_pred, smoothing)))
        except (IndexError, ValueError): 
            estimates.append((state, np.nan, np.nan, np.nan, np.nan))

estimates = pd.DataFrame(estimates)
estimates.columns = ["state", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
estimates.set_index("state", inplace=True)
estimates.to_csv(data/"Rt_estimates.csv")

timeseries = pd.DataFrame(timeseries)
timeseries.columns = ["state", "date", "Rt", "Rt_upper", "Rt_lower"]
timeseries.set_index("state", inplace=True)
timeseries.to_csv(data/"Rt_timeseries_india.csv")
