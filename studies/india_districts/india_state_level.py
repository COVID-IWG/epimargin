from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from adaptive.plots import plot_RR_est
from adaptive.estimators import gamma_prior
from adaptive.smoothing import convolution
from adaptive.utils import cwd, days
from adaptive.etl.covid19india import download_data, load_statewise_data, get_time_series
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def project(dates, R_values, smoothing, period = 7*days):
    julian_dates = [_.to_julian_date() for _ in dates[-smoothing//2:None]]
    return OLS(
        RR_pred[-smoothing//2:None], 
        add_constant(julian_dates)
    )\
    .fit()\
    .predict([1, julian_dates[-1] + period])[0]

if __name__ == "__main__":
    root = cwd()
    data = root/"data"

    # model details 
    gamma      = 0.2
    smoothing = 15
    CI        = 0.95

    download_data(data, 'state_wise_daily.csv')

    state_df = load_statewise_data(data/"state_wise_daily.csv", data/"india_state_code_lookup.csv")
    country_time_series = get_time_series(state_df)

    # country level
    (
        dates,
        RR_pred, RR_CI_upper, RR_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = gamma_prior(country_time_series["Hospitalized"].iloc[:-1], CI = CI, smoothing = convolution(window = smoothing)) 

    plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0, ymax=4)\
        .title("India: Reproductive Number Estimate")\
        .xlabel("Date")\
        .ylabel("Rt", rotation=0, labelpad=20)
    plt.ylim(0, 4)
    plt.show()

    # state level rt estimates
    state_time_series = get_time_series(state_df, 'state')
    state_names = list(state_time_series.index.get_level_values(level=0).unique())
    smooth = 10
    estimates = []
    max_len = 1 + max(map(len, state_names))
    with tqdm(state_names) as states:
    	for state in states:
    		states.set_description(f"{state :<{max_len}}")
    		try: 
    			(dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates) = gamma_prior(state_time_series.loc[state]['Hospitalized'], CI = CI, smoothing = convolution(window = smooth))
    			estimates.append((state, RR_pred[-1], RR_CI_lower[-1], RR_CI_upper[-1], project(dates, RR_pred, smooth))) 
    		except (IndexError, ValueError): 
    			estimates.append((state, np.nan, np.nan, np.nan, np.nan))
    estimates = pd.DataFrame(estimates)
    estimates.columns = ["state", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
    estimates.set_index("state", inplace=True)
    estimates.to_csv(data/"Rt_estimates.csv")
    print(estimates)

