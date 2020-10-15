from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from warnings import simplefilter

import adaptive.plots as plt
import numpy as np
import pandas as pd
from adaptive.estimators import analytical_MPVS, linear_projection
from adaptive.models import SIR, NetworkedSIR
from adaptive.smoothing import convolution, notched_smoothing
from adaptive.utils import cwd, days, setup
from matplotlib import rcParams
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

import etl

simplefilter("ignore")

(data, figs) = setup()

gamma     = 0.2
smoothing = 10
CI        = 0.95

state_cases = pd.read_csv(data/"Bihar_cases_data_Oct03.csv", parse_dates=["date_reported"], dayfirst=True)
state_ts = state_cases["date_reported"].value_counts().sort_index()
district_names, population_counts, _ = etl.district_migration_matrix(data/"Migration Matrix - District.csv")
populations = dict(zip(district_names, population_counts))

# first, look at state level predictions
(
    dates,
    Rt_pred, Rt_CI_upper, Rt_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = analytical_MPVS(state_ts, CI = CI, smoothing = notched_smoothing(window = smoothing), totals=False) 

plt.Rt(dates, Rt_pred[1:], Rt_CI_upper[1:], Rt_CI_lower[1:], CI, ymin=0, ymax=4)\
    .title("\nBihar: Reproductive Number Estimate")\
    .annotate(f"data from {str(dates[0]).split()[0]} to {str(dates[-1]).split()[0]}")\
    .xlabel("date")\
    .ylabel("$R_t$", rotation=0, labelpad=20)\
    .show()

np.random.seed(33)
Bihar = SIR("Bihar", 99_000_000, dT0 = T_pred[-1], Rt0 = Rt_pred[-1], lower_CI =T_CI_lower[-1], upper_CI =  T_CI_upper[-1], mobility = 0)
Bihar.run(14)

t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(len(Bihar.dT))]

plt.daily_cases(dates, T_pred[1:], T_CI_upper[1:], T_CI_lower[1:], new_cases_ts[1:], anomaly_dates, anomalies, CI,  
    prediction_ts = [
        (Bihar.dT, Bihar.lower_CI, Bihar.upper_CI, None, "predicted cases")
    ])\
    .title("\nBihar: New Daily Cases")\
    .annotate(f"data from {str(dates[0]).split()[0]} to {str(dates[-1]).split()[0]}; predictions until {str(t_pred[-1]).split()[0]}")\
    .xlabel("date").ylabel("cases")\
    .show()

# now, do district-level estimation 
smoothing = 10
state_cases["geo_reported"] = state_cases.geo_reported.str.strip()
district_time_series = state_cases.groupby(["geo_reported", "date_reported"])["date_reported"].count().sort_index()
migration = np.zeros((len(district_names), len(district_names)))
estimates = []
max_len = 1 + max(map(len, district_names))
with tqdm(etl.normalize(district_names)) as districts:
    for district in districts:
        districts.set_description(f"{district :<{max_len}}")
        try: 
            (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, *_) = analytical_MPVS(district_time_series.loc[district], CI = CI, smoothing = convolution(window = smoothing), totals=False)
            estimates.append((district, Rt_pred[-1], Rt_CI_lower[-1], Rt_CI_upper[-1], linear_projection(dates, Rt_pred, smoothing))) 
        except (IndexError, ValueError): 
            estimates.append((district, np.nan, np.nan, np.nan, np.nan))
estimates = pd.DataFrame(estimates)
estimates.columns = ["district", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
estimates.set_index("district", inplace=True)
estimates.to_csv(data/"Rt_estimates_private.csv")
print(estimates)
