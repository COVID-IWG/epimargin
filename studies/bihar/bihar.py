from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.regression.linear_model import OLS 
from statsmodels.tools import add_constant

import etl
from adaptive.estimators import gamma_prior
from adaptive.model import Model, ModelUnit
from adaptive.plots import PlotDevice, plot_RR_est, plot_T_anomalies
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

root = cwd()
data = root/"data"
figs = root/"figs"

gamma     = 0.2
smoothing = 12
CI        = 0.95

state_cases = pd.read_csv(data/"Bihar_cases_data_Jul23.csv", parse_dates=["date_reported"], dayfirst=True)
state_ts = state_cases["date_reported"].value_counts().sort_index()
district_names, population_counts, _ = etl.district_migration_matrix(data/"Migration Matrix - District.csv")
populations = dict(zip(district_names, population_counts))

# first, look at state level predictions
(
    dates,
    RR_pred, RR_CI_upper, RR_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = gamma_prior(state_ts, CI = CI, smoothing = convolution(window = smoothing)) 

plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0, ymax=4)\
    .title("Bihar: Reproductive Number Estimate")\
    .xlabel("Date")\
    .ylabel("Rt", rotation=0, labelpad=20)
plt.ylim(0, 4)
plt.show()

np.random.seed(33)
Bihar = Model([ModelUnit("Bihar", 99_000_000, I0 = T_pred[-1], RR0 = RR_pred[-1], mobility = 0)])
Bihar.run(14, np.zeros((1,1)))

t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(len(Bihar[0].delta_T))]

Bihar[0].lower_CI[0] = T_CI_lower[-1]
Bihar[0].upper_CI[0] = T_CI_upper[-1]
plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
plt.scatter(t_pred, Bihar[0].delta_T, color = "tomato", s = 4, label = "Predicted Net Cases")
plt.fill_between(t_pred, Bihar[0].lower_CI, Bihar[0].upper_CI, color = "tomato", alpha = 0.3, label="99% CI (forecast)")
plt.legend()
PlotDevice().title("Bihar: Net Daily Cases").xlabel("Date").ylabel("Cases")
# plt.semilogy()
plt.ylim(0, 1600)
plt.show()

# now, do district-level estimation 
smoothing = 10
district_time_series = state_cases.groupby(["geo_reported", "date_reported"])["date_reported"].count().sort_index()
migration = np.zeros((len(district_names), len(district_names)))
estimates = []
max_len = 1 + max(map(len, district_names))
with tqdm([etl.replacements.get(dn, dn) for dn in district_names]) as districts:
    for district in districts:
        districts.set_description(f"{district :<{max_len}}")
        try: 
            (dates, RR_pred, RR_CI_upper, RR_CI_lower, *_) = gamma_prior(district_time_series.loc[district], CI = CI, smoothing = convolution(window = smoothing))
            estimates.append((district, RR_pred[-1], RR_CI_lower[-1], RR_CI_upper[-1], project(dates, RR_pred, smoothing))) 
        except (IndexError, ValueError): 
            estimates.append((district, np.nan, np.nan, np.nan, np.nan))
estimates = pd.DataFrame(estimates)
estimates.columns = ["district", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
estimates.set_index("district", inplace=True)
estimates.to_csv(data/"Rt_estimates.csv")
print(estimates)
