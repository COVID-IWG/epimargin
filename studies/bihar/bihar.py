from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from warnings import simplefilter

import adaptive.plots as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adaptive.estimators import analytical_MPVS
from adaptive.model import Model, ModelUnit
from adaptive.smoothing import convolution, notched_smoothing
from adaptive.utils import cwd, days
from matplotlib import rcParams
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

import etl

rcParams["savefig.dpi"] = 300

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
smoothing = 10
CI        = 0.95

state_cases = pd.read_csv(data/"Bihar_cases_data_Oct03.csv", parse_dates=["date_reported"], dayfirst=True)
state_ts = state_cases["date_reported"].value_counts().sort_index()
# state_ts = state_ts[state_ts.index <= "2020-09-01"]
district_names, population_counts, _ = etl.district_migration_matrix(data/"Migration Matrix - District.csv")
populations = dict(zip(district_names, population_counts))

# first, look at state level predictions
(
    dates,
    RR_pred, RR_CI_upper, RR_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = analytical_MPVS(state_ts, CI = CI, smoothing = notched_smoothing(window = smoothing), totals=False) 

plt.Rt(dates, RR_pred[1:], RR_CI_upper[1:], RR_CI_lower[1:], CI, ymin=0, ymax=4)\
    .title("\nBihar: Reproductive Number Estimate")\
    .annotate(f"data from {str(dates[0]).split()[0]} to {str(dates[-1]).split()[0]}")\
    .xlabel("date")\
    .ylabel("$R_t$", rotation=0, labelpad=20)
plt.xlim(left = dates[0], right = dates[-1])
plt.ylim(0, 4)
plt.hlines(1, xmin=dates[0], xmax=dates[-1], color="dimgray")
plt.show()

np.random.seed(33)
Bihar = Model([ModelUnit("Bihar", 99_000_000, I0 = T_pred[-1], RR0 = RR_pred[-1], mobility = 0)])
Bihar.run(14, np.zeros((1,1)))

t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(len(Bihar[0].delta_T))]

Bihar[0].lower_CI[0] = T_CI_lower[-1]
Bihar[0].upper_CI[0] = T_CI_upper[-1]
plt.daily_cases(dates, T_pred[1:], T_CI_upper[1:], T_CI_lower[1:], new_cases_ts[1:], anomaly_dates, anomalies, CI, 
    prediction_ts = [
        
    ]
)
PlotDevice().title("\nBihar: New Daily Cases")\
    .annotate(f"data from {str(dates[0]).split()[0]} to {str(dates[-1]).split()[0]}; predictions until {str(t_pred[-1]).split()[0]}")\
    .xlabel("date").ylabel("cases")\
    .show()

# now, do district-level estimation 
smoothing = 10
state_cases["geo_reported"] = state_cases.geo_reported.str.strip()
district_time_series = state_cases.groupby(["geo_reported", "date_reported"])["date_reported"].count().sort_index()
district_names = sorted([_ if _ != "MUZZAFARPUR" else "MUZAFFARPUR" for _ in district_names + ["ARWAL"]])
migration = np.zeros((len(district_names), len(district_names)))
estimates = []
max_len = 1 + max(map(len, district_names))
with tqdm([etl.replacements.get(dn, dn) for dn in district_names]) as districts:
    for district in districts:
        districts.set_description(f"{district :<{max_len}}")
        try: 
            (dates, RR_pred, RR_CI_upper, RR_CI_lower, *_) = analytical_MPVS(district_time_series.loc[district], CI = CI, smoothing = convolution(window = smoothing), totals=False)
            estimates.append((district, RR_pred[-1], RR_CI_lower[-1], RR_CI_upper[-1], project(dates, RR_pred, smoothing))) 
        except (IndexError, ValueError): 
            estimates.append((district, np.nan, np.nan, np.nan, np.nan))
estimates = pd.DataFrame(estimates)
estimates.columns = ["district", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
estimates.set_index("district", inplace=True)
estimates.to_csv(data/"Rt_estimates_private_data.csv")
print(estimates)
