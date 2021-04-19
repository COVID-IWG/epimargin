from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

import etl
from epimargin.estimators import analytical_MPVS
from epimargin.etl.commons import download_data
from epimargin.etl.covid19india import data_path, get_time_series, load_all_data
from epimargin.model import Model, ModelUnit
from epimargin.plots import PlotDevice, plot_RR_est, plot_T_anomalies
from epimargin.smoothing import convolution
from epimargin.utils import cwd, days

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

# private data
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
) = analytical_MPVS(state_ts, CI = CI, smoothing = convolution(window = smoothing)) 

plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0, ymax=4)\
    .title("Bihar: Reproductive Number Estimate Comparisons")\
    .xlabel("Date")\
    .ylabel("Rt", rotation=0, labelpad=20)
plt.ylim(0, 4)

# public data 
paths = { 
    "v3": [data_path(_) for _ in (1, 2)],
    "v4": [data_path(_) for _ in range(3, 13)]
}

for target in paths['v3'] + paths['v4']:
    download_data(data, target)

dfn = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)
 
state_ts = get_time_series(dfn, "detected_state").loc["Bihar"]
district_names, population_counts, _ = etl.district_migration_matrix(data/"Migration Matrix - District.csv")
populations = dict(zip(district_names, population_counts))

# first, look at state level predictions
(dates_public, RR_pred_public, RR_CI_upper_public, RR_CI_lower_public, T_pred_public, T_CI_upper_public, T_CI_lower_public, total_cases_public, new_cases_ts_public, anomalies_public, anomaly_dates_public) = analytical_MPVS(state_ts.Hospitalized, CI = CI, smoothing = convolution(window = smoothing)) 
plt.plot(dates_public, RR_pred_public, label = "Estimated $R_t$", color = "midnightblue")
plt.fill_between(dates_public, RR_CI_lower_public, RR_CI_upper_public, label = f"{100*CI}% CI", color = "midnightblue", alpha = 0.3)
plt.legend(["private data estimate", "public data estimate"])
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
PlotDevice().title("Bihar Net Daily Cases: Private Data Projection vs. Public Reported Data").xlabel("Date").ylabel("Cases")
plt.plot(dates_public, new_cases_ts_public, "k-", alpha = 0.6, label="Empirical Public Data")
plt.legend()
plt.semilogy()
plt.ylim(0, 2000)
plt.show()

# # now, do district-level estimation 
# smoothing = 10
# district_time_series = state_cases.groupby(["geo_reported", "date_reported"])["date_reported"].count().sort_index()
# migration = np.zeros((len(district_names), len(district_names)))
# estimates = []
# max_len = 1 + max(map(len, district_names))
# with tqdm([etl.replacements.get(dn, dn) for dn in district_names]) as districts:
#     for district in districts:
#         districts.set_description(f"{district :<{max_len}}")
#         try: 
#             (dates, RR_pred, RR_CI_upper, RR_CI_lower, *_) = analytical_MPVS(district_time_series.loc[district], CI = CI, smoothing = convolution(window = smoothing))
#             estimates.append((district, RR_pred[-1], RR_CI_lower[-1], RR_CI_upper[-1], project(dates, RR_pred, smoothing))) 
#         except (IndexError, ValueError): 
#             estimates.append((district, np.nan, np.nan, np.nan, np.nan))
# estimates = pd.DataFrame(estimates)
# estimates.columns = ["district", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
# estimates.set_index("district", inplace=True)
# estimates.to_csv(data/"Rt_estimates.csv")
# print(estimates)
