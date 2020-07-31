from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

from adaptive.estimators import gamma_prior
from adaptive.etl.covid19india import (data_path, download_data,
                                       get_time_series, load_all_data)
from adaptive.etl.devdatalab import district_migration_matrices
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

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

gamma     = 0.2
smoothing = 12
CI        = 0.95

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
 
state_ts = get_time_series(dfn, "detected_state").loc["Gujarat"]
district_names, population_counts, _ = district_migration_matrices(data/"Migration Matrix - 2011 District.csv", ["Gujarat"])["Gujarat"]
populations = dict(zip(district_names, population_counts))

# first, look at state level predictions
(
    dates,
    RR_pred, RR_CI_upper, RR_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = gamma_prior(state_ts[state_ts.date <= "2020-07-17"].Hospitalized, CI = CI, smoothing = convolution(window = smoothing)) 

plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0, ymax=4)\
    .title("Gujarat: Reproductive Number Estimate (Covid19India Data)")\
    .xlabel("Date")\
    .ylabel("Rt", rotation=0, labelpad=20)
plt.ylim(0, 4)
plt.show()

np.random.seed(33)
Gujarat = Model([ModelUnit("Gujarat", 99_000_000, I0 = T_pred[-1], RR0 = RR_pred[-1], mobility = 0)])
Gujarat.run(14, np.zeros((1,1)))

t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(len(Gujarat[0].delta_T))]

Gujarat[0].lower_CI[0] = T_CI_lower[-1]
Gujarat[0].upper_CI[0] = T_CI_upper[-1]
plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
plt.scatter(t_pred, Gujarat[0].delta_T, color = "tomato", s = 4, label = "Predicted Net Cases")
plt.fill_between(t_pred, Gujarat[0].lower_CI, Gujarat[0].upper_CI, color = "tomato", alpha = 0.3, label="99% CI (forecast)")
plt.legend()
PlotDevice().title("Gujarat: Net Daily Cases (Covid19India Data)").xlabel("Date").ylabel("Cases")
plt.plot(state_ts[state_ts.date > "2020-07-17"].index, state_ts[state_ts.date > "2020-07-17"].Hospitalized, "k.", alpha = 0.6, label = "Empirical Observed Cases")
plt.legend()
# plt.semilogy()
plt.show()
