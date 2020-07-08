from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import etl
from adaptive.estimators import gamma_prior
from adaptive.model import Model, ModelUnit
from adaptive.plots import PlotDevice, plot_RR_est, plot_T_anomalies
from adaptive.smoothing import convolution
from adaptive.utils import cwd

if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"
    
    gamma  = 0.2
    CI     = 0.95
    smoothing = 15

    state_cases    = etl.load_cases(data/"Bihar_Case_data_Jul320.csv")
    # deal with malformed dates
    state_cases["DATE OF DISCHARGE"] = state_cases["DATE OF DISCHARGE"].mask((state_cases["DATE OF DISCHARGE"].isna()) | (state_cases["DATE OF DISCHARGE"].isna() == "00/01/00"), state_cases["DATE OF POSITIVE TEST CONFIRMATION"])
    state_cases = state_cases.drop(state_cases.index[state_cases["DISTRICT"] == "PURNEA"])

    (
        dates,
        RR_pred, RR_CI_upper, RR_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = gamma_prior(etl.get_time_series(state_cases)["Hospitalized"].iloc[:-1], CI = CI, smoothing = convolution(window = smoothing)) 

    plot_RR_est(dates[dates > "25-04-2020"], RR_pred[-67:None], RR_CI_upper[-67:None], RR_CI_lower[-67:None], CI, ymin=0, ymax=4)\
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
    print(Bihar[0].delta_T)
    print(Bihar[0].lower_CI)
    print(Bihar[0].upper_CI)
    plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
    plt.scatter(t_pred, Bihar[0].delta_T, color = "tomato", s = 4, label = "Predicted Net Cases")
    plt.fill_between(t_pred, Bihar[0].lower_CI, Bihar[0].upper_CI, color = "tomato", alpha = 0.3, label="99% CI (forecast)")
    plt.legend()
    PlotDevice().title("Bihar: Net Daily Cases").xlabel("Date").ylabel("Cases")
    plt.show()
