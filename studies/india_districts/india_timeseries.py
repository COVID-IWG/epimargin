import sys
from pathlib import Path
from typing import Dict, Optional, Sequence
from warnings import simplefilter

import flat_table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

from adaptive.estimators import analytical_MPVS
from adaptive.etl.covid19india import download_data, state_name_lookup
from adaptive.plots import plot_RR_est
from adaptive.smoothing import notched_smoothing
from adaptive.utils import cwd, days

simplefilter("ignore")

sns.set(palette="bright", style="darkgrid", font="Libre Franklin")                                                                                                                                                                                                                                                
sns.despine()

save_columns = [
    "state", "date", "Rt", "Rt_upper", "Rt_lower", 
    "cases", "total_cases",
    "recovered", "total_recovered",
    "deceased", "total_deceased",
    "tested", "total_tested",
    "active", "total_active",
    "active_per_mn", "total_active_per_mn",
    "cfr", "total_cfr",
    "infection_rate", "total_infection_rate",
    "recovery_rate", "total_recovery_rate"
]

population = {
    "AN": 0.397, 
    "AP": 52.221, 
    "AR": 1.504, 
    "AS": 34.293, 
    "BR": 119.52, 
    "CH": 1.179, 
    "CT": 28.724,
    "DN": 0.959, 
    "DL": 19.814, 
    "GA": 1.54, 
    "GJ": 67.936, 
    "HR": 28.672, 
    "HP": 7.3, 
    "JK": 13.203, 
    "JH": 37.403, 
    "KA": 65.798,
    "KL": 35.125, 
    "LA": 0.293, 
    "LD": 0.064, 
    "MP": 82.232, 
    "MH": 122.153, 
    "MN": 3.103, 
    "ML": 3.224, 
    "MZ": 1.192, 
    "NL": 2.15,
    "OR": 43.671, 
    "PY": 1.504, 
    "PB": 29.859, 
    "RJ": 77.264, 
    "SK": 0.664, 
    "TN": 75.695, 
    "TG": 37.22, 
    "TR": 3.992, 
    "UP": 224.979,
    "UT": 11.141, 
    "WB": 96.906, 
    "TT": 1332.83
}

def estimate(time_series: pd.Series, CI: float, window: int) -> pd.DataFrame:
    (dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates) =\
         analytical_MPVS(time_series, CI = CI, smoothing = notched_smoothing(window = window), totals=True)
    print([len(_) for _ in (dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)])
    return pd.DataFrame(data = {
        "date": dates,
        "Rt": RR_pred,
        "Rt_upper": RR_CI_upper,
        "Rt_lower": RR_CI_lower,
        "total_cases": total_cases[2:],
        "new_cases": new_cases_ts,
    })

root = cwd()
data = root/"data"

# model details 
gamma  = 0.2
window = 7 * days
CI     = 0.95
smooth = notched_smoothing(window=window)

download_data(data, 'timeseries.json', "https://api.covid19india.org/v3/")

# data prep
with (data/'timeseries.json').open("rb") as fp:
    df = flat_table.normalize(pd.read_json(fp)).fillna(0)
df.columns = df.columns.str.split('.', expand = True)
dates = np.squeeze(df["index"][None].values)
df = df.drop(columns = "index").set_index(dates).stack([1, 2]).drop("UN", axis = 1)
all_estimates = pd.DataFrame()

geos = sorted(df.columns)
print("dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates")
for geography in ["WB"]: #["DL", "AP", "AS", "CH", "CT", "HP", "MN", "MP", "OR", "PY", "UT", "WB"]:
    print(geography, end=" ")
    try: 
        geo_estimate = estimate(df[geography][:, "total", "confirmed"], CI, window)
        geo_estimate["state"] = geography

        T  = geo_estimate["total_cases"].values
        dT = geo_estimate["new_cases"].values
        N  = -len(T)

        D  = smooth(df[geography][:, "total", "deceased"])[N:]
        dD = smooth(df[geography][:, "delta", "deceased"])[N:]

        R  = smooth(df[geography][:, "total", "recovered"])[N:]
        dR = smooth(df[geography][:, "delta", "recovered"])[N:]

        # testing rates
        Ts  = smooth(df[geography][:, "total", "tested"])[N:]
        dTs = smooth(df[geography][:, "delta", "tested"])[N:]
        
        # active infections
        I  =  T -  D -  R
        dI = dT - dD - dR 

        geo_estimate["cases"]                = dT
        geo_estimate["total_cases"]          =  T
    
        geo_estimate["recovered"]            = dR
        geo_estimate["total_recovered"]      =  R
    
        geo_estimate["deceased"]             = dD
        geo_estimate["total_deceased"]       =  D 
    
        geo_estimate["tested"]               = dTs
        geo_estimate["total_tested"]         =  Ts

        geo_estimate["cfr"]                  = (dD/dT).clip(min=0)
        geo_estimate["total_cfr"]            =  (D/ T).clip(min=0)
    
        geo_estimate["active"]               = dI
        geo_estimate["total_active"]         =  I

        geo_estimate["active_per_mn"]        = dI/population[geography]
        geo_estimate["total_active_per_mn"]  =  I/population[geography]
        
        geo_estimate["recovery_rate"]        = dR/(population[geography]*1e6) * 100
        geo_estimate["total_recovery_rate"]  =  R/(population[geography]*1e6) * 100
        
        geo_estimate["infection_rate"]       = (dT/dTs).clip(min=0)
        geo_estimate["total_infection_rate"] =  (T/ Ts).clip(min=0)

        all_estimates = pd.concat([all_estimates, geo_estimate[save_columns]])
        print("- success")
    except Exception as e:
        print(f"- failure: {e}")

# print(all_estimates.state.unique())
# plot_cols = [ "Rt", 
#     "cases", "total_cases", 
#     "recovered", "total_recovered", 
#     "deceased", "total_deceased", 
#     "active", "total_active", 
#     "active_per_mn", "total_active_per_mn", 
#     "cfr", "total_cfr", 
#     "infection_rate", "total_infection_rate", 
#     "recovery_rate", "total_recovery_rate" 
# ]

# for col in plot_cols[::-1]: 
#     plt.figure() 
#     plt.plot(all_estimates[col]) 
#     plt.title(col, loc="left") 
# plt.show()

# all_estimates.to_csv(data/"aux_metrics.csv")
