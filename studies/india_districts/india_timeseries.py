from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import flat_table
from epimargin.estimators import analytical_MPVS
from epimargin.etl.commons import download_data
from epimargin.smoothing import notched_smoothing
from epimargin.utils import days, fillna, setup

simplefilter("ignore")

sns.set(palette="bright", style="darkgrid", font="Helvetica Neue")
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

# pipeline details, options
gamma  = 0.2
window = 5 * days
CI     = 0.95
smooth = notched_smoothing(window)
start_date = pd.Timestamp(year = 2020, month = 3, day = 1)
time_period = 120

def estimate(time_series: pd.Series) -> pd.DataFrame:
    estimates = analytical_MPVS(time_series, CI = CI, smoothing = smooth, totals=True)
    return pd.DataFrame(data = {
        "date": estimates[0],
        "Rt": estimates[1],
        "Rt_upper": estimates[2],
        "Rt_lower": estimates[3],
        "total_cases": estimates[-4][2:],
        "new_cases": estimates[-3],
    })

data, figs = setup()

download_data(data, 'timeseries.json', "https://api.covid19india.org/v3/")
download_data(data, 'state_wise.csv',  "https://api.covid19india.org/v3/")
download_data(data, 'states.csv',      "https://api.covid19india.org/v3/")
download_data(data, 'districts.csv',   "https://api.covid19india.org/v3/")

# data prep
with (data/'timeseries.json').open("rb") as fp:
    df = flat_table.normalize(pd.read_json(fp)).fillna(0)
df.columns = df.columns.str.split('.', expand = True)
dates = np.squeeze(df["index"][None].values)
df = df.drop(columns = "index")\
    .set_index(dates)\
    .stack([1, 2])\
    .drop("UN", axis = 1)\
    .fillna(0)

# drop last 2 days to avoid count drops 
df = df[(start_date <= df.index.get_level_values(0)) & (df.index.get_level_values(0) <= pd.Timestamp.now().normalize() - pd.Timedelta(days = 2))]

all_estimates = pd.DataFrame()

dfs = {}
for geography in sorted(df.columns):
    print(geography, end=" ")
    try: 
        geo_estimate = estimate(df[geography][:, "total", "confirmed"])
        geo_estimate["state"] = geography

        T  = geo_estimate["total_cases"].values.astype(int)
        dT = geo_estimate["new_cases"].values.astype(int)
        N  = -len(T)

        D  = smooth(df[geography][:, "total", "deceased"])[-time_period:].astype(int)
        dD = smooth(df[geography][:, "delta", "deceased"])[-time_period:].astype(int)

        R  = smooth(df[geography][:, "total", "recovered"])[-time_period:].astype(int)
        dR = smooth(df[geography][:, "delta", "recovered"])[-time_period:].astype(int)

        # testing rates
        Ts  = smooth(df[geography][:, "total", "tested"])[-time_period:].astype(int)
        dTs = smooth(df[geography][:, "delta", "tested"])[-time_period:].astype(int)
        
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

        geo_estimate["cfr"]                  = ((dD/dT).clip(min=0))
        geo_estimate["total_cfr"]            =  ((D/ T).clip(min=0))
    
        geo_estimate["active"]               = dI
        geo_estimate["total_active"]         =  I

        geo_estimate["active_per_mn"]        = dI/population[geography]
        geo_estimate["total_active_per_mn"]  =  I/population[geography]
        
        geo_estimate["recovery_rate"]        = dR/(population[geography]*1e6) * 100
        geo_estimate["total_recovery_rate"]  =  R/(population[geography]*1e6) * 100
        
        geo_estimate["infection_rate"]       = ((dT/dTs).clip(min=0))
        geo_estimate["total_infection_rate"] =  ((T/ Ts).clip(min=0))

        all_estimates = pd.concat([all_estimates, geo_estimate[save_columns]])
        dfs[geography] = geo_estimate
        print("- success")
    except ValueError as e:
        print(f"- failure: {e}")

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
# plot_cols = ["cases", "recovered", "deceased", "active", "cfr", "infection_rate", "recovery_rate"]
# geo = "DL"
# plot_RR_est(dfs[geo].date, dfs[geo].Rt, dfs[geo].Rt_upper, dfs[geo].Rt_lower, 0.95)\
#     .title(geo).show()
# for col in plot_cols[::-1]: 
#     plt.figure() 
#     plt.plot(dfs[geo][col]) 
#     plt.title(col, loc="left") 
# plt.show()


all_estimates.to_csv(data/"aux_metrics.csv")
