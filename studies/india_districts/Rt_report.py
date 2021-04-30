import epimargin.plots as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from epimargin.estimators import analytical_MPVS
from epimargin.smoothing import notched_smoothing
from epimargin.utils import cwd

# model details
CI        = 0.95
smoothing = 10

plt.set_theme("substack")

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

lookback = 120
cutoff = 2

state ="Maharashtra"
state_code = "MH"

state_ts = pd.read_csv("/Users/satej/Downloads/pipeline_raw_state_case_timeseries.csv")\
    .set_index("detected_state")

district_ts = pd.read_csv("/Users/satej/Downloads/pipeline_raw_district_case_timeseries.csv")\
    .set_index(["detected_state", "detected_district"])\
    .loc[state]

(
    dates,
    Rt_pred, Rt_CI_upper, Rt_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = analytical_MPVS(state_ts.loc[state].set_index("status_change_date").iloc[-lookback:-cutoff].Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)

gdf = gpd.read_file("data/maharashtra.json", dpi = 600)
district_Rt = {}

excluded = ["Unknown", "Other State", "Airport Quarantine", "Railway Quarantine"]
for district in filter(lambda _: _ not in excluded, district_ts.index.get_level_values(0).unique()):
    try:
        (
            dates,
            Rt_pred, Rt_CI_upper, Rt_CI_lower,
            T_pred, T_CI_upper, T_CI_lower,
            total_cases, new_cases_ts,
            anomalies, anomaly_dates
        ) = analytical_MPVS(district_ts.loc[district].set_index("status_change_date").iloc[-(cutoff*10):-cutoff].Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)
        district_Rt[district] = Rt_pred[-1]
    except Exception:
        district_Rt[district] = np.nan

top10 = [(k, f"{v:.2f}") for (k, v) in sorted(district_Rt.items(), key = lambda t:t[1], reverse = True)[:10]]