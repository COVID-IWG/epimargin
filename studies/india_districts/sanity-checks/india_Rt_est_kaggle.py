import matplotlib.pyplot as plt
import pandas as pd

from epimargin.estimators import box_filter, analytical_MPVS
from epimargin.plots import plot_RR_est, plot_T_anomalies
from epimargin.utils import cwd
from etl import download_data, get_time_series, load_all_data

# model details
CI        = 0.99
smoothing = 5

root = cwd()
data = root/"data"
figs = root/"figs/comparison/kaggle"

states = ["Maharashtra"]#, "Bihar", "Delhi", "Andhra Pradesh", "Telangana", "Tamil Nadu", "Madhya Pradesh"]

kaggle = pd.read_csv(data/"covid_19_india.csv", parse_dates=[1], dayfirst=True).set_index("Date")

for state in states: 
    print(state)
    print("  + running estimation...")
    (
        dates,
        RR_pred, RR_CI_upper, RR_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = analytical_MPVS(kaggle[kaggle["State/UnionTerritory"] == state].Confirmed, CI = CI, smoothing = lambda ts: box_filter(ts, smoothing, 3))

    estimates = pd.DataFrame(data = {
        "dates": dates,
        "RR_pred": RR_pred,
        "RR_CI_upper": RR_CI_upper,
        "RR_CI_lower": RR_CI_lower,
        "T_pred": T_pred,
        "T_CI_upper": T_CI_upper,
        "T_CI_lower": T_CI_lower,
        "total_cases": total_cases[2:],
        "new_cases_ts": new_cases_ts,
    })
    print("  + Rt today:", RR_pred[-1])
    
    plot_RR_est(dates, RR_pred, RR_CI_lower, RR_CI_upper, CI)\
        .ylabel("Estimated $R_t$")\
        .xlabel("Date")\
        .title(state)\
        .size(11, 8)\
        .save(figs/f"Rt_est_{state}.png", dpi=600, bbox_inches="tight")\
        .show()

    plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)\
        .ylabel("Predicted/Observed Cases")\
        .xlabel("Date")\
        .title(state)\
        .size(11, 8)\
        .save(figs/f"T_est_{state}.png", dpi=600, bbox_inches="tight")\
        .show()



    # estimates["anomaly"] = estimates["dates"].isin(set(anomaly_dates))
    # estimates.to_csv(data/f"india_rt_data{data_recency}_run{run_date}.csv")
