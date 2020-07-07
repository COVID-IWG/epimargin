import matplotlib.pyplot as plt
import pandas as pd

from adaptive.estimators import gamma_prior
from adaptive.etl.covid19india import (data_path, download_data,
                                       get_time_series, load_all_data)
from adaptive.plots import plot_RR_est, plot_T_anomalies
from adaptive.smoothing import convolution
from adaptive.utils import cwd

# model details
CI        = 0.95
smoothing = 14

root = cwd()
data = root/"data"
figs = root/"figs"

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in (3, 4, 5, 6, 7)]
}

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)
data_recency = str(df["date_announced"].max()).split()[0]
run_date     = str(pd.Timestamp.now()).split()[0]

ts = get_time_series(df, "detected_state")

states = ["Maharashtra", "Bihar", "Delhi", "Andhra Pradesh", "Telangana", "Tamil Nadu", "Madhya Pradesh"]

for state in states: 
    print(state)
    print("  + running estimation...")
    (
        dates,
        RR_pred, RR_CI_upper, RR_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = gamma_prior(ts.loc[state].Hospitalized, CI = CI, smoothing = convolution(window = smoothing))
    # estimates = pd.DataFrame(data = {
    #     "dates": dates,
    #     "RR_pred": RR_pred,
    #     "RR_CI_upper": RR_CI_upper,
    #     "RR_CI_lower": RR_CI_lower,
    #     "T_pred": T_pred,
    #     "T_CI_upper": T_CI_upper,
    #     "T_CI_lower": T_CI_lower,
    #     "total_cases": total_cases[2:],
    #     "new_cases_ts": new_cases_ts,
    # })
    print("  + Rt today:", RR_pred[-1])

    plot_RR_est(dates, RR_pred, RR_CI_lower, RR_CI_upper, CI)\
        .ylabel("Estimated $R_t$")\
        .xlabel("Date")\
        .title(state)\
        .size(11, 8)\
        .save(figs/f"Rt_est_{state}.png", dpi=600, bbox_inches="tight")\
        .show()

    # plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)\
    #     .ylabel("Predicted/Observed Cases")\
    #     .xlabel("Date")\
    #     .title(state)\
    #     .size(11, 8)\
    #     .save(figs/f"T_est_{state}.png", dpi=600, bbox_inches="tight")\
    #     .show()



    estimates["anomaly"] = estimates["dates"].isin(set(anomaly_dates))
    estimates.to_csv(data/f"india_rt_data{data_recency}_run{run_date}.csv")
