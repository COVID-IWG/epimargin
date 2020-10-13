import matplotlib.pyplot as plt
import pandas as pd

from adaptive.estimators import analytical_MPVS
from adaptive.etl.commons import download_data
from adaptive.etl.covid19india import data_path, get_time_series, load_all_data
import adaptive.plots as plt
from adaptive.smoothing import notched_smoothing
from adaptive.utils import cwd

# model details
CI        = 0.95
smoothing = 14

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in (3, 4, 5, 6, 7)]
}

for target in paths['v3'] + paths['v4']:
    download_data(data, target)

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
        Rt_pred, RR_CI_upper, RR_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = analytical_MPVS(ts.loc[state].Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing))
    estimates = pd.DataFrame(data = {
        "dates": dates,
        "Rt_pred": Rt_pred,
        "RR_CI_upper": RR_CI_upper,
        "RR_CI_lower": RR_CI_lower,
        "T_pred": T_pred,
        "T_CI_upper": T_CI_upper,
        "T_CI_lower": T_CI_lower,
        "total_cases": total_cases[2:],
        "new_cases_ts": new_cases_ts,
    })
    print("  + Rt today:", Rt_pred[-1])

    # plt.Rt(dates, Rt_pred, RR_CI_lower, RR_CI_upper, CI)\
    #     .ylabel("Estimated $R_t$")\
    #     .xlabel("Date")\
    #     .title(state)\
    #     .size(11, 8)\
    #     .save(figs/f"Rt_est_{state}.png", dpi=600, bbox_inches="tight")\
    #     .show()

    # plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)\
    #     .ylabel("Predicted/Observed Cases")\
    #     .xlabel("Date")\
    #     .title(state)\
    #     .size(11, 8)\
    #     .save(figs/f"T_est_{state}.png", dpi=600, bbox_inches="tight")\
    #     .show()

    estimates["anomaly"] = estimates["dates"].isin(set(anomaly_dates))
    estimates.to_csv(data/f"india_rt_data{data_recency}_run{run_date}.csv")
