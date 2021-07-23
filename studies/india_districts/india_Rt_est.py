import pandas as pd

from epimargin.estimators import analytical_MPVS
from epimargin.etl.commons import download_data
from epimargin.etl.covid19india import data_path, get_time_series, load_all_data
import epimargin.plots as plt
from epimargin.smoothing import notched_smoothing
from epimargin.utils import cwd

# model details
CI        = 0.95
smoothing = 10

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in range(3, 31)]
}

for target in paths['v3'] + paths['v4']:
    try: 
        download_data(data, target)
    except Exception as e:
        print("error", target, e)
        pass 

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)
data_recency = str(df["date_announced"].max()).split()[0]
run_date     = str(pd.Timestamp.now()).split()[0]

ts = get_time_series(df, "detected_state")

states = ["Tamil Nadu", "Karnataka"] #["Maharashtra", "Punjab", "West Bengal", "Bihar", "Delhi", "Andhra Pradesh", "Telangana", "Tamil Nadu", "Madhya Pradesh"]

for state in states: 
    print(state)
    print("  + running estimation...")
    (
        dates,
        Rt_pred, Rt_CI_upper, Rt_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = analytical_MPVS(ts.loc[state].Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)
    estimates = pd.DataFrame(data = {
        "dates": dates,
        "Rt_pred": Rt_pred,
        "Rt_CI_upper": Rt_CI_upper,
        "Rt_CI_lower": Rt_CI_lower,
        "T_pred": T_pred,
        "T_CI_upper": T_CI_upper,
        "T_CI_lower": T_CI_lower,
        "total_cases": total_cases[2:],
        "new_cases_ts": new_cases_ts,
    })
    print("  + Rt today:", Rt_pred[-5:])

    plt.Rt(dates, Rt_pred, Rt_CI_lower, Rt_CI_upper, CI)\
        .ylabel("Estimated $R_t$")\
        .xlabel("Date")\
        .title(state)\
        .size(11, 8)\
        .save(figs/f"Rt_est_{state}.png", dpi=600, bbox_inches="tight")\
        .show()

    estimates["anomaly"] = estimates["dates"].isin(set(anomaly_dates))
    estimates.to_csv(data/f"india_rt_data_{state}_{data_recency}_run{run_date}.csv")

tn_ts = get_time_series(df.query("detected_state == 'Tamil Nadu'"), "detected_district") 
for district in tn_ts.index.get_level_values(0).unique()[19:]: 
    print(district)
    print("  + running estimation...")
    (
        dates,
        Rt_pred, Rt_CI_upper, Rt_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = analytical_MPVS(tn_ts.loc[district].Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)
    estimates = pd.DataFrame(data = {
        "dates": dates,
        "Rt_pred": Rt_pred,
        "Rt_CI_upper": Rt_CI_upper,
        "Rt_CI_lower": Rt_CI_lower,
        "T_pred": T_pred,
        "T_CI_upper": T_CI_upper,
        "T_CI_lower": T_CI_lower,
        "total_cases": total_cases[2:],
        "new_cases_ts": new_cases_ts,
    })
    estimates.to_csv(data/f"TN_Rt_data_{district}_{data_recency}_run{run_date}.csv")
    print("  + Rt today:", Rt_pred[-1])

    plt.figure()
    plt.Rt(dates, Rt_pred, Rt_CI_lower, Rt_CI_upper, CI)\
        .ylabel("Estimated $R_t$")\
        .xlabel("Date")\
        .title(district)\
        .size(11, 8)\
        .save(figs/f"Rt_est_TN{district}.png", dpi=600, bbox_inches="tight")#\
        #.show()


mh_ts = get_time_series(df.query("detected_state == 'Maharashtra'"), "detected_district") 
for district in mh_ts.index.get_level_values(0).unique()[-3:]: 
    print(district)
    print("  + running estimation...")
    (
        dates,
        Rt_pred, Rt_CI_upper, Rt_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = analytical_MPVS(mh_ts.loc[district].Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)
    estimates = pd.DataFrame(data = {
        "dates": dates,
        "Rt_pred": Rt_pred,
        "Rt_CI_upper": Rt_CI_upper,
        "Rt_CI_lower": Rt_CI_lower,
        "T_pred": T_pred,
        "T_CI_upper": T_CI_upper,
        "T_CI_lower": T_CI_lower,
        "total_cases": total_cases[2:],
        "new_cases_ts": new_cases_ts,
    })
    estimates.to_csv(data/f"MH_Rt_data_{district}_{data_recency}_run{run_date}.csv")
    print("  + Rt today:", Rt_pred[-1])

    plt.figure()
    plt.Rt(dates, Rt_pred, Rt_CI_lower, Rt_CI_upper, CI)\
        .ylabel("Estimated $R_t$")\
        .xlabel("Date")\
        .title(district)\
        .size(11, 8)\
        .save(figs/f"Rt_est_MH{district}.png", dpi=600, bbox_inches="tight")#\
        #.show()
    plt.close()


mp_ts = get_time_series(df.query("detected_state == 'Madhya Pradesh'"), "detected_district") 
for district in mp_ts.index.get_level_values(0).unique(): 
    print(district)
    print("  + running estimation...")
    try:
        (
            dates,
            Rt_pred, Rt_CI_upper, Rt_CI_lower,
            T_pred, T_CI_upper, T_CI_lower,
            total_cases, new_cases_ts,
            anomalies, anomaly_dates
        ) = analytical_MPVS(mp_ts.loc[district].Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing), totals = False)
    except Exception as e:
        print(e)
        continue
    estimates = pd.DataFrame(data = {
        "dates": dates,
        "Rt_pred": Rt_pred,
        "Rt_CI_upper": Rt_CI_upper,
        "Rt_CI_lower": Rt_CI_lower,
        "T_pred": T_pred,
        "T_CI_upper": T_CI_upper,
        "T_CI_lower": T_CI_lower,
        "total_cases": total_cases[2:],
        "new_cases_ts": new_cases_ts,
    })
    estimates.to_csv(data/f"MP_Rt_data_{district}_{data_recency}_run{run_date}.csv")
    print("  + Rt today:", Rt_pred[-1])

    plt.figure()
    plt.Rt(dates, Rt_pred, Rt_CI_lower, Rt_CI_upper, CI)\
        .ylabel("Estimated $R_t$")\
        .xlabel("Date")\
        .title(district)\
        .size(11, 8)\
        .save(figs/f"Rt_est_MP{district}.png", dpi=600, bbox_inches="tight")#\
        #.show()
    plt.close()

from matplotlib.dates import DateFormatter
formatter = DateFormatter("%b\n%Y")

f = notched_smoothing(window = smoothing)
plt.plot(ts.loc["Maharashtra"].index, ts.loc["Maharashtra"].Hospitalized, color = "black", label = "raw case counts from API")
plt.plot(ts.loc["Maharashtra"].index, f(ts.loc["Maharashtra"].Hospitalized), color = "black", linestyle = "dashed", alpha = 0.5, label = "smoothed, seasonality-adjusted case counts")
plt.PlotDevice()\
    .l_title("daily case counts in Maharashtra")\
    .axis_labels(x = "date", y = "daily cases")
plt.gca().xaxis.set_major_formatter(formatter)
plt.legend(prop = plt.theme.note, handlelength = 1, framealpha = 0)
plt.show()