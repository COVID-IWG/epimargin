import pandas as pd

from epimargin.estimators import analytical_MPVS
from epimargin.etl.commons import download_data
from epimargin.etl.covid19india import data_path, get_time_series, load_all_data
import epimargin.plots as plt
from epimargin.smoothing import notched_smoothing
from epimargin.utils import cwd

# model details
CI        = 0.95
smoothing = 14
infectious_period = 10

root = cwd()
data = root/"data"
figs = root/"figs"

data.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

plt.set_theme("substack")
# define data versions for api files
paths = {
    "v3": [data_path(i) for i in (1, 2)],
    "v4": [data_path(i) for i in range(3, 26)]
}

# for target in paths['v3'] + paths['v4']:
#     download_data(data, target)

df = load_all_data(
    v3_paths = [data/filepath for filepath in paths['v3']], 
    v4_paths = [data/filepath for filepath in paths['v4']]
)
data_recency = str(df["date_announced"].max()).split()[0]
run_date     = str(pd.Timestamp.now()).split()[0]

ts = get_time_series(df, "detected_state")

states = ["Maharashtra", "Punjab", "West Bengal", "Bihar", "Delhi", "Andhra Pradesh", "Telangana", "Tamil Nadu", "Madhya Pradesh"]


for state in states[:1]: 
    print(state)
    print("  + running estimation...")
    
    (
        inf_dates,
        inf_Rt_pred, inf_Rt_CI_upper, inf_Rt_CI_lower,
        inf_T_pred, inf_T_CI_upper, inf_T_CI_lower,
        inf_total_cases, inf_new_cases_ts,
        inf_anomalies, inf_anomaly_dates
    ) = analytical_MPVS(ts.loc[state].Hospitalized, CI = CI, smoothing = notched_smoothing(window = smoothing), infectious_period = infectious_period, totals = False)
    inf_estimates = pd.DataFrame(data = {
        "dates": inf_dates,
        "Rt_pred": inf_Rt_pred,
        "Rt_CI_upper": inf_Rt_CI_upper,
        "Rt_CI_lower": inf_Rt_CI_lower,
        "T_pred": inf_T_pred,
        "T_CI_upper": inf_T_CI_upper,
        "T_CI_lower": inf_T_CI_lower,
        "total_cases": inf_total_cases[2:],
        "new_cases_ts": inf_new_cases_ts,
    })
    inf_estimates["anomaly"] = inf_estimates["dates"].isin(set(inf_anomaly_dates))
    print("  + Rt (inf) today:", inf_Rt_pred[-1])

    (
        dth_dates,
        dth_Rt_pred, dth_Rt_CI_upper, dth_Rt_CI_lower,
        dth_T_pred, dth_T_CI_upper, dth_T_CI_lower,
        dth_total_cases, dth_new_cases_ts,
        dth_anomalies, dth_anomaly_dates
    ) = analytical_MPVS(ts.loc[state].Deceased, CI = CI, smoothing = notched_smoothing(window = smoothing), infectious_period = infectious_period, totals = False)
    dth_estimates = pd.DataFrame(data = {
        "dates": dth_dates,
        "Rt_pred": dth_Rt_pred,
        "Rt_CI_upper": dth_Rt_CI_upper,
        "Rt_CI_lower": dth_Rt_CI_lower,
        "T_pred": dth_T_pred,
        "T_CI_upper": dth_T_CI_upper,
        "T_CI_lower": dth_T_CI_lower,
        "total_cases": dth_total_cases[2:],
        "new_cases_ts": dth_new_cases_ts,
    })
    dth_estimates["anomaly"] = dth_estimates["dates"].isin(set(dth_anomaly_dates))
    print("  + Rt (dth) today:", inf_Rt_pred[-1])

    fig, axs = plt.subplots(1, 2, sharey = True)
    plt.sca(axs[0])
    plt.Rt(inf_dates, inf_Rt_pred, inf_Rt_CI_lower, inf_Rt_CI_upper, CI)\
        .axis_labels("date", "$R_t$")
    plt.title("estimated from infections", loc = "left", fontdict = plt.theme.label)

    # fig, axs = plt.subplots(3, 1, sharex = True)
    # plt.sca(axs[0])
    # plt.plot(dth_dates, delhi_dD_smoothed[2:], color = "orange")
    # plt.title("d$D$/d$t$", loc = "left", fontdict = plt.theme.label)
    
    # plt.sca(axs[1])
    # plt.plot(dth_dates, np.diff(delhi_dD_smoothed)[1:], color = "red")
    # plt.title("d$^2D$/d$t^2$", loc = "left", fontdict = plt.theme.label)
    
    plt.sca(axs[1])
    plt.Rt(dth_dates, dth_Rt_pred, dth_Rt_CI_lower, dth_Rt_CI_upper, CI)\
        .axis_labels("date", "$R_t$")
    plt.title("estimated from deaths", loc = "left", fontdict = plt.theme.label)
    plt.PlotDevice()\
        .title(f"$R_t$ estimates: {state}", ha = "center", x = 0.5)\
        .adjust(left = 0.08, right = 0.96)
    plt.show()

    inf_estimates.to_csv(data/f"india_rt_comparison_inf_{state}_{data_recency}_run{run_date}.csv")
    dth_estimates.to_csv(data/f"india_rt_comparison_dth_{state}_{data_recency}_run{run_date}.csv")
