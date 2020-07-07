from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from adaptive.estimators import gamma_prior
from adaptive.plots import plot_RR_est, plot_T_anomalies
from adaptive.smoothing import convolution, kernels
from etl import data_path, download_data, get_time_series, load_all_data

sns.set_palette("deep")

def get_dataframe(data: Path = Path("./data")):
    # define data versions for api files
    paths = {
        "v3": [data_path(i) for i in (1, 2)],
        "v4": [data_path(i) for i in (3, 4, 5, 6, 7, 8)]
    }

    # download data from india covid 19 api
    for target in paths['v3'] + paths['v4']:
        download_data(data, target)

    return load_all_data(
        v3_paths = [data/filepath for filepath in paths['v3']], 
        v4_paths = [data/filepath for filepath in paths['v4']]
    )


def state_checks():
    root = cwd()
    data = root/"data"

    # model details
    gamma      = 0.2
    prevalence = 1

    dfn = get_dataframe()

    states = ["Maharashtra", "Andhra Pradesh", "Tamil Nadu", "Madhya Pradesh", "Punjab", "Gujarat", "Kerala", "Bihar"]

    # first, check reimplementation against known figures
    CI = 0.99
    for state in states: 
        ts = get_time_series(dfn[
            (dfn.date_announced <= "2020-05-08") & 
            (dfn.detected_state == state)
        ])
        (
            dates,
            RR_pred, RR_CI_upper, RR_CI_lower,
            T_pred, T_CI_upper, T_CI_lower,
            total_cases, new_cases_ts,
            anomalies, anomaly_date
        ) = gamma_prior(ts.Hospitalized, CI = CI)
        plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI)
        plt.title(f"{state}: Estimated $R_t$", loc="left", fontsize=20)
        plt.figure()
        plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
        plt.title(f"{state}: Predicted Cases", loc="left", fontsize=20)

        plt.show()

    # generate smoothing checks for national time series
    ts = get_time_series(dfn)

    CI = 0.95
    # 1: box filter
    RRfig, RRax = plt.subplots(3, 1, sharex=True)
    T_fig, T_ax = plt.subplots(3, 1, sharex=True)
    for (i, n) in enumerate((5, 10, 15)): 
        (
            dates, 
            RR_pred, RR_CI_upper, RR_CI_lower, 
            T_pred, T_CI_upper, T_CI_lower, 
            total_cases, new_cases_ts, 
            anomalies, anomaly_dates
        ) = gamma_prior(ts.Hospitalized, CI = CI, smoothing = lambda ts: box_filter(ts, n, None))
        
        plt.sca(RRax[i])
        plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI)
        plt.title(f"smoothing window: {n}, no local smoothing", loc = "left", fontsize = 16)
        
        plt.sca(T_ax[i])
        plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
        plt.title(f"smoothing window: {n}, no local smoothing", loc = "left", fontsize = 16)
    
    plt.figure(RRfig.number)
    RRfig.tight_layout()
    plt.suptitle("Estimated Rt (box filter convolution)", fontsize = 20)
    plt.figure(T_fig.number)
    T_fig.tight_layout()
    plt.suptitle("Estimated Daily Cases (box filter convolution)", fontsize = 20)
    plt.show()


    # 2: box filter, local smooothing
    RRfig, RRax = plt.subplots(3, 1, sharex=True)
    T_fig, T_ax = plt.subplots(3, 1, sharex=True)
    for (i, n) in enumerate((5, 10, 15)): 
        s = n//2 + 1
        (
            dates, 
            RR_pred, RR_CI_upper, RR_CI_lower, 
            T_pred, T_CI_upper, T_CI_lower, 
            total_cases, new_cases_ts, 
            anomalies, anomaly_dates
        ) = gamma_prior(ts.Hospitalized, CI = CI, smoothing = lambda ts: box_filter(ts, n, s))
        
        plt.sca(RRax[i])
        plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI)
        plt.title(f"smoothing window: {n}, local smoothing: last {s} points", loc = "left", fontsize = 16)
        
        plt.sca(T_ax[i])
        plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
        plt.title(f"smoothing window: {n}, local smoothing: last {s} points", loc = "left", fontsize = 16)
    
    plt.figure(RRfig.number)
    RRfig.tight_layout()
    plt.suptitle("Estimated Rt (box filter convolution, local smoothing)", fontsize = 20)
    plt.subplots_adjust(right = 0.9)

    plt.figure(T_fig.number)
    T_fig.tight_layout()
    plt.suptitle("Estimated Daily Cases (box filter convolution, local smoothing)", fontsize = 20)
    plt.subplots_adjust(right = 0.9)
    
    plt.show()


def implementation_checks(window: int = 7):
    df = get_dataframe()
    ts = get_time_series(df)

    I = ts.Hospitalized.values
    for kernel in kernels.keys():
        plt.figure()
        plt.bar(x = ts.index, height = I, label = "raw")
        smoother = convolution(kernel, window)
        plt.plot(ts.index, smoother(I)[:-window+1], label = "smooth")
        plt.title(f"kernel: {kernel} (window: {window})", loc = "left")
    plt.show()

implementation_checks(10)