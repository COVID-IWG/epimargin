import pandas as pd

from adaptive.estimators import box_filter, gamma_prior
from adaptive.utils import cwd
from etl import download_data, get_time_series, load_all_data

# model details
CI        = 0.95
smoothing = 15

if __name__ == "__main__":
    root = cwd()
    data = root/"data"

    # define data versions for api files
    paths = {
        "v3": ["raw_data1.csv", "raw_data2.csv"],
        "v4": ["raw_data3.csv", "raw_data4.csv",
               "raw_data5.csv", "raw_data6.csv",
               "raw_data7.csv"]
    }

    # download data from india covid 19 api
    for target in paths['v3'] + paths['v4']:
        download_data(data, target)

    # run rolling regressions on historical national case data
    df = load_all_data(
        v3_paths = [data/filepath for filepath in paths['v3']], 
        v4_paths = [data/filepath for filepath in paths['v4']]
    )
    data_recency = str(df["date_announced"].max()).split()[0]
    run_date     = str(pd.Timestamp.now()).split()[0]

    ts = get_time_series(df)

    (
        dates,
        RR_pred, RR_CI_upper, RR_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = gamma_prior(ts.Hospitalized[ts.Hospitalized > 0], CI = CI, smoothing = lambda ts: box_filter(ts, smoothing, None))
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
    estimates["anomaly"] = estimates["dates"].isin(set(anomaly_dates))
    estimates.to_csv(data/f"india_rt_data{data_recency}_run{run_date}.csv")
