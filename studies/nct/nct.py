import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epimargin.estimators import analytical_MPVS
from epimargin.etl.commons import download_data
from epimargin.etl.covid19india import data_path, get_time_series, load_all_data
from epimargin.model import Model, ModelUnit
from epimargin.plots import PlotDevice, plot_RR_est, plot_T_anomalies
from epimargin.smoothing import convolution
from epimargin.utils import cwd

# model details
CI        = 0.99
smoothing = 10

if __name__ == "__main__":
    root   = cwd()
    data   = root/"data"
    output = root/"output"
    if not data.exists():
        data.mkdir()
    if not output.exists():
        output.mkdir()

    # define data versions for api files
    paths = {
        "v3": [data_path(i) for i in (1, 2)],
        "v4": [data_path(i) for i in (3, 4, 5, 6, 7, 8)]
    }

    # download data from india covid 19 api
    for target in paths['v3'] + paths['v4']:
        download_data(data, target)

    df = load_all_data(
        v3_paths = [data/filepath for filepath in paths['v3']], 
        v4_paths = [data/filepath for filepath in paths['v4']]
    )
    data_recency = str(df["date_announced"].max()).split()[0]
    run_date     = str(pd.Timestamp.now()).split()[0]

    ts = get_time_series(df[df.detected_state == "Delhi"])

    (
        dates,
        RR_pred, RR_CI_upper, RR_CI_lower,
        T_pred, T_CI_upper, T_CI_lower,
        total_cases, new_cases_ts,
        anomalies, anomaly_dates
    ) = analytical_MPVS(ts.delta[ts.delta > 0], CI = CI, smoothing = convolution(window = smoothing)) 
    #= analytical_MPVS(ts.Hospitalized[ts.Hospitalized > 0], CI = CI, smoothing = lambda ts: box_filter(ts, smoothing, 10))

    np.random.seed(33)
    delhi = Model([ModelUnit("Delhi", 18_000_000, I0 = T_pred[-1], RR0 = RR_pred[-1], mobility = 0)])
    delhi.run(14, np.zeros((1,1)))

    t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(len(delhi[0].delta_T))]

    plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI)
    PlotDevice().title("Delhi: Reproductive Number Estimate").xlabel("Date").ylabel("Rt", rotation=0, labelpad=20)
    plt.show()
    
    delhi[0].lower_CI[0] = T_CI_lower[-1]
    delhi[0].upper_CI[0] = T_CI_upper[-1]
    print(delhi[0].delta_T)
    print(delhi[0].lower_CI)
    print(delhi[0].upper_CI)
    plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)
    plt.scatter(t_pred, delhi[0].delta_T, color = "tomato", s = 4, label = "Predicted Net Cases")
    plt.fill_between(t_pred, delhi[0].lower_CI, delhi[0].upper_CI, color = "tomato", alpha = 0.3, label="99% CI (forecast)")
    plt.legend()
    PlotDevice().title("Delhi: Net Daily Cases").xlabel("Date").ylabel("Cases")
    plt.show()

    pd.DataFrame(data={
        "date"       : dates, 
        "Rt"         : RR_pred, 
        "Rt_CI_upper": RR_CI_upper, 
        "Rt_CI_lower": RR_CI_lower
    }).set_index("date").to_csv(output/"Rt.csv")
    
    pd.DataFrame(data={
        "date"                    : list(dates) + t_pred[1:], 
        "net_daily_cases"         : T_pred + delhi[0].delta_T[1:], 
        "net_daily_cases_CI_upper": T_CI_upper + delhi[0].upper_CI[1:],
        "net_daily_cases_CI_lower": T_CI_lower + delhi[0].lower_CI[1:]
    }).set_index("date").to_csv(output/"dT.csv")
