import json
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adaptive.estimators import analytical_MPVS
from adaptive.etl.commons import download_data
from adaptive.plots import plot_RR_est, plot_T_anomalies
from adaptive.smoothing import notched_smoothing
from adaptive.utils import days, setup

logger = getLogger("IDN")

provinces = ['DKI JAKARTA', 'JAWA TIMUR', 'JAWA TENGAH', 'SULAWESI SELATAN',
    'JAWA BARAT', 'KALIMANTAN SELATAN', 'SUMATERA UTARA', 'BALI',
    'SUMATERA SELATAN', 'PAPUA', 'SULAWESI UTARA', 'KALIMANTAN TIMUR',
    'BANTEN', 'NUSA TENGGARA BARAT', 'KALIMANTAN TENGAH', 'GORONTALO',
    'MALUKU UTARA', 'MALUKU', 'SUMATERA BARAT', 'SULAWESI TENGGARA',
    'RIAU', 'ACEH', 'DAERAH ISTIMEWA YOGYAKARTA', 'KEPULAUAN RIAU',
    'PAPUA BARAT', 'KALIMANTAN BARAT', 'LAMPUNG', 'KALIMANTAN UTARA',
    'SULAWESI BARAT', 'BENGKULU', 'JAMBI', 'SULAWESI TENGAH',
    'KEPULAUAN BANGKA BELITUNG', 'NUSA TENGGARA TIMUR'
]

# model/sim details
gamma     = 0.2
window    = 7
CI        = 0.95
smoothing = notched_smoothing(window = window)

# data fields
date_scale  = 1000000.0
date        = "tanggal"
timeseries  = "list_perkembangan"
total_cases = "AKUMULASI_KASUS"

filename = lambda province: "prov_detail_{}.json".format(province.replace(" ", "_"))

def load_province_timeseries(data_path: Path, province: str) -> pd.DataFrame:
    with (data_path/filename(province)).open() as fp:
        top_level = json.load(fp)
    df = pd.DataFrame([(_[date], _[total_cases]) for _ in top_level[timeseries]], columns=["date", "total_cases"])
    df["date"] = (date_scale * df["date"]).apply(pd.Timestamp)
    return df.set_index("date")


(data, figs) = setup(level = "INFO")
for province in provinces:
    logger.info("downloading data for %s", province)
    #download_data(data, filename(province), base_url = "https://data.covid19.go.id/public/api/")

province_cases = {province: load_province_timeseries(data, province) for province in provinces}
bgn = min(cases.index.min() for cases in province_cases.values())
end = max(cases.index.max() for cases in province_cases.values())
idx = pd.date_range(bgn, end)
province_cases = {province: cases.reindex(idx, method = "pad").fillna(0) for (province, cases) in province_cases.items()}
natl_cases = sum(province_cases.values())


# national level projections
(
    dates,
    RR_pred, RR_CI_upper, RR_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = analytical_MPVS(natl_cases, CI = CI, smoothing = smoothing) 
plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0, ymax=4)\
    .title("Indonesia: Reproductive Number Estimate")\
    .xlabel("Date")\
    .ylabel("Rt", rotation=0, labelpad=20)
plt.ylim(0, 4)
plt.show()
