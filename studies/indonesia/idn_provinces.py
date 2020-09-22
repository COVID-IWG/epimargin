import json
from logging import getLogger
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import adaptive.plots as plt
from adaptive.estimators import analytical_MPVS, linear_projection
from adaptive.etl.commons import download_data
from adaptive.model import Model
from adaptive.smoothing import notched_smoothing
from adaptive.utils import days, setup

logger = getLogger("IDNPROV")

provinces = (jakarta, central_java) = [
    'DKI JAKARTA',
    'JAWA TENGAH',
]

replacements = { 
    'YOGYAKARTA'     : "DAERAH ISTIMEWA YOGYAKARTA",
    'BANGKA BELITUNG': "KEPULAUAN BANGKA BELITUNG",
    'JAKARTA RAYA'   : "DKI JAKARTA"
}

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
    download_data(data, filename(province), base_url = "https://data.covid19.go.id/public/api/")

province_cases = {province: load_province_timeseries(data, province) for province in provinces}
bgn = min(cases.index.min() for cases in province_cases.values())
end = max(cases.index.max() for cases in province_cases.values())
idx = pd.date_range(bgn, end)
province_cases = {province: cases.reindex(idx, method = "pad").fillna(0) for (province, cases) in province_cases.items()}

logger.info("running DKIJ-level Rt estimate")
(dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
     = analytical_MPVS(province_cases[jakarta], CI = CI, smoothing = smoothing) 

plt.Rt(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0.75, ymax=4)\
    .title("\nJakarta")\
    .xlabel("\ndate")\
    .ylabel("$R_t$", rotation=0, labelpad=30)\
    .show()

plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)\
    .title("\nJakarta")\
    .xlabel("\ndate")\
    .ylabel("cases")\
    .show()

logger.info("running JT-level Rt estimate")
(dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
     = analytical_MPVS(province_cases[central_java], CI = CI, smoothing = smoothing) 

plt.Rt(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0.75, ymax=4)\
    .title("\nCentral Java")\
    .xlabel("\ndate")\
    .ylabel("$R_t$", rotation=0, labelpad=30)\
    .show()

plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)\
    .title("\nCentral Java")\
    .xlabel("\ndate")\
    .ylabel("cases")\
    .show()