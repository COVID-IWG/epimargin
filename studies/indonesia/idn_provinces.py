import json
from logging import getLogger
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import adaptive.plots as plt
from adaptive.estimators import analytical_MPVS, linear_projection
from adaptive.etl.commons import download_data
from adaptive.smoothing import notched_smoothing
from adaptive.utils import days, setup

logger = getLogger("IDNPROV")

# provinces = (jakarta, central_java) = [
#     'DKI JAKARTA',
#     'JAWA TENGAH',
# ]
provinces = [
    "DKI JAKARTA",
    "SULAWESI SELATAN",
    # "JAWA BARAT",
    # "JAWA TENGAH",
    # "JAWA TIMUR",
    # "BALI",
    # "KALIMANTAN SELATAN",
    # "SULAWESI UTARA",
    # "PAPUA"
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

priority_pops = { 
    "DKI JAKARTA"       : 10_154_134,
    "JAWA BARAT"        : 46_668_214, 
    "JAWA TENGAH"       : 33_753_023, 
    "JAWA TIMUR"        : 38_828_061, 
    "BALI"              : 4_148_588, 
    "KALIMANTAN SELATAN": 3_984_315, 
    "SULAWESI UTARA"    : 639_639, 
    "SULAWESI SELATAN"  : 8_512_608, 
    "PAPUA"             : 3_143_088
}

filename = lambda province: "prov_detail_{}.json".format(province.replace(" ", "_"))

def load_province_timeseries(data_path: Path, province: str, start_date: Optional[str] = None) -> pd.DataFrame:
    with (data_path/filename(province)).open() as fp:
        top_level = json.load(fp)
    df = pd.DataFrame([(_[date], _[total_cases]) for _ in top_level[timeseries]], columns=["date", "total_cases"])
    df["date"] = (date_scale * df["date"]).apply(pd.Timestamp)
    df.set_index("date", inplace = True)
    if start_date:
        return df[df.index >= start_date]
    return df 
    

(data, figs) = setup(level = "INFO")
for province in provinces:
    logger.info("downloading data for %s", province)
    download_data(data, filename(province), base_url = "https://data.covid19.go.id/public/api/")

province_cases = {province: load_province_timeseries(data, province, "Apr 1, 2020") for province in provinces}
bgn = min(cases.index.min() for cases in province_cases.values())
end = max(cases.index.max() for cases in province_cases.values())
idx = pd.date_range(bgn, end)
province_cases = {province: cases.reindex(idx, method = "pad").fillna(0) for (province, cases) in province_cases.items()}

prediction_period = 14*days
for province in provinces:
    title = province.title().replace("Dki", "DKI")
    logger.info(title)
    (dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
         = analytical_MPVS(province_cases[province], CI = CI, smoothing = smoothing) 
    IDN = Model.single_unit(name = province, population = priority_pops[province], I0 = T_pred[-1], RR0 = RR_pred[-1], upper_CI = T_CI_upper[-1], lower_CI = T_CI_lower[-1], mobility = 0, random_seed = 0)\
            .run(prediction_period)

    plt.Rt(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0.2, ymax=4.5)\
        .title(f"{title}")\
        .xlabel("\ndate")\
        .ylabel("$R_t$", rotation=0, labelpad=30)\
        .show()

    # plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, IDN[0].delta_T[:-1], IDN[0].lower_CI[1:], IDN[0].upper_CI[1:])\
    #     .title(f"\n{title}")\
    #     .xlabel("\ndate")\
    #     .ylabel("cases")\
    #     .show()