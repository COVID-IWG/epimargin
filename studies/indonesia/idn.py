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

logger = getLogger("IDN")

provinces =[
    'ACEH',
    'BALI',
    'BANTEN',
    'BENGKULU',
    'DAERAH ISTIMEWA YOGYAKARTA',
    'DKI JAKARTA',
    'GORONTALO',
    'JAMBI',
    'JAWA BARAT',
    'JAWA TENGAH',
    'JAWA TIMUR',
    'KALIMANTAN BARAT',
    'KALIMANTAN SELATAN',
    'KALIMANTAN TENGAH',
    'KALIMANTAN TIMUR',
    'KALIMANTAN UTARA',
    'KEPULAUAN BANGKA BELITUNG',
    'KEPULAUAN RIAU',
    'LAMPUNG',
    'MALUKU',
    'MALUKU UTARA',
    'NUSA TENGGARA BARAT',
    'NUSA TENGGARA TIMUR',
    'PAPUA',
    'PAPUA BARAT',
    'RIAU',
    'SULAWESI BARAT',
    'SULAWESI SELATAN',
    'SULAWESI TENGAH',
    'SULAWESI TENGGARA',
    'SULAWESI UTARA',
    'SUMATERA BARAT',
    'SUMATERA SELATAN',
    'SUMATERA UTARA'
]

# provinces = [
#     'SUMATERA BARAT',
#     'SUMATERA UTARA',
#     'RIAU',
#     'JAMBI',
#     'SUMATERA SELATAN',
#     'LAMPUNG',
#     'KEPULAUAN BANGKA BELITUNG',
#     'KEPULAUAN RIAU',
#     'DKI JAKARTA',
#     'JAWA BARAT',
#     'JAWA TENGAH',
#     'DAERAH ISTIMEWA YOGYAKARTA',
#     'JAWA TIMUR',
#     'BANTEN',
#     'BALI',
#     'NUSA TENGGARA BARAT',
#     'KALIMANTAN TENGAH',
#     'KALIMANTAN SELATAN',
#     'KALIMANTAN TIMUR',
#     'SULAWESI SELATAN',
#     'SULAWESI BARAT'
# ]

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
natl_cases = sum(province_cases.values())


logger.info("running national-level Rt estimate")
(dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
     = analytical_MPVS(natl_cases, CI = CI, smoothing = smoothing) 

plt.Rt(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0, ymax=4)\
    .title("\nIndonesia: Reproductive Number Estimate")\
    .xlabel("\ndate")\
    .ylabel("$R_t$", rotation=0, labelpad=30)\
    .annotate(f"\n{window}-day smoothing window, gamma-prior Bayesian estimation method")\
    .show()

logger.info("running case-forward prediction")
IDN = Model.single_unit("IDN", 267.7e6, I0 = T_pred[-1], RR0 = RR_pred[-1], mobility = 0, random_seed = 0)\
           .run(14)
plt.daily_cases(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI, IDN[0].delta_T[:-1], IDN[0].lower_CI[1:], IDN[0].upper_CI[1:])\
    .title("\nIndonesia: Net Daily Cases")\
    .xlabel("\ndate")\
    .ylabel("cases")\
    .annotate("\nBayesian training process on empirical data, with anomalies identified")\
    .show()


logger.info("province-level projections")
migration = np.zeros((len(provinces), len(provinces)))
estimates = []
max_len = 1 + max(map(len, provinces))
with tqdm(provinces) as progress:
    for (province, cases) in province_cases.items():
        progress.set_description(f"{province :<{max_len}}")
        (dates, RR_pred, RR_CI_upper, RR_CI_lower, *_) = analytical_MPVS(cases, CI = CI, smoothing = smoothing)
        apr_idx = np.argmax(dates >  "31 Mar, 2020")
        may_idx = np.argmax(dates >= "01 May, 2020")
        max_idx = np.argmax(RR_pred[apr_idx:may_idx])
        apr_max_idx = apr_idx + max_idx
        estimates.append((province, RR_pred[-1], RR_CI_lower[-1], RR_CI_upper[-1], max(0, linear_projection(dates, RR_pred, window)), RR_pred[apr_max_idx], RR_CI_lower[apr_max_idx], RR_CI_upper[apr_max_idx], dates[apr_max_idx], cases.iloc[-1][0]))
        progress.update()
estimates = pd.DataFrame(estimates)
estimates.columns = ["province", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj", "Rt_max", "Rt_CI_lower_at_max", "RR_CI_upper_at_max", "date_at_max_Rt", "total_cases"]
estimates.set_index("province", inplace=True)
estimates.to_csv(data/"IDN_only_apr_Rt_max_filtered.csv")
print(estimates)

# choropleths
logger.info("generating choropleths")
gdf = gpd.read_file("data/gadm36_IDN_shp/gadm36_IDN_1.shp").drop(["GID_0", "NAME_0", "GID_1", "VARNAME_1", "NL_NAME_1", "TYPE_1", "ENGTYPE_1", "CC_1", "HASC_1"], axis = 1)
gdf["NAME_1"] = gdf.NAME_1.str.upper().map(lambda s: replacements.get(s, s))
gdf = gdf.merge(estimates, left_on="NAME_1", right_on="province")


sm = mpl.cm.ScalarMappable(
    norm = mpl.colors.Normalize(vmin = 0.9, vmax = 1.4),
    cmap = "viridis"
)

MYS = gpd.read_file("data/gadm36_MYS_shp/gadm36_MYS_0.shp") 
TLS = gpd.read_file("data/gadm36_TLS_shp/gadm36_TLS_0.shp") 
PNG = gpd.read_file("data/gadm36_PNG_shp/gadm36_PNG_0.shp") 

choro = plt.choropleth.vertical(gdf, lambda _: "", mappable = sm).adjust(left = 0.01)
ax1, ax2, _ = choro.figure.axes 
for ax in (ax1, ax2):
    MYS.plot(color = "gray", ax = ax)
    PNG.plot(color = "gray", ax = ax)
    TLS.plot(color = "gray", ax = ax)
    ax.set_xlim(left = 94.65, right = 144.0)
    ax.set_ylim(bottom = -11.32)
plt.show()