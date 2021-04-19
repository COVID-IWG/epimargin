import json
from logging import getLogger
from pathlib import Path

import epimargin.plots as plt
import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from epimargin.estimators import analytical_MPVS, linear_projection
from epimargin.etl.commons import download_data
from epimargin.models import SIR
from epimargin.smoothing import notched_smoothing
from epimargin.utils import days, setup, weeks
from tqdm import tqdm

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
# for province in provinces:
#     logger.info("downloading data for %s", province)
#     download_data(data, filename(province), base_url = "https://data.covid19.go.id/public/api/")

province_cases = {province: load_province_timeseries(data, province) for province in provinces}
bgn = min(cases.index.min() for cases in province_cases.values())
end = max(cases.index.max() for cases in province_cases.values())
idx = pd.date_range(bgn, end)
province_cases = {province: cases.reindex(idx, method = "pad").fillna(0) for (province, cases) in province_cases.items()}
natl_cases = sum(province_cases.values())


logger.info("running national-level Rt estimate")
(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
     = analytical_MPVS(natl_cases, CI = CI, smoothing = smoothing) 

plt.Rt(dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, CI, ymin=0, ymax=4)\
    .title("\nIndonesia: Reproductive Number Estimate")\
    .xlabel("\ndate")\
    .ylabel("$R_t$", rotation=0, labelpad=30)\
    .annotate(f"\n{window}-day smoothing window, gamma-prior Bayesian estimation method")\
    .show()

logger.info("running case-forward prediction")
IDN = SIR("IDN", 267.7e6, dT0 = T_pred[-1], Rt0 = Rt_pred[-1], mobility = 0, random_seed = 0).run(14)


logger.info("province-level projections")
migration = np.zeros((len(provinces), len(provinces)))
estimates = []
max_len = 1 + max(map(len, provinces))
with tqdm(provinces) as progress:
    for (province, cases) in province_cases.items():
        progress.set_description(f"{province :<{max_len}}")
        (dates, Rt_pred, Rt_CI_upper, Rt_CI_lower, *_) = analytical_MPVS(cases, CI = CI, smoothing = smoothing)
        apr_idx = np.argmax(dates >  "31 Mar, 2020")
        may_idx = np.argmax(dates >= "01 May, 2020")
        max_idx = np.argmax(Rt_pred[apr_idx:may_idx])
        apr_max_idx = apr_idx + max_idx
        estimates.append((province, Rt_pred[-1], Rt_CI_lower[-1], Rt_CI_upper[-1], max(0, linear_projection(dates, Rt_pred, window, period = 2*weeks)), Rt_pred[apr_max_idx], Rt_CI_lower[apr_max_idx], Rt_CI_upper[apr_max_idx], dates[apr_max_idx], cases.iloc[-1][0]))
        progress.update()
estimates = pd.DataFrame(estimates)
estimates.columns = ["province", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj", "Rt_max", "Rt_CI_lower_at_max", "Rt_CI_upper_at_max", "date_at_max_Rt", "total_cases"]
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


max_cmap = plt.get_cmap(vmin = 1, vmax = 4)
fig, ax = plt.subplots()
gdf.plot(
    color = [max_cmap.to_rgba(_) for _ in gdf.Rt_max],
    ax = ax, legend = True
)
plt.PlotDevice(fig)\
    .adjust(left = 0.1, bottom = 0.11, right = 0.9, top = 0.9)\
    .title("\n\nIndonesia: maximum $R_t$ by province for April, 2020", x = 0.1)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
MYS.plot(color = "gray", ax = ax, alpha = 0.5)
PNG.plot(color = "gray", ax = ax, alpha = 0.5)
TLS.plot(color = "gray", ax = ax, alpha = 0.5)
ax.set_xlim(left = 94.65, right = 141.379)
ax.set_ylim(bottom = -11.32, top = 5.93)
for (_, row) in gdf.iterrows():
    Rtm = round(row["Rt_max"], 2)
    Rtlabel = str(Rtm) if Rtm < 4 else ">4"
    ax.annotate(s=f"{Rtlabel}", xy=list(row["pt"].coords)[0], ha = "center", fontfamily = plt.note_font["family"], color="white")\
      .set_path_effects([plt.Stroke(linewidth = 3, foreground = "black"), plt.Normal()])
cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
cb = fig.colorbar(mappable = max_cmap, orientation = "vertical", cax = cbar_ax)
cbar_ax.set_title("$R_t$", fontdict = plt.note_font)
plt.show()
