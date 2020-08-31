import json
from logging import getLogger
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

from adaptive.estimators import analytical_MPVS
from adaptive.etl.commons import download_data
from adaptive.model import Model, ModelUnit
from adaptive.plots import plot_RR_est, plot_T_anomalies, label_font, note_font
from adaptive.policy import simulate_PID_controller
from adaptive.smoothing import notched_smoothing
from adaptive.utils import days, setup


def project(dates, R_values, smoothing, period = 7*days):
    julian_dates = [_.to_julian_date() for _ in dates[-smoothing//2:None]]
    return OLS(
        RR_pred[-smoothing//2:None], 
        add_constant(julian_dates)
    )\
    .fit()\
    .predict([1, julian_dates[-1] + period])[0]

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
    #download_data(data, filename(province), base_url = "https://data.covid19.go.id/public/api/")

province_cases = {province: load_province_timeseries(data, province) for province in provinces}
bgn = min(cases.index.min() for cases in province_cases.values())
end = max(cases.index.max() for cases in province_cases.values())
idx = pd.date_range(bgn, end)
province_cases = {province: cases.reindex(idx, method = "pad").fillna(0) for (province, cases) in province_cases.items()}
natl_cases = sum(province_cases.values())


logger.info("running national-level Rt estimate")
(
    dates,
    RR_pred, RR_CI_upper, RR_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = analytical_MPVS(natl_cases, CI = CI, smoothing = smoothing) 
plot_RR_est(dates, RR_pred, RR_CI_upper, RR_CI_lower, CI, ymin=0, ymax=4)\
    .title("\nIndonesia: Reproductive Number Estimate")\
    .xlabel("\ndate")\
    .ylabel("$R_t$", rotation=0, labelpad=30)\
    .annotate(f"\n{window}-day smoothing window, gamma-prior Bayesian estimation method")
plt.legend(prop = {'size': 18})
plt.xlim(left=bgn, right=end)
plt.ylim(0.75, 2.5)
plt.show()

logger.info("running case-forward prediction")
np.random.seed(0)
IDN = Model([ModelUnit("IDN", 267.7e6, I0 = T_pred[-1], RR0 = RR_pred[-1], mobility = 0)])
IDN.run(14, np.zeros((1,1)))
t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(len(IDN[0].delta_T))]

IDN[0].lower_CI[0] = T_CI_lower[-1]
IDN[0].upper_CI[0] = T_CI_upper[-1]
plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)\
    .title("\nIndonesia: Net Daily Cases")\
    .xlabel("\ndate")\
    .ylabel("cases")\
    .annotate("\nBayesian training process on empirical data, with anomalies identified")
# plt.scatter(t_pred, IDN[0].delta_T, color = "tomato", s = 4, label = "Predicted Net Cases")
# plt.fill_between(t_pred, IDN[0].lower_CI, IDN[0].upper_CI, color = "tomato", alpha = 0.3, label="95% CI (forecast)")
# plt.xlim(left=bgn, right=t_pred[-1])
plt.ylim(top=3000)
plt.xlim(left=bgn, right=end)
plt.legend(prop = {'size': 18})
plt.show()

IDN[0].lower_CI[0] = T_CI_lower[-1]
IDN[0].upper_CI[0] = T_CI_upper[-1]
plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)\
    .title("\nIndonesia: Net Daily Cases under Current $R_t$")\
    .xlabel("\ndate")\
    .ylabel("cases")\
    .annotate("\n14 day projection using stochastic compartmental model")
plt.scatter(t_pred, IDN[0].delta_T, color = "tomato", s = 4, label = "Predicted Net Cases")
plt.fill_between(t_pred, IDN[0].lower_CI, IDN[0].upper_CI, color = "tomato", alpha = 0.3, label="95% CI (forecast)")
plt.xlim(left=bgn, right=t_pred[-1])
plt.ylim(top=3000)
plt.legend(prop = {'size': 18})
plt.show()


logger.info("running PID controller")
np.random.seed(0)
IDN = Model([ModelUnit("IDN", 267.7e6, I0 = T_pred[-1], RR0 = RR_pred[-1], mobility = 0)])
print(len(IDN[0].RR))
simulate_PID_controller(IDN, 0, 14, 0.8)
print(len(IDN[0].RR))
t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(len(IDN[0].delta_T))]

IDN[0].lower_CI[0] = T_CI_lower[-1]
IDN[0].upper_CI[0] = T_CI_upper[-1]
plot_T_anomalies(dates, T_pred, T_CI_upper, T_CI_lower, new_cases_ts, anomaly_dates, anomalies, CI)\
    .title("\nIndonesia: Net Daily Cases under Targeted Control")\
    .xlabel("\ndate")\
    .ylabel("cases")\
    .annotate("\n14 day projection using stochastic compartmental model")
plt.scatter(t_pred, IDN[0].delta_T, color = "tomato", s = 4, label = "Predicted Net Cases under target $R_t$ = 0.8")
plt.fill_between(t_pred, IDN[0].lower_CI, IDN[0].upper_CI, color = "tomato", alpha = 0.3, label="95% CI (forecast)")
plt.legend(prop = {'size': 18})
plt.xlim(left=bgn, right=t_pred[-1])
plt.ylim(top=3000)
plt.show()


logger.info("province-level projections")
migration = np.zeros((len(provinces), len(provinces)))
estimates = []
max_len = 1 + max(map(len, provinces))
with tqdm(provinces) as progress:
    for (province, cases) in province_cases.items():
        progress.set_description(f"{province :<{max_len}}")
        (dates, RR_pred, RR_CI_upper, RR_CI_lower, *_) = analytical_MPVS(cases, CI = CI, smoothing = smoothing)
        estimates.append((province, RR_pred[-1], RR_CI_lower[-1], RR_CI_upper[-1], project(dates, RR_pred, window)))
estimates = pd.DataFrame(estimates)
estimates.columns = ["province", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
estimates.set_index("province", inplace=True)
estimates.to_csv(data/"IDN_Rt_projections.csv")
print(estimates)

# choropleths
logger.info("generating choropleths")
gdf = gpd.read_file("data/gadm36_IDN_shp/gadm36_IDN_1.shp").drop(["GID_0", "NAME_0", "GID_1", "VARNAME_1", "NL_NAME_1", "TYPE_1", "ENGTYPE_1", "CC_1", "HASC_1"], axis = 1)
gdf["NAME_1"] = gdf.NAME_1.str.upper().map(lambda s: replacements.get(s, s))
gdf = gdf.merge(estimates, left_on="NAME_1", right_on="province")

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.grid(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("Current $R_t$ by Province", loc="left", fontdict=label_font) 
ax2.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("1-Week Projected $R_t$ by Province", loc="left", fontdict=label_font) 
gdf.plot(column = "Rt",      cmap="RdYlGn_r", ax = ax1, edgecolors="black", linewidth=0.5)
gdf.plot(column = "Rt_proj", cmap="RdYlGn_r", ax = ax2, edgecolors="black", linewidth=0.5)
cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.5])
cb = fig.colorbar(
    mappable = mlp.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin = 0.8, vmax = 2.0), cmap="RdYlGn_r"), 
    orientation = "vertical",  
    cax = cbar_ax
)  
cbar_ax.set_title("$R_t$", fontdict = note_font)
plt.show()

# divider = make_axes_locatable(ax)
gdf.plot(column = "Rt", cmap="RdYlGn_r", legend=True,  ax = ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
