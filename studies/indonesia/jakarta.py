from logging import getLogger
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from adaptive.estimators import analytical_MPVS, linear_projection
from adaptive.etl.commons import download_data
from adaptive.model import Model, ModelUnit
from adaptive.plots import label_font, note_font, plot_RR_est, plot_T_anomalies
from adaptive.policy import simulate_PID_controller
from adaptive.smoothing import notched_smoothing
from adaptive.utils import days, setup

logger = getLogger("JKRT")

# model/sim details
gamma     = 0.2
window    = 7
CI        = 0.95
smoothing = notched_smoothing(window = window)

dkij_drop_cols = [
    'age', 'sex', 'fever', 'temp', 'cough', 'flu', 'sore_throat', 'shortness_breath', 'shivering', 'headache', 'malaise', 'muscle_pain',
    'nausea_vomiting', 'abdominal_pain', 'diarrhoea', 'date_recovered',
    'date_died', 'heart_disease', 'diabetes', 'pneumonia', 'hypertension', 'malignant',
    'immunology_disorder', 'chronic_kidney', 'chronic_liver', 'copd',
    'obesity', 'pregnant', 'tracing', 'otg', 'icu', 'intubation', 'ecmo',
    'criteria_cases', 'age_group', 'age_group2', 'date_discharge',
    'patient_status', 'death'
]

shp_drop_cols = ['GID_0', 'NAME_0', 'GID_1', 'NAME_1', 'NL_NAME_1', 'GID_2', 'VARNAME_2', 'NL_NAME_2', 'TYPE_2', 'ENGTYPE_2', 'CC_2', 'HASC_2']

(data, figs) = setup(level = "INFO")
dkij = pd.read_stata(data/"dkijakarta_180820.dta")\
         .query("province == 'DKI JAKARTA'")\
         .drop(columns=dkij_drop_cols + ["province"])
dkij["district"] = dkij.district.str.title()

gdf = gpd.read_file("data/gadm36_IDN_shp/gadm36_IDN_2.shp")\
         .query("NAME_1 == 'Jakarta Raya'")\
         .drop(columns=shp_drop_cols)
bbox = shapely.geometry.box(minx = 106.65, maxx = 107.00, miny = -6.40, maxy=-6.05)
gdf = gdf[gdf.intersects(bbox)]

jakarta_districts = dkij.district.str.title().unique()

logger.info("running province-level Rt estimate")
jakarta_cases = dkij.groupby("date_positiveresult")["id"].count().rename("cases")
(
    dates,
    RR_pred, RR_CI_upper, RR_CI_lower,
    T_pred, T_CI_upper, T_CI_lower,
    total_cases, new_cases_ts,
    anomalies, anomaly_dates
) = analytical_MPVS(jakarta_cases, CI = CI, smoothing = smoothing, totals=False) 
plot_RR_est(dates, RR_pred[1:], RR_CI_upper[1:], RR_CI_lower[1:], CI, ymin=0, ymax=4)\
    .title("\nDKI Jakarta: Reproductive Number Estimate")\
    .xlabel("\ndate")\
    .ylabel("$R_t$", rotation=0, labelpad=30)\
    .annotate(f"\n{window}-day smoothing window, gamma-prior Bayesian estimation method")
plt.legend(prop = {'size': 18})
plt.xlim(left=dates[0], right=dates[-1])
plt.ylim(0.75, 2.75)
plt.show()

logger.info("running case-forward prediction")
np.random.seed(0)
IDN = Model([ModelUnit("IDN", 267.7e6, I0 = T_pred[-1], RR0 = RR_pred[-1], mobility = 0)])
IDN.run(14, np.zeros((1,1)))
t_pred = [dates[-1] + pd.Timedelta(days = i) for i in range(len(IDN[0].delta_T))]

IDN[0].lower_CI[0] = T_CI_lower[-1]
IDN[0].upper_CI[0] = T_CI_upper[-1]
plot_T_anomalies(dates, T_pred[1:], T_CI_upper[1:], T_CI_lower[1:], new_cases_ts[1:], anomaly_dates, anomalies, CI)\
    .title("\nDKI Jakarta: Daily Cases")\
    .xlabel("\ndate")\
    .ylabel("cases")\
    .annotate("\nBayesian training process on empirical data, with anomalies identified")
plt.xlim(left=dates[0], right=dates[-1])
plt.legend(prop = {'size': 18})
plt.show()

logger.info("district-level projections")
district_cases = dkij.groupby(["district", "date_positiveresult"])["id"].count().sort_index()
districts = dkij.district.unique()
migration = np.zeros((len(districts), len(districts)))
estimates = []
max_len = 1 + max(map(len, districts))
with tqdm(districts) as progress:
    for district in districts:
        progress.set_description(f"{district :<{max_len}}")
        (dates, RR_pred, RR_CI_upper, RR_CI_lower, *_) = analytical_MPVS(district_cases.loc[district], CI = CI, smoothing = smoothing, totals=False)
        estimates.append((district, RR_pred[-1], RR_CI_lower[-1], RR_CI_upper[-1], linear_projection(dates, RR_pred, window)))
estimates = pd.DataFrame(estimates)
estimates.columns = ["district", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
estimates.set_index("district", inplace=True)
estimates.to_csv(data/"Jakarta_Rt_projections.csv")
print(estimates)

logger.info("generating choropleths")
gdf = gdf.merge(estimates, left_on = "NAME_2", right_on = "district")
# gdf["pt"] = gdf['geometry'].apply(lambda _: _.representative_point().coords[0])


sm = mpl.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin = 0.6, vmax = 1.3), cmap="RdYlGn_r")
gdf["pt"] = gdf['geometry'].centroid

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.grid(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("Current $R_t$ by District", loc="left", fontdict=label_font) 
ax2.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("1-Week Projected $R_t$ by District", loc="left", fontdict=label_font) 
gdf.plot(column = "Rt",      color=[sm.to_rgba(_) for _ in gdf.Rt],      ax = ax1, edgecolors="black", linewidth=0.5)
gdf.plot(column = "Rt_proj", color=[sm.to_rgba(_) for _ in gdf.Rt_proj], ax = ax2, edgecolors="black", linewidth=0.5)
for (_, row) in gdf.iterrows():
    initials = "".join(_[0] for _ in row["NAME_2"].split())
    Rt_round = round(row["Rt"], 2)
    Rt_proj_round = round(row["Rt_proj"], 2)
    ax1.annotate(s=f"{initials}: {Rt_round}        ",      xy=list(row["pt"].coords)[0], ha = "center", fontfamily=note_font["family"], color="white")
    ax2.annotate(s=f"{initials}: {Rt_proj_round}        ", xy=list(row["pt"].coords)[0], ha = "center", fontfamily=note_font["family"], color="white")
cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
cb = fig.colorbar(mappable = sm, orientation = "vertical", cax = cbar_ax)
cbar_ax.set_title("$R_t$", fontdict = note_font)
plt.show()