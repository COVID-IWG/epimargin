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

import adaptive.plots as plt
from adaptive.estimators import analytical_MPVS, linear_projection
from adaptive.etl.commons import download_data
from adaptive.model import Model, ModelUnit
from adaptive.policy import simulate_PID_controller
from adaptive.smoothing import notched_smoothing
from adaptive.utils import days, setup

logger = getLogger("DKIJ")

# model/sim details
gamma     = 0.2
window    = 7
CI        = 0.95
smoothing = notched_smoothing(window = window)

replacements = { 
    "CAKUNG":           ['CAKUNG'],
    "CEMPAKA PUTIH":    ['CEMPA PUTIH', 'CEMPAKA PUTIH'],
    "CENGKARENG":       ['CENGAKRENG', 'CENGKARENG'],
    "CILANDAK":         ['CIILANDAK', 'CILANDAK'],
    "CILINCING":        ['CILINCIN','CILINCING', 'CILNCING', 'CLILINCING'],
    "CIPAYUNG":         ['CIPAYUNG'],
    "CIRACAS":          ['CIRACAS'],
    "DUREN SAWIT":      ['DUREN SAWIT'],
    "GAMBIR":           ['GAMBIR'],
    "GROGOLPETAMBURAN": ['GROGOL','GROGOL PEATAMBURAN', 'GROGOL PETAMBURAN', 'GROGOL PETEMBURAN', 'PETAMBURAN'],
    "JAGAKARSA":        ['JAGAKARSA'],
    "JATINEGARA":       ['JATIENGARA', 'JATINEGARA', 'JATINEGARAA'],
    "JOHAR BARU":       ['JOHAR BARU', 'JOHOR BARU'],
    "KALIDERES":        ['KALI DERES', 'KALIDERES'],
    "KEBAYORAN BARU":   ['KEBAYORAN BARU'],
    "KEBAYORAN LAMA":   ['KEBAYORAN LAMA', 'KEBAYORAN LAMA SELATAN', 'KEBAYORAN LIMA', 'KEBYORAN LAMA'],
    "KEBONJERUK":       ['KEBONJERUK', 'KEBON JERUK', 'KEBON JEURK'],
    "KELAPA GADING":    ['KELAPA GADING', 'KRELAPA GADING'],
    "KEMAYORAN":        ['KEMAYORAN'],
    "KEMBANGAN":        ['KEMBAGAN', 'KEMBANGAN'],
    "KOJA":             ['KOJA'],
    "KRAMATJATI":       ['KRAMAT JATI', 'KRAMATJATI'],
    "MAKASAR":          ['MAKASAR', 'MAKASSAR'],
    "MAMPANG PRAPATAN": ['MAMPANG', 'MAMPANG PERAPATAN', 'MAMPANG PRAPATAN'],
    "MATRAMAN":         ['MATRAMAN'],
    "MENTENG":          ["MENTENG"],
    "PADEMANGAN":       ['PADEMANGAN'],
    "PALMERAH":         ['PALMERAH'],
    "PANCORAN":         ['PANCORAN'],
    "PASAR MINGGU":     ['PASAR MINGGU', 'PAASAR MINGGU'],
    "PASARREBO":        ['PASARREBO', 'PASAR REBO'],
    "PENJARINGAN":      ['PENJAGALAN', 'PENJARINGAN', 'PENJARINGAN UTARA'],
    "PESANGGRAHAN":     ['PESANGGRAHAN'],
    "PULOGADUNG":       ['PULO GADUNG', 'PULOGADUNG'],
    "SAWAH BESAR":      ['SAWAH BESAR'],
    "SENEN":            ['SENEN'],
    "SETIA BUDI":       ['SETIA BUDI', 'SETIA BUSI', 'SEIA BUDI'],
    "SETIABUDI":        ['SETIABUDI'],
    "TAMANSARI":        ['TAMAN SARI', 'TAMANSARI'],
    "TAMBORA":          ['TAMBORA'],
    "TANAHABANG":       ['TANAHABANG', 'TANAH ABANG'],
    "TANJUNG PRIOK":    ['TAMAN SARI', 'TAMANSARI'],
    "TEBET":            ['TEBET'],
}


dkij_drop_cols = [
    'age', 'sex', 'fever', 'temp', 'cough', 'flu', 'sore_throat', 'shortness_breath', 'shivering', 'headache', 'malaise', 'muscle_pain',
    'nausea_vomiting', 'abdominal_pain', 'diarrhoea', 'date_recovered',
    'date_died', 'heart_disease', 'diabetes', 'pneumonia', 'hypertension', 'malignant',
    'immunology_disorder', 'chronic_kidney', 'chronic_liver', 'copd',
    'obesity', 'pregnant', 'tracing', 'otg', 'icu', 'intubation', 'ecmo',
    'criteria_cases', 'age_group', 'age_group2', 'date_discharge',
    'patient_status', 'death'
]

shp_drop_cols = ['GID_0', 'NAME_0', 'GID_1', 'NAME_1', 'NL_NAME_1', 'GID_2', 'VARNAME_3', 'NAME_2', 'NL_NAME_2', 'TYPE_3', 'ENGTYPE_3', 'CC_3', 'HASC_3', 'NL_NAME_3', "GID_3"]

(data, figs) = setup(level = "INFO")
dkij = pd.read_stata(data/"dkijakarta_180820.dta")\
         .query("province == 'DKI JAKARTA'")\
         .drop(columns=dkij_drop_cols + ["province"])
dkij["district"]    = dkij.district.str.title()
dkij["subdistrict"] = dkij.subdistrict.apply(lambda name: next((k for (k, v) in replacements.items() if name in v), name))

gdf = gpd.read_file("data/gadm36_IDN_shp/gadm36_IDN_3.shp")\
         .query("NAME_1.str.startswith('Jakarta')")\
         .drop(columns=shp_drop_cols)
gdf.NAME_3 = gdf.NAME_3.str.upper()
bbox = shapely.geometry.box(minx = 106.65, maxx = 107.00, miny = -6.40, maxy=-6.05)
gdf = gdf[gdf.intersects(bbox)]

jakarta_districts = dkij.district.str.title().unique()
jakarta_cases = dkij.groupby("date_positiveresult")["id"].count().rename("cases")

logger.info("running province-level Rt estimate")
(dates, RR_pred, RR_CI_upper, RR_CI_lower, T_pred, T_CI_upper, T_CI_lower, total_cases, new_cases_ts, anomalies, anomaly_dates)\
    = analytical_MPVS(jakarta_cases, CI = CI, smoothing = smoothing, totals=False) 

# plt.Rt(dates, RR_pred[1:], RR_CI_upper[1:], RR_CI_lower[1:], CI)\
#     .title("\nDKI Jakarta: Reproductive Number Estimate")\
#     .xlabel("\ndate")\
#     .ylabel("$R_t$\n", rotation=0, labelpad=30)\
#     .annotate(f"\n{window}-day smoothing window, gamma-prior Bayesian estimation method")\
#     .show()


# logger.info("running case-forward prediction")
# prediction_period = 14*days
# IDN = Model.single_unit(name = "IDN", population = 267.7e6, I0 = T_pred[-1], RR0 = RR_pred[-1], upper_CI = T_CI_upper[-1], lower_CI = T_CI_lower[-1], mobility = 0, random_seed = 0)\
#            .run(prediction_period)
 
# plt.daily_cases(dates, T_pred[1:], T_CI_upper[1:], T_CI_lower[1:], new_cases_ts[1:], anomaly_dates, anomalies, CI, IDN[0].delta_T[:-1], IDN[0].lower_CI[1:], IDN[0].upper_CI[1:])\
#     .title("\nDKI Jakarta: Daily Cases")\
#     .xlabel("\ndate")\
#     .ylabel("cases\n")\
#     .annotate("\nBayesian training process on empirical data, with anomalies identified")\
#     .show()

logger.info("subdistrict-level projections")
subdistrict_cases = dkij.groupby(["subdistrict", "date_positiveresult"])["id"].count().sort_index()
subdistricts = dkij.subdistrict.unique()
migration = np.zeros((len(subdistricts), len(subdistricts)))
estimates = []
max_len = 1 + max(map(len, subdistricts))
with tqdm(subdistricts) as progress:
    for subdistrict in subdistricts:
        progress.set_description(f"{subdistrict :<{max_len}}")
        try:
            (dates, RR_pred, RR_CI_upper, RR_CI_lower, *_) = analytical_MPVS(subdistrict_cases.loc[subdistrict], CI = CI, smoothing = smoothing, totals=False)
            estimates.append((subdistrict, RR_pred[-1], RR_CI_lower[-1], RR_CI_upper[-1], linear_projection(dates, RR_pred, window)))
        except Exception:
            estimates.append((subdistrict, np.nan, np.nan, np.nan, np.nan))
estimates = pd.DataFrame(estimates)
estimates.columns = ["subdistrict", "Rt", "Rt_CI_lower", "Rt_CI_upper", "Rt_proj"]
estimates.set_index("subdistrict", inplace=True)
estimates.to_csv(data/"Jakarta_Rt_projections.csv")
print(estimates)

logger.info("generating choropleths")

sm = mpl.cm.ScalarMappable(
    norm = mpl.colors.Normalize(vmin = 0.9, vmax = 1.4),
    cmap = "viridis"
)

gdf = gdf.merge(estimates, left_on = "NAME_3", right_on = "subdistrict")
plt.choropleth(gdf, label_fn= lambda _: "", mappable = sm)\
   .adjust(left = 0.06)\
   .show()

# crop to right
plt.choropleth(gdf, label_fn = None, Rt_col= "Rt_proj", Rt_proj_col= "Rt", titles = ["Projected $R_t$ (1 Week)", "Current $R_t$"], mappable = sm)\
   .adjust(left = 0.06)\
   .show()
