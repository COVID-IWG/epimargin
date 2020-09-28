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
plt.choropleth(gdf, lambda row: row["NAME_2"]+"\n")\
   .adjust(left = 0.06)\
   .title("\nDKI Jakarta: $R_t$ by District")\
   .show()