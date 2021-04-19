from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import epimargin.plots as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from epimargin.estimators import analytical_MPVS
from epimargin.models import SIR, NetworkedSIR
from epimargin.policy import simulate_adaptive_control, simulate_lockdown
from epimargin.smoothing import convolution, notched_smoothing
from epimargin.utils import days, setup, weeks
from scipy.spatial import distance_matrix
from tqdm import tqdm

from jakarta_granular import replacements

logger = getLogger("DKIJ.AC")

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


def model(districts, populations, cases, seed) -> NetworkedSIR:
    units = [
        SIR(district, populations[i], 
        dT0 = cases[i], 
        R0 = 0, 
        D0 = 0, 
        mobility = 0.000001)
        for (i, district) in enumerate(districts)
    ]
    return NetworkedSIR(units, random_seed=seed)

def run_policies(
        district_cases:  Dict[str, pd.DataFrame], # timeseries for each district 
        populations:     pd.Series,               # population for each district
        districts:       Sequence[str],           # list of district names 
        migrations:      np.matrix,               # O->D migration matrix, normalized
        gamma:           float,                   # 1/infectious period 
        Rmw:             Dict[str, float],        # mandatory regime R
        Rvw:             Dict[str, float],        # voluntary regime R
        lockdown_period: int,                     # how long to run lockdown 
        total:           int   = 90*days,         # how long to run simulation
        eval_period:     int   = 2*weeks,         # adaptive evaluation period
        beta_scaling:    float = 1.0,             # robustness scaling: how much to shift empirical beta by 
        seed:            int   = 0                # random seed for simulation
    ):
    lockdown_matrix = np.zeros(migrations.shape)

    # lockdown 1
    model_A = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_A, lockdown_period, total, Rmw, Rvw, lockdown_matrix, migrations)

    # lockdown 1
    model_B = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_B, lockdown_period + 2*weeks, total, Rmw, Rvw, lockdown_matrix, migrations)

    # lockdown + adaptive controls
    model_C = model(districts, populations, district_cases, seed)
    simulate_adaptive_control(model_C, lockdown_period + 2*weeks, total, lockdown_matrix, migrations, Rmw,
        {district: beta_scaling * Rv * gamma for (district, Rv) in Rvw.items()},
        {district: beta_scaling * Rm * gamma for (district, Rm) in Rmw.items()},
        evaluation_period=eval_period
    )

    return model_A, model_B, model_C

(data, figs) = setup(level = "INFO")

total_time = 45 * days 
lockdown_period = 7

gamma  = 0.2
window = 10
CI = 0.95

logger.info("district-level projections")
dkij = pd.read_stata(data/"coviddkijakarta_290920.dta")\
         .query("province == 'DKI JAKARTA'")\
         .drop(columns = dkij_drop_cols + ["province"])
dkij = dkij\
    .set_axis(dkij.columns.str.lower(), 1)\
    .assign(
        district    = dkij.district.str.title(),
        subdistrict = dkij.subdistrict.apply(lambda name: next((k for (k, v) in replacements.items() if name in v), name)), 
    )

district_cases = dkij.groupby(["district", "date_positiveresult"])["id"].count().sort_index()
districts = sorted(dkij.district.unique())
migration = np.zeros((len(districts), len(districts)))
R_mandatory = dict()
R_voluntary = dict() 
max_len = 1 + max(map(len, districts))
with tqdm(districts) as progress:
    for district in districts:
        progress.set_description(f"{district :<{max_len}}")
        (dates, RR_pred, *_) = analytical_MPVS(district_cases.loc[district], CI = CI, smoothing = notched_smoothing(window = window), totals=False)
        Rt = pd.DataFrame(data = {"Rt": RR_pred[1:]}, index = dates)
        R_mandatory[district] = np.mean(Rt[(Rt.index > "April 1, 2020") & (Rt.index < "June 1, 2020")])[0]
        R_voluntary[district] = np.mean(Rt[(Rt.index < "April 1, 2020")])[0]

pops = [
    2_430_410,
    910_381,
    2_164_070,
    2_817_994,
    1_729_444,
    23_011
]

gdf = gpd.read_file("data/gadm36_IDN_shp/gadm36_IDN_2.shp")\
         .query("NAME_1 == 'Jakarta Raya'")\
         .drop(columns=shp_drop_cols)
centroids = [list(pt.coords)[0] for pt in gdf.centroid]
P = distance_matrix(centroids, centroids)
P[P != 0] = P[P != 0] ** -1.0 
P *= np.array(pops)[:, None]
P /= P.sum(axis = 0)

si, sf = 0, 10

simulation_results = [ 
    run_policies([district_cases[district][-1] for district in districts], pops, districts, P, gamma, R_mandatory, R_voluntary, lockdown_period = lockdown_period, total = total_time, seed = seed)
    for seed in tqdm(range(si, sf))
]

historical = dkij.groupby("date_positiveresult")["id"].count().rename("cases")

plt.simulations(simulation_results, 
    ["04 October: Release From Lockdown", "11 October: Release From Lockdown", "11 October: Start Adaptive Control"], 
    historical = historical[historical.index >= "01 May, 2020"])\
    .title("\nJakarta Policy Scenarios: Projected Cases over Time")\
    .xlabel("date")\
    .ylabel("cases")\
    .annotate(f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {window}")\
    .show()
