import json
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from tqdm import tqdm

import adaptive.plots as plt
from adaptive.estimators import analytical_MPVS
from adaptive.model import Model, ModelUnit, gravity_matrix
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.smoothing import convolution, notched_smoothing
from adaptive.utils import days, setup, weeks

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


def model(districts, populations, cases, seed) -> Model:
    units = [
        ModelUnit(district, populations[i], 
        I0 = cases[i], 
        R0 = 0, 
        D0 = 0, 
        mobility = 0.000001)
        for (i, district) in enumerate(districts)
    ]
    return Model(units, random_seed=seed)

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
        eval_period:     int   = 2*weeks,         # adaptive evaluation perion
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

total_time = 90 * days 
lockdown_period = 0

gamma  = 0.2
window = 10
CI = 0.95

# data fields
date_scale  = 1000000.0
date        = "tanggal"
timeseries  = "list_perkembangan"
total_cases = "AKUMULASI_KASUS"
district    = "DKI JAKARTA"
districts   = [district]

filename = lambda province: "prov_detail_{}.json".format(province.replace(" ", "_"))

def load_province_timeseries(data_path: Path, province: str) -> pd.DataFrame:
    with (data_path/filename(province)).open() as fp:
        top_level = json.load(fp)
    df = pd.DataFrame([(_[date], _[total_cases]) for _ in top_level[timeseries]], columns=["date", "total_cases"])
    df["date"] = (date_scale * df["date"]).apply(pd.Timestamp)
    return df.set_index("date")

logger.info("district-level projections")

pops = [sum([2_430_410, 910_381, 2_164_070, 2_817_994, 1_729_444, 23_011])]
dkij = load_province_timeseries(data, district)
R_mandatory = dict()
R_voluntary = dict() 

(dates, RR_pred, *_) = analytical_MPVS(dkij, CI = CI, smoothing = notched_smoothing(window = window), totals=False)
Rt = pd.DataFrame(data = {"Rt": RR_pred[1:]}, index = dates)
R_mandatory[district] = np.mean(Rt[(Rt.index >= "Sept 21, 2020")])[0]
R_voluntary[district] = np.mean(Rt[(Rt.index <  "April 1, 2020")])[0]

si, sf = 0, 500

simulation_results = [ 
    run_policies([dkij.iloc[-1][0]], pops, districts, np.zeros((1, 1)), gamma, R_mandatory, R_voluntary, lockdown_period = lockdown_period, total = total_time, seed = seed)
    for seed in tqdm(range(si, sf))
]

def simulations(
    simulation_results: Sequence[Tuple[Model]], 
    labels: Sequence[str], 
    historical: Optional[pd.Series] = None, 
    historical_label: str = "Empirical Case Data", 
    curve: str = "delta_T", 
    smoothing: Optional[np.ndarray] = None) -> PlotDevice:

    aggregates = [tuple(model.aggregate(curve) for model in model_set) for model_set in simulation_results]

    policy_outcomes = list(zip(*aggregates))

    num_sims   = len(simulation_results)
    total_time = len(policy_outcomes[0][0])

    ranges = [{"max": [], "min": [], "mdn": [], "avg": []} for _ in range(len(policy_outcomes))]

    for (i, policy) in enumerate(policy_outcomes):
        for t in range(total_time):
            curve_sorted = sorted([curve[t] for curve in policy])
            ranges[i]["min"].append(curve_sorted[0])
            ranges[i]["max"].append(curve_sorted[-1])
            ranges[i]["mdn"].append(curve_sorted[num_sims//2])
            ranges[i]["avg"].append(np.mean(curve_sorted))

    legends = []
    legend_labels  = []
    if historical is not None:
        p, = plt.plot(historical.index, historical, 'k-', alpha = 0.8, zorder = 10)
        t  = [historical.index.max() + pd.Timedelta(days = n) for n in range(total_time)]
        legends.append(p)
        legend_labels.append(historical_label)
    else:
        t = list(range(total_time))

    if smoothing is not None:
        plt.plot([pd.Timestamp(t) for t in smoothing[:, 0]], smoothing[:, 1], 'k-', label = "LOESS smoothed data", linewidth = 1)
        
    for (rng, label, color) in zip(ranges, labels, SIM_PALETTE):
        p, = plt.plot(t, rng["avg"], color = color, linewidth = 2)
        f  = plt.fill_between(t, rng["min"], rng["max"], color = color, alpha = 0.2)
        legends.append((p, f))
        legend_labels.append(label)
    
    plt.gca().xaxis.set_major_formatter(DATE_FMT)
    plt.gca().xaxis.set_minor_formatter(DATE_FMT)
    plt.legend(legends, legend_labels, prop = dict(size = 20), handlelength = 1, framealpha = 1)

    plt.xlim(left = historical.index[0], right = t[-1])
    
    return plt.PlotDevice()

simulations(simulation_results, 
    ["28 September Release", "12 October Release", "28 September Adaptive Control"], 
    historical = dkij[dkij.index >= "01 Aug, 2020"] )\
    .title("\nJakarta Policy Scenarios: Projected Cases over Time")\
    .xlabel("date")\
    .ylabel("cases")\
    .show()
