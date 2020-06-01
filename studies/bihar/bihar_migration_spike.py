from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import etl
from adaptive.estimators import rollingOLS, lowess
from adaptive.model import MigrationSpikeModel, Model, ModelUnit
from adaptive.plots import gantt_chart, plot_simulation_range
from adaptive.policy import AUC, simulate_adaptive_control, simulate_lockdown
from adaptive.utils import cwd, days, fmt_params, weeks


def model(districts, populations, cases, migratory_influx, seed) -> Model:
    units = [
        ModelUnit(district, populations[i], 
        I0 = cases[district].sort_index().iloc[-1].Hospitalized if district in cases.keys() else 0, 
        R0 = cases[district].sort_index().iloc[-1].Recovered    if district in cases.keys() else 0, 
        D0 = cases[district].sort_index().iloc[-1].Deceased     if district in cases.keys() else 0)
        for (i, district) in enumerate(districts)
    ]
    if migratory_influx:
        return MigrationSpikeModel(units, introduction_time = 6, migratory_influx = migratory_influx, random_seed=seed)
    else: 
        return Model(units, random_seed=seed)

def run_policies(
        district_cases:   Dict[str, pd.DataFrame], # timeseries for each district 
        populations:      pd.Series,               # population for each district
        districts:        Sequence[str],           # list of district names 
        migrations:       np.matrix,               # O->D migration matrix, normalized
        migratory_influx: Dict[str, float],        # exogenous migration influx 
        gamma:            float,                   # 1/infectious period 
        Rmw:              Dict[str, float],        # mandatory regime R
        Rvw:              Dict[str, float],        # voluntary regime R
        total:            int   = 90*days,        # how long to run simulation
        eval_period:      int   = 2*weeks,         # adaptive evaluation perion
        beta_scaling:     float = 1.0,             # robustness scaling: how much to shift empirical beta by 
        seed:             int   = 0                # random seed for simulation
    ):
    lockdown = np.zeros(migrations.shape)

    # 31 may full release 
    model_A = model(districts, populations, district_cases, migratory_influx, seed)
    simulate_lockdown(model_A, 13*days, total, Rmw, Rvw, lockdown, migrations)

    # adaptive control
    model_B = model(districts, populations, district_cases, migratory_influx, seed)
    simulate_adaptive_control(model_B, 13*days, total, lockdown, migrations, Rmw,
        {district: beta_scaling * Rv * gamma for (district, Rv) in Rvw.items()},
        {district: beta_scaling * Rm * gamma for (district, Rm) in Rmw.items()},
        evaluation_period=eval_period
    )
    return model_A, model_B 

def project(p: pd.Series):
    t = (p.R - p.Intercept)/p.gradient
    return (max(0, p.R), max(0, p.Intercept + p.gradient*(t + 7)), max(0, p.Intercept + p.gradient*(t + 14)), np.sqrt(p.gradient_stderr))

def estimate(district, ts, default = 1.5, window = 2, use_last = False):
    try:
        regressions = rollingOLS(etl.log_delta_smoothed(ts), window = window, infectious_period = 1/gamma)[["R", "Intercept", "gradient", "gradient_stderr"]]
        if use_last:
            return next((x for x in regressions.R.iloc[-3:-1] if not np.isnan(x) and x > 0), default)
        return regressions
    except (ValueError, IndexError):
        return default

