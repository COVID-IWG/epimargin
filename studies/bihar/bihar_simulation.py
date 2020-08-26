from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import etl
from adaptive.estimators import analytical_MPVS
from adaptive.smoothing import convolution
from adaptive.model  import Model, ModelUnit
from adaptive.plots  import gantt_chart, plot_simulation_range
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.utils  import cwd, days, weeks, fmt_params


def model(districts, populations, cases, seed) -> Model:
    max_ts = cases["date_reported"].max()
    units = [
        ModelUnit(district, populations[i], 
        I0 = len(cases[(cases.geo_reported == district) & (cases.date_reported == max_ts)]), 
        R0 = 0, 
        D0 = 0)
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
    simulate_lockdown(model_B, lockdown_period + 11, total, Rmw, Rvw, lockdown_matrix, migrations)

    # lockdown + adaptive controls
    model_C = model(districts, populations, district_cases, seed)
    simulate_adaptive_control(model_C, lockdown_period, total, lockdown_matrix, migrations, Rmw,
        {district: beta_scaling * Rv * gamma for (district, Rv) in Rvw.items()},
        {district: beta_scaling * Rm * gamma for (district, Rm) in Rmw.items()},
        evaluation_period=eval_period
    )

    return model_A, model_B, model_C

def project(p: pd.Series):
    t = (p.R - p.Intercept)/p.gradient
    return (max(0, p.R), max(0, p.Intercept + p.gradient*(t + 7)), max(0, p.Intercept + p.gradient*(t + 14)), np.sqrt(p.gradient_stderr))

if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"

    total_time = 90 * days 
    release_date = pd.to_datetime("20 July, 2020")
    # lockdown_period = (release_date - pd.to_datetime("today")).days
    lockdown_period = 0
    
    gamma  = 0.2
    window = 5
    CI = 0.95

    state_cases = pd.read_csv(data/"Bihar_cases_data_Jul23.csv", parse_dates=["date_reported", "date_status_change"], dayfirst=True)
    state_cases = state_cases[state_cases.date_reported <= "2020-07-20"]
    state_ts = state_cases["date_reported"].value_counts().sort_index()
    district_ts = state_cases.groupby(["geo_reported", "date_reported"])["date_reported"].count().sort_index()
    R_mandatory = dict()
    for district in district_ts.index.get_level_values(0).unique():
        try: 
            (_, Rt, *_) = analytical_MPVS(district_ts.loc[district], CI = CI, smoothing = convolution(window = 5))
            Rm = np.mean(Rt)
        except ValueError as v:
            Rm = 1.5
        R_mandatory[district] = Rm
    districts, pops, migrations = etl.district_migration_matrix(data/"Migration Matrix - District.csv")
    districts = [etl.replacements.get(dn, dn) for dn in districts]
    
    R_voluntary = {district: 1.5*R for (district, R) in R_mandatory.items()}

    si, sf = 0, 100

    simulation_results = [ 
        run_policies(state_cases, pops, districts, migrations, gamma, R_mandatory, R_voluntary, lockdown_period = lockdown_period, total = total_time, seed = seed)
        for seed in tqdm(range(si, sf))
    ]

    plot_simulation_range(
        simulation_results, 
        ["20 July Release", "31 July Release", "Adaptive Control Starting 20 July"], 
        historical = state_ts)\
        .title("Bihar Policy Scenarios: Projected Infections over Time")\
        .xlabel("Date")\
        .ylabel("Number of Infections")\
        .annotate(f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {window}")\
        .show()
