from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import epimargin.plots as plt
import numpy as np
import pandas as pd
from epimargin.estimators import analytical_MPVS
from epimargin.models import SIR, NetworkedSIR
from epimargin.policy import simulate_adaptive_control, simulate_lockdown
from epimargin.smoothing import convolution, notched_smoothing
from epimargin.utils import cwd, days, fmt_params, weeks
from tqdm import tqdm

import etl


def model(districts, populations, cases, seed) -> NetworkedSIR:
    max_ts = cases["date_reported"].max()
    units = [
        SIR(district, populations[i], 
        dT0 = len(cases[(cases.geo_reported == district) & (cases.date_reported == max_ts)]), 
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
    simulate_lockdown(model_B, lockdown_period + 6, total, Rmw, Rvw, lockdown_matrix, migrations)

    # lockdown + adaptive controls
    model_C = model(districts, populations, district_cases, seed)
    simulate_adaptive_control(model_C, lockdown_period, total, lockdown_matrix, migrations, Rmw,
        {district: beta_scaling * Rv * gamma for (district, Rv) in Rvw.items()},
        {district: beta_scaling * Rm * gamma for (district, Rm) in Rmw.items()},
        evaluation_period=eval_period
    )

    return model_A, model_B, model_C

if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"

    total_time = 90 * days 
    lockdown_period = 7
    
    gamma  = 0.2
    window = 10
    CI = 0.95

    state_cases = pd.read_csv(data/"Bihar_cases_data_Oct03.csv", parse_dates=["date_reported", "date_status_change"], dayfirst=True)
    state_cases["geo_reported"] = state_cases.geo_reported.str.strip()
    state_cases = state_cases[state_cases.date_reported <= "2020-09-30"]
    state_ts = state_cases["date_reported"].value_counts().sort_index()
    district_ts = state_cases.groupby(["geo_reported", "date_reported"])["date_reported"].count().sort_index()
    districts, pops, migrations = etl.district_migration_matrix(data/"Migration Matrix - District.csv")
    districts = sorted([etl.replacements.get(dn, dn) for dn in districts])
    
    R_mandatory = dict()
    for district in districts:#district_ts.index.get_level_values(0).unique():
        try: 
            (_, Rt, *_) = analytical_MPVS(district_ts.loc[district], CI = CI, smoothing = notched_smoothing(window = 10), totals = False)
            Rm = np.mean(Rt)
        except ValueError as v:
            Rm = 1.5
        R_mandatory[district] = Rm
    
    R_voluntary = {district: 1.2*R for (district, R) in R_mandatory.items()}

    si, sf = 0, 10

    simulation_results = [ 
        run_policies(state_cases, pops, districts, migrations, gamma, R_mandatory, R_voluntary, lockdown_period = lockdown_period, total = total_time, seed = seed)
        for seed in tqdm(range(si, sf))
    ]

    plt.simulations(
        simulation_results, 
        ["10 Oct: full release", "16 Oct: full release", "10 Oct: adaptive control begins"], 
        historical = state_ts)\
        .title("\nBihar Policy Scenarios: Projected Cases over Time")\
        .xlabel("date")\
        .ylabel("cases")\
        .annotate(f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {window}")\
        .show()
