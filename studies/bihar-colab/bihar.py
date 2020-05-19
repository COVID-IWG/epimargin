from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import etl
from adaptive.estimators import rollingOLS
from adaptive.model  import Model, ModelUnit
from adaptive.plots  import gantt_chart, plot_simulation_range
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.utils  import cwd, days, weeks, fmt_params


def model(districts, populations, cases, seed) -> Model:
    units = [
        ModelUnit(district, populations[i], 
        I0 = cases[district].iloc[-1].Hospitalized if district in cases.keys() else 0, 
        R0 = cases[district].iloc[-1].Recovered    if district in cases.keys() else 0, 
        D0 = cases[district].iloc[-1].Deceased     if district in cases.keys() else 0)
        for (i, district) in enumerate(districts)
    ]
    return Model(units, random_seed=seed)

def run_policies(
        district_cases: Dict[str, pd.DataFrame], # timeseries for each district 
        populations:    pd.Series,               # population for each district
        districts:      Sequence[str],           # list of district names 
        migrations:     np.matrix,               # O->D migration matrix, normalized
        gamma:          float,                   # 1/infectious period 
        Rmw:            Dict[str, float],        # mandatory regime R
        Rvw:            Dict[str, float],        # voluntary regime R
        total:          int   = 90*days,         # how long to run simulation
        eval_period:    int   = 2*weeks,         # adaptive evaluation perion
        beta_scaling:   float = 1.0,             # robustness scaling: how much to shift empirical beta by 
        seed:           int   = 0                # random seed for simulation
    ):
    lockdown = np.zeros(migrations.shape)

    # 9 day lockdown 
    model_A = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_A, 9*days, total, Rmw, Rvw, lockdown, migrations)

    # 9 day + 2 week lockdown 
    model_B = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_B, 9*days + 2*weeks, total, Rmw, Rvw, lockdown, migrations)

    # 9 day lockdown + adaptive controls
    model_C = model(districts, populations, district_cases, seed)
    simulate_adaptive_control(model_C, 9*days, total, lockdown, migrations, Rmw,
        {district: beta_scaling * Rv * gamma for (district, Rv) in Rvw.items()},
        {district: beta_scaling * Rm * gamma for (district, Rm) in Rmw.items()},
        evaluation_period=eval_period
    )

    return model_A, model_B, model_C

def estimate(district, ts, default = 1.5, window = 2, use_last = False):
    try:
        regressions = rollingOLS(etl.log_delta(ts), window = window, infectious_period = 1/gamma)[["R", "Intercept", "gradient", "gradient_stderr"]]
        if use_last:
            return next((x for x in regressions.R.iloc[-3:-1] if not np.isnan(x) and x > 0), default)
        return regressions
    except (ValueError, IndexError):
        return default

def gantt_seed(seed, note = ""):
    _, _, mc = run_policies(district_ts, pops, districts, migrations, gamma, R_mandatory, R_voluntary, seed = seed) 
    gantt_chart(mc.gantt)\
        .title(f"Bihar: Example Adaptive Lockdown Mobility Regime Scenario {note if note else str(seed)}")\
        .show()

def project(p: pd.Series):
    t = (p.R - p.Intercept)/p.gradient
    return (max(0, p.R), max(0, p.Intercept + p.gradient*(t + 7)), max(0, p.Intercept + p.gradient*(t + 14)), np.sqrt(p.gradient_stderr))

if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"
    
    gamma  = 0.2
    window = 2

    state_cases    = etl.load_cases(data/"Bihar_Case_data_May11.csv")
    district_cases = etl.split_cases_by_district(state_cases)
    district_ts    = {district: etl.get_time_series(cases) for (district, cases) in district_cases.items()}
    R_mandatory    = {district: estimate(district, ts, use_last = True) for (district, ts) in district_ts.items()}
    districts, pops, migrations = etl.district_migration_matrix(data/"Migration Matrix - District.csv")
    for district in districts:
        if district not in R_mandatory.keys():
            R_mandatory[district] = 1.5
    
    R_voluntary    = {district: 1.5*R for (district, R) in R_mandatory.items()}

    si, sf = 0, 1000

    simulation_results = [ 
        run_policies(district_ts, pops, districts, migrations, gamma, R_mandatory, R_voluntary, seed = seed)
        for seed in tqdm(range(si, sf))
    ]

    plot_simulation_range(simulation_results, ["17 May Release", "31 May Release", "Adaptive Control"], etl.get_time_series(state_cases)["Hospitalized"])\
        .title("Bihar Policy Scenarios: Projected Infections over Time")\
        .xlabel("Date")\
        .ylabel("Number of Infections")\
        .annotate(f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {window}")\
        .show()

    best_index  = 1
    worst_index = 665

    gantt_chart(simulation_results[best_index][-1].gantt, start_date="May 17, 2020")\
        .title("Bihar: Example Adaptive Lockdown Mobility Regime Scenario " + str(best_index))\
        .show()

    gantt_chart(simulation_results[worst_index][-1].gantt, start_date="May 17, 2020")\
        .title("Bihar: Example Adaptive Lockdown Mobility Regime Scenario " + str(worst_index))\
        .show()

    # projections
    estimates = {district: estimate(district, ts, default = -1) for (district, ts) in district_ts.items()}
    index = {k: v.last_valid_index() if v is not -1 else v for (k, v) in estimates.items()}
    projections = []
    for district, estimate in estimates.items():
        if estimate is -1:
            projections.append((district, None, None, None, None))
        else:
            idx = index[district]
            if idx is None or idx is -1:
                projections.append((district, None, None, None, None))
            else: 
                projections.append((district, *project(estimate.loc[idx])))
    projdf = pd.DataFrame(data = projections, columns = ["district", "current R", "1 week projection", "2 week projection", "stderr"])
