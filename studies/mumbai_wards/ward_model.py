from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import etl
from adaptive.estimators import rollingOLS
from adaptive.model import Model, ModelUnit
from adaptive.plots import plot_simulation_range
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.utils import cwd, days, weeks

gamma = 0.2
Rv_Rm = 1.4836370631808469

def model(wards, populations, cases, seed) -> Model:
    units = [
        ModelUnit(ward, populations.iloc[i], I0 = cases[ward].iloc[-1].cases) 
        for (i, ward) in enumerate(wards)
    ]
    return Model(units, random_seed=seed)

def get_R(ward_cases: Dict[str, pd.DataFrame], gamma: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    tsw = {ward: etl.log_delta(cases) for (ward, cases) in ward_cases.items()}
    grw = {ward: rollingOLS(ts, infectious_period = 1/gamma) for (ward, ts) in tsw.items()}
    
    Rmw = {ward: np.mean(growth_rates.R) for (ward, growth_rates) in grw.items()}
    Rvw = {ward: Rv_Rm*Rm for (ward, Rm) in Rmw.items()}

    return Rmw, Rvw

def run_policies(
        ward_cases:   Dict[str, pd.DataFrame], # timeseries for each ward 
        populations:  pd.Series,               # population for each ward
        wards:        Sequence[str],           # list of ward names 
        migrations:   np.matrix,               # O->D migration matrix, normalized
        gamma:        float,                   # 1/infectious period 
        Rmw:          Dict[str, float],        # mandatory regime R
        Rvw:          Dict[str, float],        # mandatory regime R
        total:        int   = 188*days,        # how long to run simulation
        eval_period:  int   = 2*weeks,         # adaptive evaluation perion
        beta_scaling: float = 1.0,             # robustness scaling: how much to shift empirical beta by 
        seed:         int   = 0                # random seed for simulation
    ):
    lockdown = np.zeros(migrations.shape)

    # 8 day lockdown 
    model_A = model(wards, populations, ward_cases, seed)
    simulate_lockdown(model_A, 8*days, total, Rmw, Rvw, lockdown, migrations)

    # 8 day + 4 week lockdown 
    model_B = model(wards, populations, ward_cases, seed)
    simulate_lockdown(model_B, 8*days + 4*weeks, total, Rmw, Rvw, lockdown, migrations)

    # 8 day lockdown + adaptive controls
    model_C = model(wards, populations, ward_cases, seed)
    simulate_adaptive_control(model_C, 8*days, total, lockdown, migrations, Rmw,
        {ward: beta_scaling * Rv * gamma for (ward, Rv) in Rvw.items()},
        {ward: beta_scaling * Rm * gamma for (ward, Rm) in Rmw.items()},
        evaluation_period = eval_period
    )

    return model_A, model_B, model_C


if __name__ == "__main__":
    root = cwd()
    data = root/"data"
    figs = root/"figs"

    all_cases           = etl.load_case_data(data/"mumbai_wards_30Apr.csv")
    population_data     = etl.load_population_data(data/"ward_data_Mumbai_empt_slums.csv")
    (wards, migrations) = etl.load_migration_data(data/"Ward_rly_matrix_Mumbai.csv")
    lockdown = np.zeros(migrations.shape)

    ward_cases = {ward: all_cases[all_cases.ward == ward] for ward in wards}
    Rmw, Rvw = get_R(ward_cases, gamma)

    si, sf = 0, 1000

    simulation_results = [
        run_policies(ward_cases, population_data, wards, migrations, gamma, Rmw, Rvw, total = 188*days, eval_period = 2*weeks, seed = i) 
    for i in range(si, sf)]

    plot_simulation_range(simulation_results, ["03 May Release", "31 May Release", "Adapative Lockdown"], all_cases[all_cases.ward == "ALL"].cases)\
        .title(f"Mumbai Policy Scenarios: Projected Infections over Time")\
        .xlabel("Date")\
        .ylabel("Number of Infections")\
        .annotate(f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {3}")\
        .show()