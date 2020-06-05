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

globla gamma

def model(districts, populations, cases, seed) -> Model:
    max_ts = max([ts.index.max() for ts in cases.values()]).isoformat()
    units = [
        ModelUnit(district, populations[i], 
        I0 = cases[district].loc[max_ts].Hospitalized[0] if district in cases.keys() and max_ts in cases[district].index else 0, 
        R0 = cases[district].loc[max_ts].Recovered[0]    if district in cases.keys() and max_ts in cases[district].index else 0, 
        D0 = cases[district].loc[max_ts].Deceased[0]     if district in cases.keys() and max_ts in cases[district].index else 0)
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

    # lockdown 1
    model_A = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_A, 5*days, total, Rmw, Rvw, lockdown, migrations)

    # lockdown 2
    model_B = model(districts, populations, district_cases, seed)
    simulate_lockdown(model_B, 35*days, total, Rmw, Rvw, lockdown, migrations)

    # 9 day lockdown + adaptive controls
    model_C = model(districts, populations, district_cases, seed)
    simulate_adaptive_control(model_C, 5*days, total, lockdown, migrations, Rmw,
        {district: beta_scaling * Rv * gamma for (district, Rv) in Rvw.items()},
        {district: beta_scaling * Rm * gamma for (district, Rm) in Rmw.items()},
        evaluation_period=eval_period
    )

    return model_A, model_B, model_C

def estimate(district, ts, default = 1.5, window = 5, use_last = False):
    try:
        regressions = rollingOLS(etl.log_delta_smoothed(ts), window = window, infectious_period = 1/gamma)[["R", "Intercept", "gradient", "gradient_stderr"]]
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

