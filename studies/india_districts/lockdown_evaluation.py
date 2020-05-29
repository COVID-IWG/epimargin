from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from adaptive.estimators import rollingOLS as run_regressions
from adaptive.model import Model, ModelUnit, gravity_matrix
from adaptive.plots import plot_simulation_range
from adaptive.policy import simulate_adaptive_control, simulate_lockdown
from adaptive.utils import cwd, days, weeks
from etl import download_data, district_migration_matrices, get_time_series, load_all_data


def get_model(districts, populations, timeseries, seed = 0):
    units = [ModelUnit(
        name       = district, 
        population = populations[i],
        I0  = timeseries[district].iloc[-1]['Hospitalized'] if not timeseries[district].empty and 'Hospitalized' in timeseries[district].iloc[-1] else 0,
        R0  = timeseries[district].iloc[-1]['Recovered']    if not timeseries[district].empty and 'Recovered'    in timeseries[district].iloc[-1] else 0,
        D0  = timeseries[district].iloc[-1]['Deceased']     if not timeseries[district].empty and 'Deceased'     in timeseries[district].iloc[-1] else 0,
    ) for (i, district) in enumerate(districts)]
    return Model(units, random_seed = seed)

def run_policies(migrations, district_names, populations, district_time_series, Rm, Rv, gamma, seed, initial_lockdown = 13*days, total_time = 190*days):    
    # run various policy scenarios
    lockdown = np.zeros(migrations.shape)

    # 1. release lockdown 31 May 
    release_31_may = get_model(district_names, populations, district_time_series, seed)
    simulate_lockdown(release_31_may, 
        lockdown_period = initial_lockdown + 4*weeks, 
        total_time      = total_time, 
        RR0_mandatory   = Rm,              RR0_voluntary = Rv, 
        lockdown        = lockdown.copy(), migrations    = migrations)

    # 3. adaptive release starting 31 may 
    adaptive = get_model(district_names, populations, district_time_series, seed)
    simulate_adaptive_control(adaptive, initial_lockdown, total_time, lockdown, migrations, Rm, {district: R * gamma for (district, R) in Rv.items()}, {district: R * gamma for (district, R) in Rm.items()}, evaluation_period=1*weeks)

    return (release_31_may, adaptive)
