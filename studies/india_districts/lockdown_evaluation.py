from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from adaptive.model import Model, ModelUnit
from adaptive.utils import *
# from adaptive.estimators import rollingOLS
from etl import (get_time_series, load_data, load_district_migration_matrices,
                 run_regressions, v2)


def units():
    pass 

def model(units, seed):
    pass 

# lockdown policy of different lengths
def lockdown(model: Model, lockdown_length: int, total_time: int, RR0_mandatory: Dict[str, float], RR0_voluntary: Dict[str, float], lockeddown: np.matrix, migrations: np.matrix) -> Model:
    return model.set_parameters(RR0 = RR0_mandatory)\
        .run(lockdown_length,  migrations = lockeddown)\
        .set_parameters(RR0 = RR0_voluntary)\
        .run(total_time - lockdown_length, migrations = migrations)

# policy C: adaptive release
def adaptive_control(seed: int):
    model = Model(units, random_seed = seed)

if __name__ == "__main__":
    # set up folders
    root = cwd()
    import sys
    print(f"sys0: {sys.argv[0]}")
    print(f"root: {root}")
    data = root/"data"
    figs = root/"figs"

    # simulation parameters 
    seed   = 11235813
    states = ['Andhra Pradesh', 'Uttar Pradesh', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'West Bengal', 'Kerala', 'Gujarat'] 
    
    # model details 
    gamma      = 0.2
    prevalence = 1

    # run rolling regressions on historical case data 
    dfn = load_data(data/"india_case_data_23_4_resave.csv", reduced = True, schema = v2).dropna(subset = ["detected district"]) # can't do anything about this :( 
    tsn = get_time_series(dfn)
    grn = run_regressions(tsn, window = 7, infectious_period = 5)

    dfs = {state: dfn[dfn["detected state"] == state] for state in states}
    tss = {state: get_time_series(state_data) for (state, state_data) in dfs.items()}
    for (_, ts) in tss.items():
        ts['Hospitalized'] *= prevalence
    grs = {state: run_regressions(state_data, window = 3, infectious_period = 5) for (state, state_data) in tss.items() if len(state_data) > 3}
    
    grn["2020-03-24":"2020-03-31"]

    migration_matrices = load_district_migration_matrices(data/"Migration Matrix - District.csv", states = states)

    # release_03_may   = lockdown(10*days,           190*days, units(), RR0_mandatory, RR0_voluntary, lockeddown, migrations, seed)
    # release_31_may   = lockdown(10*days + 4*weeks, 190*days, units(), RR0_mandatory, RR0_voluntary, lockeddown, migrations, seed)
    # adaptive_control = None
