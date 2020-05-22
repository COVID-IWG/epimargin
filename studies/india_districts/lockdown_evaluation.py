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
from etl import district_migration_matrices, get_time_series, load_all_data


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

if __name__ == "__main__":
    root = cwd()
    data = root/"data"

    # model details 
    gamma      = 0.2
    prevalence = 1
    total_time = 90 * days 
    release_date = pd.to_datetime("May 31, 2020")
    lockdown_period = (release_date - pd.to_datetime("today")).days

    states = ["Telangana", "Maharashtra", "Andhra Pradesh", "Tamil Nadu", "Madhya Pradesh", "Punjab", "Gujarat", "Kerala"][:4]
    
    # use gravity matrix for states after 2001 census 
    new_state_data_paths = { 
        "Telangana": (data/"telangana.json", data/"telangana_pop.csv")
    }

    # run rolling regressions on historical national case data 
    dfn = load_all_data(
        v3_paths = (data/"raw_data1.csv", data/"raw_data2.csv"), 
        v4_paths = (data/"raw_data3.csv", data/"raw_data4.csv")
    )
    data_recency = str(dfn["date_announced"].max()).split()[0]
    tsn = get_time_series(dfn)
    grn = run_regressions(tsn, window = 5, infectious_period = 1/gamma)

    # disaggregate down to states
    dfs = {state: dfn[dfn["detected_state"] == state] for state in states}
    tss = {state: get_time_series(cases) for (state, cases) in dfs.items()}
    for (_, ts) in tss.items():
        ts['Hospitalized'] *= prevalence
    grs = {state: run_regressions(timeseries, window = 5, infectious_period = 1/gamma) for (state, timeseries) in tss.items() if len(timeseries) > 5}
    
    # voluntary and mandatory reproductive numbers
    Rvn = np.mean(grn["2020-03-24":"2020-03-31"].R)
    Rmn = np.mean(grn["2020-04-01":].R)
    Rvs = {s: np.mean(grs[s]["2020-03-24":"2020-03-31"].R) if s in grs else Rvn for s in states}
    Rms = {s: np.mean(grs[s]["2020-04-01":].R)             if s in grs else Rmn for s in states}

    # voluntary and mandatory distancing rates 
    Bvs = {s: R * gamma for (s, R) in Rvs.items()}
    Bms = {s: R * gamma for (s, R) in Rms.items()}

    migration_matrices = district_migration_matrices(data/"Migration Matrix - District.csv", states = states)

    # seed range 
    si, sf = 0, 1000

    for state in states: 
        if state in new_state_data_paths.keys():
            districts, populations, migrations = gravity_matrix(*new_state_data_paths[state])
        else: 
            districts, populations, migrations = migration_matrices[state]
        df_state = dfs[state]
        dfd = {district: df_state[df_state["detected_district"] == district] for district in districts}
        tsd = {district: get_time_series(cases) for (district, cases) in dfd.items()}
        for (_, ts) in tsd.items():
            if 'Hospitalized' in ts:
                ts['Hospitalized'] *= prevalence
        grd = {district: run_regressions(timeseries, window = 5, infectious_period = 1/gamma) for (district, timeseries) in tsd.items() if len(timeseries) > 5}
    
        Rv = {district: np.mean(grd[district]["2020-03-24":"2020-03-31"].R) if district in grd.keys() else Rvs[state] for district in districts}
        Rm = {district: np.mean(grd[district]["2020-04-01":].R)             if district in grd.keys() else Rms[state] for district in districts}

        # # fil in missing values 
        for mapping, default in ((Rv, Rvs[state]), (Rm, Rms[state])):
            for key in mapping:
                if np.isnan(mapping[key]):
                    mapping[key] = default
        

        # simulation_results = [
        #     run_policies(migrations, districts, populations, tsd, Rm, Rv, gamma, seed, initial_lockdown = lockdown_period, total_time = total_time) 
        #     for seed in tqdm(range(si, sf))
        # ]

        # plot_simulation_range(simulation_results, ["31 May Release", "Adaptive Controls"], get_time_series(df_state).Hospitalized)\
        #     .title(f"{state} Policy Scenarios: Projected Infections over Time")\
        #     .xlabel("Date")\
        #     .ylabel("Number of net new infections")\
        #     .annotate(f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {(5, 5, 5)}, data from {data_recency}")\
        #     .show()
