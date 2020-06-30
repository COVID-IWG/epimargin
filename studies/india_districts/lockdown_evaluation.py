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
from etl import download_data, district_migration_matrices, get_time_series, load_all_data, replace_district_names, load_migration_data, load_populations


def get_model(districts, populations, timeseries, seed = 0):
    units = [ModelUnit(
        name       = district, 
        population = populations[i],
        I0  = timeseries.loc[district].iloc[-1]['Hospitalized'] if not timeseries.loc[district].empty and 'Hospitalized' in timeseries.loc[district].iloc[-1] else 0,
        R0  = timeseries.loc[district].iloc[-1]['Recovered']    if not timeseries.loc[district].empty and 'Recovered'    in timeseries.loc[district].iloc[-1] else 0,
        D0  = timeseries.loc[district].iloc[-1]['Deceased']     if not timeseries.loc[district].empty and 'Deceased'     in timeseries.loc[district].iloc[-1] else 0,
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

    states = ["Maharashtra", "Andhra Pradesh", "Tamil Nadu", "Madhya Pradesh", "Punjab", "Gujarat", "Kerala"]
    
    # use gravity matrix for states after 2001 census 
    new_state_data_paths = { 
        "Telangana": (data/"telangana.json", data/"telangana_pop.csv")
    }

    # define data versions for api files
    paths = { "v3": ["raw_data1.csv", "raw_data2.csv"],
              "v4": ["raw_data3.csv", "raw_data4.csv",
                     "raw_data5.csv", "raw_data6.csv"] } 

    # download data from india covid 19 api
    for target in paths['v3'] + paths['v4']:
        download_data(data, target)

    # run rolling regressions on historical national case data 
    dfn = load_all_data(
        v3_paths = [data/filepath for filepath in paths['v3']], 
        v4_paths = [data/filepath for filepath in paths['v4']]
    )

    # load csv mapping 2011 districts to current district names and rename districts to 2011 spellings
    district_matches = pd.read_csv(data/"india_district_matches.csv")
    dfn_renamed = replace_district_names(dfn, district_matches)

    # load migration matrix data
    migration_data = load_migration_data(data/"Migration Matrix - 2011 District.csv")

    # get list of states and districts used in model (without 'other state' etc)
    current_state_districts = get_current_state_districts(migration_data, district_matches)

    # redistribute missing cases based on district/state populations 
    populations = load_populations(data/"india_district_populations.csv - final.csv")
    dfn_redistributed = redistribute_missing_cases(dfn_renamed, current_state_districts, list(new_state_data_paths.keys()), populations)

    data_recency = str(dfn_redistributed["date_announced"].max()).split()[0]
    tsn = get_time_series(dfn_redistributed)
    grn = run_regressions(tsn, window = 5, infectious_period = 1/gamma)

    # disaggregate down to states
    tss = get_time_series(dfn_redistributed, 'detected_state')
    tss['Hospitalized'] *= prevalence

    grs = tss.groupby(level=0).apply(lambda x: run_regressions(x, window = 5, infectious_period = 1/gamma) if len(x) > 5 else None)
    
    # voluntary and mandatory reproductive numbers
    Rvn = np.mean(grn["2020-03-24":"2020-03-31"].R)
    Rmn = np.mean(grn["2020-04-01":].R)

    Rvs = {s: np.mean(grs.loc[s].loc["2020-03-24":"2020-03-31"].R) if s in grs.index else Rvn for s in states}
    Rms = {s: np.mean(grs.loc[s].loc["2020-04-01":].R)             if s in grs.index else Rmn for s in states}

    # voluntary and mandatory distancing rates 
    Bvs = {s: R * gamma for (s, R) in Rvs.items()}
    Bms = {s: R * gamma for (s, R) in Rms.items()}

    migration_matrices = district_migration_matrices(migration_data, states = states)

    # seed range 
    si, sf = 0, 1000

    for state in states: 
        if state in new_state_data_paths.keys():
            districts, populations, migrations = gravity_matrix(*new_state_data_paths[state])
        else: 
            districts, populations, migrations = migration_matrices[state]

        df_state = dfn_redistributed[dfn_redistributed['detected_state'] == state]

        # only keep district names that are present in both migration and api data
        districts = list(set(districts).intersection(set(df_state['detected_district'])))

        tsd = get_time_series(df_state, 'detected_district') 
        tsd['Hospitalized'] *= prevalence

        grd = tsd.groupby(level=0).apply(lambda x: run_regressions(x.reset_index(level=0, drop=True), window = 5, infectious_period = 1/gamma) if len(x) > 5 else None)
    
        Rv = {district: np.mean(grd.loc[district].loc["2020-03-24":"2020-03-31"].R) if district in grd.index else Rvs[state] for district in districts}
        Rm = {district: np.mean(grd.loc[district].loc["2020-04-01":].R)             if district in grd.index else Rms[state] for district in districts}

        # fill in missing values 
        for mapping, default in ((Rv, Rvs[state]), (Rm, Rms[state])):
            for key in mapping:
                if np.isnan(mapping[key]):
                    mapping[key] = default
        
        projections = []
        for district in districts:
            try:
                estimate = grd.loc[district].loc[grd.loc[district].R.last_valid_index()]
                projections.append((district, estimate.R, estimate.R + estimate.gradient*7))
            except KeyError:
                projections.append((district, np.NaN, np.NaN))
        pd.DataFrame(projections, columns = ["district", "R", "Rproj"]).to_csv(data/(state + ".csv")) 

        simulation_results = [
            run_policies(migrations, districts, populations, tsd, Rm, Rv, gamma, seed, initial_lockdown = lockdown_period, total_time = total_time) 
            for seed in tqdm(range(si, sf))
        ]

        plot_simulation_range(simulation_results, ["31 May Release", "Adaptive Controls"], get_time_series(df_state).Hospitalized)\
            .title(f"{state} Policy Scenarios: Projected Infections over Time")\
            .xlabel("Date")\
            .ylabel("Number of net new infections")\
            .annotate(f"stochastic parameter range: ({si}, {sf}), infectious period: {1/gamma} days, smoothing window: {(5, 5, 5)}, data from {data_recency}")\
            .show()
