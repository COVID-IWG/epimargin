from itertools import product
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive.estimators import rollingOLS as run_regressions
from adaptive.model import Model, ModelUnit
from adaptive.plotting import plot_curve, gantt_chart
from adaptive.utils import *
from etl import district_migration_matrices, get_time_series, load_data, v2


def get_model(districts, populations, timeseries, seed = 0):
    units = [ModelUnit(
        name       = district, 
        population = populations[i],
        I0  = timeseries[district].iloc[-1]['Hospitalized'] if not timeseries[district].empty and 'Hospitalized' in timeseries[district].iloc[-1] else 0,
        R0  = timeseries[district].iloc[-1]['Recovered']    if not timeseries[district].empty and 'Recovered'    in timeseries[district].iloc[-1] else 0,
        D0  = timeseries[district].iloc[-1]['Deceased']     if not timeseries[district].empty and 'Deceased'     in timeseries[district].iloc[-1] else 0,
    ) for (i, district) in enumerate(districts)]
    return Model(units, random_seed = seed)

# lockdown policy of different lengths
def simulate_lockdown(model: Model, lockdown_period: int, total_time: int, RR0_mandatory: Dict[str, float], RR0_voluntary: Dict[str, float], lockdown: np.matrix, migrations: np.matrix) -> Model:
    return model.set_parameters(RR0 = RR0_mandatory)\
        .run(lockdown_period,  migrations = lockdown)\
        .set_parameters(RR0 = RR0_voluntary)\
        .run(total_time - lockdown_period, migrations = migrations)

# policy C: adaptive release
def simulate_adaptive_control(model: Model, initial_run: int, total_time: int, lockdown: np.matrix, migrations: np.matrix, beta_v: Dict[str, float], beta_m: Dict[str, float], evaluation_period = 2*weeks):
    n = len(model)
    model.run(initial_run, lockdown)
    days_run = initial_run
    gantt = []
    last_category = dict()
    while days_run < total_time:
        Gs, Ys, Os, Rs = set(), set(), set(), set()
        category_transitions = {}
        for (i, unit) in enumerate(model):
            latest_RR = unit.RR[-1]
            if latest_RR < 1: 
                Gs.add(i)
                beta_cat = 0
            else: 
                if days_run < evaluation_period: # force first period to be lockdown
                    Rs.add(i)
                    beta_cat = 3
                else: 
                    if latest_RR < 1.5: 
                        Ys.add(i)
                        beta_cat = 1
                    elif latest_RR < 2: 
                        Os.add(i)
                        beta_cat = 2
                    else:
                        Rs.add(i)
                        beta_cat = 3
            if unit.name not in last_category:
                last_category[unit.name] = beta_cat
            else: 
                old_beta_cat = last_category[unit.name]
                if old_beta_cat != beta_cat:
                    if beta_cat < old_beta_cat and beta_cat != (old_beta_cat - 1): # force gradual release
                        beta_cat = old_beta_cat - 1
                    category_transitions[unit.name] = beta_cat
                    last_category[unit.name] = beta_cat 
            gantt.append([unit.name, days_run, beta_cat, max(0, latest_RR)])

        for (unit_name, beta_cat) in category_transitions.items(): 
            unit =  model[unit_name]
            new_beta = beta_v[unit.name] - (beta_cat * (beta_v[unit.name] - beta_m[unit.name])/3.0)                
            unit.beta[-1] = new_beta
            unit.RR0 = new_beta * unit.gamma

        phased_migration = migrations.copy()
        for (i, j) in product(range(n), range(n)):
            if i not in Gs or j not in Gs:
                phased_migration[i, j] = 0
        model.run(evaluation_period, phased_migration)
        days_run += evaluation_period

    model.gantt = gantt 
    return model 

if __name__ == "__main__":
    # set up folders
    root = cwd()
    data = root/"data"
    figs = root/"figs"
    if not figs.exists():
        figs.mkdir()

    # simulation parameters 
    seed       = 0
    total_time = 190 * days 
    states     = ['Andhra Pradesh', 'Uttar Pradesh', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'West Bengal', 'Kerala', 'Gujarat'][2:3]
    
    # model details 
    gamma      = 0.2
    prevalence = 1

    # run rolling regressions on historical national case data 
    dfn = load_data(data/"india_case_data_23_4_resave.csv", reduced = True, schema = v2).dropna(subset = ["detected district"]) # can't do anything about this :( 
    tsn = get_time_series(dfn)
    grn = run_regressions(tsn, window = 7, infectious_period = 1/gamma)

    # disaggregate down to states
    dfs = {state: dfn[dfn["detected state"] == state] for state in states}
    tss = {state: get_time_series(cases) for (state, cases) in dfs.items()}
    for (_, ts) in tss.items():
        ts['Hospitalized'] *= prevalence
    grs = {state: run_regressions(timeseries, window = 3, infectious_period = 1/gamma) for (state, timeseries) in tss.items() if len(timeseries) > 3}
    
    # voluntary and mandatory reproductive numbers
    Rvn = np.mean(grn["2020-03-24":"2020-03-31"].R)
    Rmn = np.mean(grn["2020-04-01":].R)
    Rvs = {s: np.mean(grs[s]["2020-03-24":"2020-03-31"].R) if s in grs else Rvn for s in states}
    Rms = {s: np.mean(grs[s]["2020-04-01":].R)             if s in grs else Rmn for s in states}

    # voluntary and mandatory distancing rates 
    Bvs = {s: R * gamma for (s, R) in Rvs.items()}
    Bms = {s: R * gamma for (s, R) in Rms.items()}

    migration_matrices = district_migration_matrices(data/"Migration Matrix - District.csv", states = states)

    for state in states: 
        districts, populations, migrations = migration_matrices[state]
        df_state = dfs[state]
        dfd = {district: df_state[df_state["detected district"] == district] for district in districts}
        tsd = {district: get_time_series(cases) for (district, cases) in  dfd.items()}
        for (_, ts) in tsd.items():
            if 'Hospitalized' in ts:
                ts['Hospitalized'] *= prevalence
        grd = {district: run_regressions(timeseries, window = 3, infectious_period = 1/gamma) for (district, timeseries) in tsd.items() if len(timeseries) > 3}
    
        Rv = {district: np.mean(grd[district]["2020-03-24":"2020-03-31"].R) if district in grd.keys() else Rvs[state] for district in districts}
        Rm = {district: np.mean(grd[district]["2020-04-01":].R)             if district in grd.keys() else Rms[state] for district in districts}

        # # fil in missing values 
        # for mapping, default in ((Rv, Rvs[state]), (Rm, Rms[state])):
        #     for key in mapping:
        #         if np.isnan(mapping[key]):
        #             mapping[key] = default

        # # policy scenarios: 
        # lockdown = np.zeros(migrations.shape)

        # ## release lockdown on 03 May 
        # release_03_may = get_model(districts, populations, tsd, seed)
        # simulate_lockdown(release_03_may, 
        #     lockdown_period = 10*days, 
        #     total_time      = total_time, 
        #     RR0_mandatory   = Rm,              RR0_voluntary = Rv, 
        #     lockdown        = lockdown.copy(), migrations    = migrations)

        # ## release lockdown on 31 May 
        # release_31_may = get_model(districts, populations, tsd, seed)
        # simulate_lockdown(release_31_may, 
        #     lockdown_period = 10*days + 4*weeks, 
        #     total_time      = total_time, 
        #     RR0_mandatory   = Rm,              RR0_voluntary = Rv, 
        #     lockdown        = lockdown.copy(), migrations    = migrations)

        # ## adaptive release starting 03 may 
        # beta_v = {district: R * gamma for (district, R) in Rv.items()}
        # beta_m = {district: R * gamma for (district, R) in Rm.items()}
        # adaptive = get_model(districts, populations, tsd, seed).set_parameters(RR0 = Rm)
        # simulate_adaptive_control(adaptive, 10*days, total_time, lockdown, migrations, beta_v, beta_m)
        # plot_curve(
        #     [release_03_may, release_31_may, adaptive], 
        #     ["03 May Release", "31 May Release", "Adaptive Release"], 
        #     title = state, xlabel = "Days Since April 23", ylabel = "Infections", subtitle = None
        # )

        # plt.show()

        # gantt_chart(adaptive.gantt, "Days Since April 23", "Release Strategy")
        
        # plt.show()
        
