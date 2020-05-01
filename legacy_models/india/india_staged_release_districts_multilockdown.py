from itertools import product
from pathlib import Path
from typing import Dict, Sequence

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from india.growthratefit import *
from model import Model, ModelUnit

# sns.set_style("whitegrid")
# sns.set_palette("bright")
sns.set(style = "whitegrid", palette = "bright", font="Fira Code")
sns.despine()

def create_empirical_migration_matrix(
    census_path:     Path, 
    migrations_path: Path, 
    output_path    : Path) -> pd.DataFrame:
    # load in census metadata 
    census = pd.read_csv(census_path, use_cols = "District")
    splits = census["District"].str.split(",", expand=True)
    # load in observed migrations 

def load_population_data(pop_path: Path) -> pd.DataFrame:
    return pd.read_csv(pop_path, names = ["name", "pop"])\
             .sort_values("name")

def load_migration_matrix(matrix_path: Path, populations: np.array) -> np.matrix:
    M  = np.loadtxt(matrix_path, delimiter=',') # read in raw data
    M *= populations[:,  None]                  # weight by population
    M /= M.sum(axis = 0)                        # normalize
    return M 

def load_district_migration_matrices(
    matrix_path: Path, 
    states: Sequence[str] = ["Maharashtra"]) -> Dict[str, np.matrix]:
    # states: Sequence[str] = ["Maharashtra", "Delhi", "Punjab", "Tamil Nadu", "Kerala"]) -> Dict[str, np.matrix]:
    mm = pd.read_csv(matrix_path)
    aggregations = dict()
    for col in  ['D_StateCensus2001', 'D_DistrictCensus2001', 'O_StateCensus2001', 'O_DistrictCensus2001']:
        mm[col] = mm[col].str.title().str.replace("&", "and")
    for state in  states:
        mm_state = mm[(mm.D_StateCensus2001 == state) & (mm.O_StateCensus2001 == state)]
        pivot    = mm_state.pivot(index = "D_DistrictCensus2001", columns = "O_DistrictCensus2001", values = "NSS_STMigrants").fillna(0)
        M  = np.matrix(pivot)
        Mn = M/M.sum(axis = 0)
        Mn[np.isnan(Mn)] = 0
        aggregations[state] = (
            pivot.index, 
            mm_state.groupby("O_DistrictCensus2001")["O_Population_2011"].agg(lambda x: list(x)[0]).values, 
            Mn
        )
    return aggregations 

if __name__ == "__main__":
    try: 
        root = Path(__file__).resolve().parent
    except NameError:
        root = Path(".").resolve()
    
    data = root/"india"
    figs = root/"figures"
    
    gamma = 1/5.0 
    prevalence = 1

    # run rolling regressions on historical case data 
    df_natl = load_data(data/"india_case_data_23_4_resave.csv", reduced = True, schema = v2)
    
    # can't do anything about this :(
    df_natl.dropna(subset = ["detected district"], inplace = True)

    df_natl["district"] = df_natl["detected district"].fillna("unknown") + "_" + df_natl["state code"]
    ts_natl = get_time_series(df_natl)
    gr_natl = run_regressions(ts_natl, window = 7, infectious_period = 5)
    
    # states = ['Andhra Pradesh', 'Uttar Pradesh', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'West Bengal', 'Kerala', 'Gujarat'][1:2]
    # states = df_natl["detected state"].unique()
    states = ['Uttar Pradesh', 'Punjab', 'Tamil Nadu', 'Kerala']

    # states = [_ for _ in df_natl["detected state"].dropna().unique() if _ not in ("Ladakh", "Uttarakhand")]
    dfs = {state: df_natl[df_natl["detected state"] == state] for state in states}
    tss = {state: get_time_series(state_data) for (state, state_data) in dfs.items()}
    for (_, ts) in tss.items():
        ts['Hospitalized'] *= prevalence
    grs = {state: run_regressions(state_data, window = 3, infectious_period = 5) for (state, state_data) in tss.items() if len(state_data) > 3}

    R_vol_natl  = np.mean(gr_natl.iloc[21:28].R)
    R_mand_natl = np.mean(gr_natl.iloc[28:-1].R)

    # get reproductive rate in realized policy regimes: pre- and post-national lockdown 
    R_vol_states  = {state: np.mean(grs[state].iloc[14:22].R) if state in grs else R_vol_natl  for state in states}
    R_mand_states = {state: np.mean(grs[state].iloc[22:-1].R) if state in grs else R_mand_natl for state in states}

    beta_vol_states  = {state: R * gamma  for  (state, R) in R_vol_states.items()}
    beta_mand_states = {state: R * gamma  for  (state, R) in R_mand_states.items()}

    migration_matrices = load_district_migration_matrices(data/"Migration Matrix - District.csv", states = states)
    for state in states:
        districts, populations, migrations = migration_matrices[state]
        df_state = dfs[state]
        dfd = {district: df_state[df_state["detected district"] == district] for district in districts}
        tsd = {district: get_time_series(district_data) for (district, district_data) in  dfd.items()}
        for (_, ts) in tsd.items():
            if 'Hospitalized' in ts:
                ts['Hospitalized'] *= prevalence
        grd = {district: run_regressions(district_data, window = 3, infectious_period = 5) for (district, district_data) in tsd.items() if len(district_data) > 3}

        # print({k: len(v) for (k, v) in tsd.items()})

        R_vol  = {district: np.mean(grd[district]["2020-03-24":"2020-03-31"].R) if district in grd.keys() else R_vol_states[state]  for district in districts}
        R_mand = {district: np.mean(grd[district]["2020-04-01":].R)             if district in grd.keys() else R_mand_states[state] for district in districts}

        for mapping, default in ((R_vol, R_vol_states[state]), (R_mand, R_mand_states[state])):
            for key in mapping:
                if np.isnan(mapping[key]):
                    mapping[key] = default

        beta_vol  = { district: R * gamma for (district, R) in R_vol.items()  }
        beta_mand = { district: R * gamma for (district, R) in R_mand.items() }

        def get_units(): 
            return [ModelUnit(
                name = district, 
                population = populations[i], 
                I0  = tsd[district].iloc[-1]['Hospitalized'] if not tsd[district].empty and 'Hospitalized' in tsd[district].iloc[-1] else 0,
                R0  = tsd[district].iloc[-1]['Recovered']    if not tsd[district].empty and 'Recovered'    in tsd[district].iloc[-1] else 0,
                D0  = tsd[district].iloc[-1]['Deceased']     if not tsd[district].empty and 'Deceased'     in tsd[district].iloc[-1] else 0,
                RR0 = R_mand[district]) 
            for (i, district) in enumerate(districts)]

        seed = 11 
        lockdown = np.zeros(migrations.shape)

        # policy A: end lockdown on May 3 
        np.random.seed(seed)
        model_A_1 = Model(num_days = 10, units = get_units(), migrations = lockdown).run()
        for unit in model_A_1: 
            unit.RR0 = R_vol[unit.name]
        model_A_2 = Model(num_days = 180, units = model_A_1.units, migrations = migrations).run()
        model_A_3 = Model(num_days = 28, units = model_A_1.units, migrations = migrations).run()

        # policy B: end lockdown May 31
        np.random.seed(seed)
        model_B_1 = Model(num_days = 38, units = get_units(), migrations = lockdown).run()
        for unit in model_B_1: 
            unit.RR0 = R_vol[unit.name]
        model_B_2 = Model(num_days = 152, units = model_B_1.units, migrations = migrations).run()

        # policy C: end lockdown Jun 30 
        np.random.seed(seed)
        model_C_1 = Model(num_days = 68, units = get_units(), migrations = lockdown).run()
        for unit in model_C_1: 
            unit.RR0 = R_vol[unit.name]
        model_C_2 = Model(num_days = 122, units = model_C_1.units, migrations = migrations).run()

        # policy C: end lockdown July 31
        np.random.seed(seed)
        model_D_1 = Model(num_days = 99, units = get_units(), migrations = lockdown).run()
        for unit in model_D_1: 
            unit.RR0 = R_vol[unit.name]
        model_D_2 = Model(num_days = 91, units = model_D_1.units, migrations = migrations).run()

        fig = plt.figure()
        plt.plot(model_A_2.aggregate("I"), label = "03 May", figure = fig) 
        plt.plot(model_B_2.aggregate("I"), label = "31 May", figure = fig) 
        plt.plot(model_C_2.aggregate("I"), label = "30 Jun", figure = fig) 
        plt.plot(model_D_2.aggregate("I"), label = "31 Jul", figure = fig) 
        plt.suptitle(state) 
        plt.xlabel("Days Since April 23") 
        plt.ylabel("Number of Infections") 
        plt.legend() 
        plt.tight_layout()
        plt.show()